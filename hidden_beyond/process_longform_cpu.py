#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def run(cmd):
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def ffprobe_duration(path:Path)->float:
    return float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1",str(path)
    ],text=True).strip())

def has_audio(path:Path)->bool:
    p=subprocess.run([
        "ffprobe","-v","error","-select_streams","a:0",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(path)
    ],capture_output=True,text=True)
    return p.returncode==0 and "audio" in p.stdout

def clean(s:str)->str:
    s=re.sub(r"\s+"," ",str(s)).strip()
    s=re.sub(r"\s+([,.;:!?])",r"\1",s)
    return s

def ts(sec:float)->str:
    ms=max(0,int(round(sec*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def wav_info(path:Path):
    with wave.open(str(path),"rb") as w:
        return w.getframerate(),w.getsampwidth(),w.getnchannels(),w.getnframes()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()

    src=Path(args.input)
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    work=outdir/"work"
    work.mkdir(exist_ok=True)
    audio=work/"source.wav"
    processed=outdir/"processed.mp4"
    srt=outdir/"vi.srt"
    meta=outdir/"metadata.json"

    if not src.exists() or src.stat().st_size<1_000_000:
        raise SystemExit("Full source file missing")
    if not has_audio(src):
        raise SystemExit("Full source has no audio")
    source_duration=ffprobe_duration(src)
    print(f"CPU_LONGFORM_SOURCE_READY bytes={src.stat().st_size} duration={source_duration:.3f}",flush=True)

    run([
        "ffmpeg","-y","-v","error","-i",str(src),
        "-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio)
    ])

    # Fast multilingual ASR on CPU. Small gives a practical first-video turnaround
    # while preserving the complete story timeline.
    whisper=WhisperModel("small",device="cpu",compute_type="int8",cpu_threads=max(2,os.cpu_count() or 2))
    seg_iter,info=whisper.transcribe(
        str(audio),
        language="zh",
        vad_filter=True,
        beam_size=4,
        condition_on_previous_text=True,
    )
    raw=[]
    for s in seg_iter:
        text=clean(s.text)
        if text:
            raw.append({"start":float(s.start),"end":float(s.end),"text":text})
    if len(raw)<20:
        raise SystemExit(f"ASR returned too few segments: {len(raw)}")
    del whisper
    gc.collect()

    # Merge very short neighbouring cues to reduce translation/TTS overhead.
    segments=[]
    for s in raw:
        if not segments:
            segments.append(dict(s))
            continue
        cur=segments[-1]
        gap=s["start"]-cur["end"]
        combined=clean(cur["text"]+" "+s["text"])
        span=s["end"]-cur["start"]
        if gap<=0.35 and span<=6.5 and len(combined)<=68:
            cur["end"]=s["end"]
            cur["text"]=combined
        else:
            segments.append(dict(s))
    print(f"CPU_LONGFORM_ASR_READY raw={len(raw)} merged={len(segments)} lang={info.language} prob={float(info.language_probability or 0):.3f}",flush=True)

    model_name="Helsinki-NLP/opus-mt-zh-vi"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
    model.eval()
    texts=[s["text"] for s in segments]+[str(cfg["title"])]
    translations=[]
    batch_size=8
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            inp=tokenizer(batch,return_tensors="pt",padding=True,truncation=True,max_length=256)
            generated=model.generate(
                **inp,
                max_new_tokens=160,
                num_beams=4,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            translations.extend(tokenizer.batch_decode(generated,skip_special_tokens=True))
            if start and start%(batch_size*20)==0:
                print(f"CPU_LONGFORM_TRANSLATE_PROGRESS {start}/{len(texts)}",flush=True)
    translations=[clean(x) for x in translations]
    if len(translations)!=len(texts) or any(not x for x in translations):
        raise SystemExit("Translation output mismatch")
    for s,vi in zip(segments,translations[:-1],strict=True):
        s["vi"]=vi
    translated_title=translations[-1]
    del model,tokenizer
    gc.collect()
    print(f"CPU_LONGFORM_TRANSLATION_READY segments={len(segments)} title={translated_title}",flush=True)

    voice_name="vi_VN-vais1000-medium"
    voices=work/"piper-voices"
    voices.mkdir(exist_ok=True)
    run([sys.executable,"-m","piper.download_voices",voice_name,"--download-dir",str(voices)])
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    voice=PiperVoice.load(str(voices/(voice_name+".onnx")),use_cuda=False)
    syn=SynthesisConfig()

    ttsdir=work/"tts"
    ttsdir.mkdir(exist_ok=True)
    # 22.05 kHz mono, 16-bit ~= 198 MB for a 75-minute program.
    rate=22050
    width=2
    channels=1
    total_frames=max(1,int(source_duration*rate))
    canvas=bytearray(total_frames*width*channels)
    timing=[]

    for i,s in enumerate(segments,1):
        wav=ttsdir/f"{i:05d}.wav"
        with wave.open(str(wav),"wb") as wf:
            voice.synthesize_wav(s["vi"],wf,syn_config=syn)
        sr,sw,ch,frames=wav_info(wav)
        if (sr,sw,ch)!=(rate,width,channels):
            # Normalize only when needed.
            norm=ttsdir/f"{i:05d}-n.wav"
            run(["ffmpeg","-y","-v","error","-i",str(wav),"-ar",str(rate),"-ac","1","-c:a","pcm_s16le",str(norm)])
            wav=norm
            sr,sw,ch,frames=wav_info(wav)
        with wave.open(str(wav),"rb") as wf:
            data=wf.readframes(wf.getnframes())
        natural=frames/rate
        slot=max(0.25,float(s["end"])-float(s["start"])-0.04)

        # For a first validation video keep processing fast: place speech on cue,
        # clipping only the rare overlong utterance rather than spawning hundreds
        # of separate ffmpeg time-stretch jobs.
        max_bytes=int(slot*rate)*width*channels
        if len(data)>max_bytes:
            data=data[:max_bytes]
            fitted=slot
            clipped=True
        else:
            fitted=len(data)/(rate*width*channels)
            clipped=False

        pos=int(float(s["start"])*rate)*width*channels
        end=min(len(canvas),pos+len(data))
        if pos<len(canvas):
            canvas[pos:end]=data[:end-pos]
        timing.append({
            "index":i,
            "start":round(float(s["start"]),3),
            "end":round(float(s["end"]),3),
            "slot_seconds":round(slot,3),
            "tts_seconds":round(natural,3),
            "placed_seconds":round(fitted,3),
            "clipped":clipped,
        })
        if i%50==0:
            print(f"CPU_LONGFORM_TTS_PROGRESS {i}/{len(segments)}",flush=True)

    voice_path=work/"vietnamese-voice.wav"
    with wave.open(str(voice_path),"wb") as wf:
        wf.setnchannels(channels); wf.setsampwidth(width); wf.setframerate(rate)
        wf.writeframes(bytes(canvas))

    mix=(
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,highpass=f=35[orig];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.22,asplit=2[sc][voice];"
        "[orig][sc]sidechaincompress=threshold=0.009:ratio=16:attack=8:release=280:makeup=1[duck];"
        "[duck]volume=0.42[bed];"
        "[bed][voice]amix=inputs=2:weights='1 1':normalize=0,alimiter=limit=0.95[outa]"
    )
    run([
        "ffmpeg","-y","-v","error","-i",str(src),"-i",str(voice_path),
        "-filter_complex",mix,
        "-map","0:v:0","-map","[outa]",
        "-c:v","copy","-c:a","aac","-b:a","160k",
        "-movflags","+faststart","-shortest",str(processed)
    ])

    if not processed.exists() or processed.stat().st_size<1_000_000 or not has_audio(processed):
        raise SystemExit("Processed video verification failed")
    final_duration=ffprobe_duration(processed)
    if abs(final_duration-source_duration)>3.0:
        raise SystemExit(f"Duration mismatch source={source_duration} final={final_duration}")

    lines=[]
    for i,s in enumerate(segments,1):
        lines += [str(i),f"{ts(s['start'])} --> {ts(s['end'])}",s["vi"],""]
    srt.write_text("\n".join(lines),encoding="utf-8")

    sha=hashlib.sha256()
    with processed.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            sha.update(chunk)
    clipped=sum(1 for x in timing if x["clipped"])
    report={
        "ok":True,
        "source_id":cfg["source_id"],
        "series_id":cfg["series_id"],
        "source_video_id":cfg["source_video_id"],
        "episode_number":cfg["episode_number"],
        "source_url":cfg["source_url"],
        "source_title":cfg["title"],
        "translated_title":translated_title,
        "duration_seconds":round(final_duration,3),
        "size_bytes":processed.stat().st_size,
        "sha256":sha.hexdigest(),
        "segments":len(segments),
        "clipped_segments":clipped,
        "asr_model":"faster-whisper-small-int8",
        "detected_language":info.language,
        "language_probability":float(info.language_probability or 0),
        "translation_model":model_name,
        "tts_voice":"piper-vi_VN-vais1000-medium",
        "original_visuals_preserved":True,
        "original_audio_preserved_and_ducked":True,
    }
    meta.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CPU_LONGFORM_PROCESS_OK",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

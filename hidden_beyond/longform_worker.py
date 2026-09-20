#!/usr/bin/env python3
from __future__ import annotations

# __CONFIG_INJECT__

import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

WORK=Path("/kaggle/working")
SOURCE=WORK/"source.mp4"
AUDIO=WORK/"source.wav"
OUT=WORK/"processed.mp4"
SRT=WORK/"vi.srt"
META=WORK/"metadata.json"
SEGDIR=WORK/"tts"
SEGDIR.mkdir(parents=True,exist_ok=True)

def run(cmd, **kwargs):
    print("+", " ".join(map(str,cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)

def pip_install(*pkgs):
    run([sys.executable,"-m","pip","install","--quiet",*pkgs])

pip_install(
    "yt-dlp>=2026.1",
    "faster-whisper>=1.1,<2",
    "transformers<5",
    "accelerate<2",
    "sentencepiece",
    "sacremoses",
    "zerotts>=0.1.5,<0.2",
    "piper-tts>=1.3,<2",
)

import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def ffprobe_duration(path:Path)->float:
    return float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1",str(path)
    ],text=True).strip())

def has_audio(path:Path)->bool:
    p=subprocess.run([
        "ffprobe","-v","error","-select_streams","a:0",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(path)
    ],text=True,capture_output=True)
    return p.returncode==0 and "audio" in p.stdout

def ts(sec:float)->str:
    ms=max(0,int(round(sec*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def clean_text(s:str)->str:
    s=re.sub(r"\s+"," ",s).strip()
    s=re.sub(r"\s+([,.;:!?])",r"\1",s)
    return s

print("HB_LONGFORM_CONFIG",json.dumps(CONFIG,ensure_ascii=False),flush=True)

# Download one full source item. 480p keeps the first validation upload manageable
# while preserving the complete story and original visual track.
run([
    sys.executable,"-m","yt_dlp",
    "--no-playlist",
    "--retries","5",
    "--fragment-retries","5",
    "-f","bv*[height<=480]+ba/b[height<=480]/b",
    "--merge-output-format","mp4",
    "-o",str(SOURCE),
    CONFIG["source_url"],
])

if not SOURCE.exists() or SOURCE.stat().st_size < 1_000_000:
    raise RuntimeError("source download missing or unexpectedly small")
if not has_audio(SOURCE):
    raise RuntimeError("source video has no audio")

duration=ffprobe_duration(SOURCE)
print("SOURCE_READY",SOURCE.stat().st_size,duration,flush=True)

run([
    "ffmpeg","-y","-v","error","-i",str(SOURCE),
    "-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(AUDIO)
])

# GPU transcription.
device="cuda" if torch.cuda.is_available() else "cpu"
compute="float16" if device=="cuda" else "int8"
whisper=WhisperModel("large-v3-turbo",device=device,compute_type=compute)
seg_iter,info=whisper.transcribe(
    str(AUDIO),
    language="zh",
    vad_filter=True,
    beam_size=5,
    condition_on_previous_text=True,
)
raw=[{"start":float(s.start),"end":float(s.end),"text":clean_text(s.text)}
     for s in seg_iter if clean_text(s.text)]
if len(raw)<10:
    raise RuntimeError(f"too few transcript segments: {len(raw)}")

# Merge adjacent micro-segments to reduce TTS overhead without losing cue order.
merged=[]
for s in raw:
    if not merged:
        merged.append(dict(s)); continue
    cur=merged[-1]
    gap=s["start"]-cur["end"]
    candidate=(cur["text"]+" "+s["text"]).strip()
    span=s["end"]-cur["start"]
    if gap<=0.45 and span<=7.0 and len(candidate)<=72:
        cur["end"]=s["end"]; cur["text"]=candidate
    else:
        merged.append(dict(s))
segments=merged
print("TRANSCRIPT_READY",len(raw),len(segments),info.language,float(info.language_probability or 0),flush=True)

del whisper
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Local GPU translation: Chinese -> Vietnamese using NLLB.
model_name="facebook/nllb-200-distilled-600M"
tokenizer=AutoTokenizer.from_pretrained(model_name,src_lang="zho_Hans")
model=AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model=model.to(device)
model.eval()
target_id=tokenizer.convert_tokens_to_ids("vie_Latn")
texts=[s["text"] for s in segments]+[CONFIG["title"]]
translations=[]
batch_size=18 if device=="cuda" else 4
with torch.inference_mode():
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        inp=tokenizer(batch,return_tensors="pt",padding=True,truncation=True,max_length=256)
        inp={k:v.to(device) for k,v in inp.items()}
        out=model.generate(
            **inp,
            forced_bos_token_id=target_id,
            max_new_tokens=180,
            num_beams=4,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
        )
        translations.extend(tokenizer.batch_decode(out,skip_special_tokens=True))
translations=[clean_text(x) for x in translations]
if len(translations)!=len(texts) or any(not x for x in translations):
    raise RuntimeError("translation output mismatch")
for s,vi in zip(segments,translations[:-1],strict=True):
    s["vi"]=vi
translated_title=translations[-1]
print("TRANSLATION_READY",translated_title,len(segments),flush=True)

del model,tokenizer
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

def wav_duration(path:Path)->float:
    with wave.open(str(path),"rb") as w:
        return w.getnframes()/w.getframerate()

def fit_audio(src:Path,dst:Path,slot:float)->dict:
    original=max(0.01,wav_duration(src))
    target=max(0.25,slot)
    speed=max(0.92,min(1.48,original/target))
    filters=[
        f"atempo={speed:.6f}",
        f"atrim=duration={target:.3f}",
        "highpass=f=70",
        "lowpass=f=11500",
        "afade=t=in:st=0:d=0.02",
    ]
    run([
        "ffmpeg","-y","-v","error","-i",str(src),
        "-af",",".join(filters),
        "-ar","22050","-ac","1","-c:a","pcm_s16le",str(dst)
    ])
    fitted=wav_duration(dst)
    return {"source":original,"fitted":fitted,"speed":speed}

# Primary voice: ZeroTTS hamy. If initialization/probe fails, use local Piper.
voice_mode="zerotts-hamy"
zerotts=None
piper_voice=None
try:
    from zerotts import ZeroTTS
    zerotts=ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
    probe=zerotts.synthesize(
        "Xin chào.",
        voice="hamy",
        cfg_scale=1.0,
        audio_temperature=0.78,
        audio_topk=25,
        audio_topp=0.95,
        audio_repetition_penalty=1.2,
    )
    zerotts.save_audio(probe,str(SEGDIR/"probe.wav"))
    if wav_duration(SEGDIR/"probe.wav")<0.1:
        raise RuntimeError("ZeroTTS probe too short")
except Exception as exc:
    print("ZEROTTS_FALLBACK",repr(exc),flush=True)
    voice_mode="piper-vais1000"
    voice_name="vi_VN-vais1000-medium"
    voices=WORK/"piper-voices"
    voices.mkdir(exist_ok=True)
    run([sys.executable,"-m","piper.download_voices",voice_name,"--download-dir",str(voices)])
    from piper import PiperVoice
    piper_voice=PiperVoice.load(str(voices/(voice_name+".onnx")),use_cuda=False)

def synth(text:str,path:Path):
    if voice_mode=="zerotts-hamy":
        audio=zerotts.synthesize(
            text,
            voice="hamy",
            cfg_scale=1.0,
            audio_temperature=0.78,
            audio_topk=25,
            audio_topp=0.95,
            audio_repetition_penalty=1.2,
        )
        zerotts.save_audio(audio,str(path))
    else:
        with wave.open(str(path),"wb") as w:
            piper_voice.synthesize_wav(text,w)

# Synthesize and duration-fit each translated dialogue cue.
timing=[]
for i,s in enumerate(segments,1):
    rawwav=SEGDIR/f"raw-{i:05d}.wav"
    fitwav=SEGDIR/f"fit-{i:05d}.wav"
    try:
        synth(s["vi"],rawwav)
    except Exception as exc:
        if voice_mode=="zerotts-hamy":
            raise RuntimeError(f"ZeroTTS failed at segment {i}: {exc}")
        raise
    slot=max(0.25,s["end"]-s["start"]-0.04)
    d=fit_audio(rawwav,fitwav,slot)
    s["audio"]=fitwav
    timing.append({"index":i,"start":s["start"],"end":s["end"],"slot":slot,**d})
    if i%25==0:
        print("TTS_PROGRESS",i,"/",len(segments),voice_mode,flush=True)

# Build a single Vietnamese voice track on the original timeline.
with wave.open(str(segments[0]["audio"]),"rb") as w:
    rate=w.getframerate(); width=w.getsampwidth(); channels=w.getnchannels()
frames=max(1,int(duration*rate))
canvas=bytearray(b"\0"*(frames*width*channels))
for s in segments:
    with wave.open(str(s["audio"]),"rb") as w:
        if (w.getframerate(),w.getsampwidth(),w.getnchannels())!=(rate,width,channels):
            raise RuntimeError("TTS wave format mismatch")
        data=w.readframes(w.getnframes())
    pos=int(s["start"]*rate)*width*channels
    end=min(len(canvas),pos+len(data))
    if pos<len(canvas): canvas[pos:end]=data[:end-pos]
voice=WORK/"vietnamese-voice.wav"
with wave.open(str(voice),"wb") as w:
    w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate); w.writeframes(bytes(canvas))

# Mix original SFX/ambience under Vietnamese voice; aggressively duck source speech.
mix=(
    "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,highpass=f=35[orig];"
    "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.18,asplit=2[sc][voice];"
    "[orig][sc]sidechaincompress=threshold=0.010:ratio=14:attack=8:release=260:makeup=1[duck];"
    "[duck]volume=0.48[bed];"
    "[bed][voice]amix=inputs=2:weights='1 1':normalize=0,alimiter=limit=0.95[outa]"
)
run([
    "ffmpeg","-y","-v","error","-i",str(SOURCE),"-i",str(voice),
    "-filter_complex",mix,
    "-map","0:v:0","-map","[outa]",
    "-c:v","copy","-c:a","aac","-b:a","160k",
    "-movflags","+faststart","-shortest",str(OUT)
])

if not OUT.exists() or OUT.stat().st_size<1_000_000 or not has_audio(OUT):
    raise RuntimeError("processed video validation failed")
out_duration=ffprobe_duration(OUT)
if abs(out_duration-duration)>3.0:
    raise RuntimeError(f"duration mismatch source={duration} output={out_duration}")

# External Vietnamese subtitles, kept as evidence and optional future caption upload.
lines=[]
for i,s in enumerate(segments,1):
    lines += [str(i),f"{ts(s['start'])} --> {ts(s['end'])}",s["vi"],""]
SRT.write_text("\n".join(lines),encoding="utf-8")

sha=hashlib.sha256()
with OUT.open("rb") as fh:
    for chunk in iter(lambda:fh.read(1024*1024),b""): sha.update(chunk)
metadata={
    "ok":True,
    "source_video_id":CONFIG["source_video_id"],
    "series_id":CONFIG["series_id"],
    "source_id":CONFIG["source_id"],
    "episode_number":CONFIG["episode_number"],
    "source_url":CONFIG["source_url"],
    "source_title":CONFIG["title"],
    "translated_title":translated_title,
    "duration_seconds":round(out_duration,3),
    "size_bytes":OUT.stat().st_size,
    "sha256":sha.hexdigest(),
    "segments":len(segments),
    "detected_language":info.language,
    "language_probability":float(info.language_probability or 0),
    "translation_model":model_name,
    "tts_voice":voice_mode,
    "video_height_limit":480,
    "original_visuals_preserved":True,
    "original_audio_preserved_and_ducked":True,
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
META.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("HB_LONGFORM_PROCESS_OK",json.dumps(metadata,ensure_ascii=False),flush=True)

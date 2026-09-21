#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np


def run(cmd:list[str]) -> None:
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)


def clean(s:str)->str:
    value=re.sub(r"\s+"," ",str(s)).strip().strip('“”"')
    value=re.sub(r"\s+([,.;:!?])",r"\1",value)
    return value.strip()


def wav_duration(path:Path)->float:
    with wave.open(str(path),"rb") as w:
        return w.getnframes()/w.getframerate()


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    meta_path=Path(args.metadata)
    out_path=Path(args.output)
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    segments=meta.get("timed_segments") or []
    if not segments:
        raise RuntimeError("metadata has no timed_segments")

    style=meta.get("style") or {}
    max_tempo=float(style.get("max_tempo",1.22))
    min_tempo=float(style.get("min_tempo",0.90))
    pitch_ratio=float(style.get("pitch_ratio",1.025))
    max_extra_gap=float(style.get("max_extra_gap",0.24))
    voice_name=str(meta.get("voice_name") or "vi_VN-vais1000-medium")

    work=out_path.parent/"cpu-tts"
    segdir=work/"segments"
    voices=work/"voices"
    segdir.mkdir(parents=True,exist_ok=True)
    voices.mkdir(parents=True,exist_ok=True)

    run([sys.executable,"-m","piper.download_voices",voice_name,"--download-dir",str(voices)])
    from piper import PiperVoice
    voice=PiperVoice.load(str(voices/(voice_name+".onnx")),use_cuda=False)

    filters=subprocess.run(
        ["ffmpeg","-hide_banner","-filters"],
        check=True,capture_output=True,text=True
    ).stdout
    has_rubberband="rubberband" in filters
    print("CPU_TTS_FILTER", "rubberband" if has_rubberband else "atempo", flush=True)

    fitted=[]
    overflow_count=0
    speed_values=[]
    started=time.monotonic()

    for i,s in enumerate(segments,1):
        rawwav=segdir/f"raw-{i:05d}.wav"
        fitwav=segdir/f"fit-{i:05d}.wav"
        with wave.open(str(rawwav),"wb") as w:
            voice.synthesize_wav(clean(s["vi"]),w)

        original=max(0.01,wav_duration(rawwav))
        next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+max_extra_gap
        available_end=min(next_start-0.04,float(s["end"])+max_extra_gap)
        slot=max(0.25,available_end-float(s["start"]))
        required=original/slot
        tempo=max(min_tempo,min(max_tempo,required))
        speed_values.append(tempo)

        if has_rubberband:
            filt=(
                f"rubberband=tempo={tempo:.6f}:pitch={pitch_ratio:.6f}:formant=preserved,"
                "highpass=f=70,lowpass=f=11500,"
                "acompressor=threshold=-20dB:ratio=2.2:attack=8:release=120:makeup=1.4"
            )
        else:
            filt=(
                f"atempo={tempo:.6f},highpass=f=70,lowpass=f=11500,"
                "acompressor=threshold=-20dB:ratio=2.2:attack=8:release=120:makeup=1.4"
            )
        run(["ffmpeg","-y","-v","error","-i",str(rawwav),"-af",filt,
             "-ar","22050","-ac","1","-c:a","pcm_s16le",str(fitwav)])

        if wav_duration(fitwav)>slot+0.08:
            overflow_count+=1
            trimmed=segdir/f"trim-{i:05d}.wav"
            run(["ffmpeg","-y","-v","error","-i",str(fitwav),
                 "-af",f"atrim=duration={slot:.3f},afade=t=out:st={max(0.0,slot-0.06):.3f}:d=0.06",
                 "-ar","22050","-ac","1","-c:a","pcm_s16le",str(trimmed)])
            fitwav=trimmed

        fitted.append((s,fitwav))
        if i%50==0 or i==len(segments):
            elapsed=max(0.001,time.monotonic()-started)
            rate=i/elapsed
            remain=(len(segments)-i)/rate if rate>0 else 0
            print(f"CPU_TTS_PROGRESS {i}/{len(segments)} eta={remain/60:.1f}m overflow={overflow_count}",flush=True)

    duration=max(float(s["end"]) for s in segments)+1.0
    with wave.open(str(fitted[0][1]),"rb") as w:
        rate=w.getframerate(); width=w.getsampwidth(); channels=w.getnchannels()
    if width!=2 or channels!=1:
        raise RuntimeError("unexpected fitted WAV format")

    frames=max(1,int(math.ceil(duration*rate)))
    canvas=np.zeros(frames,dtype=np.float32)
    for s,path in fitted:
        with wave.open(str(path),"rb") as w:
            data=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)
        pos=max(0,int(round(float(s["start"])*rate)))
        end=min(len(canvas),pos+len(data))
        if pos<len(canvas):
            canvas[pos:end]+=data[:end-pos]

    canvas=np.clip(canvas,-32768,32767).astype("<i2")
    voice_wav=work/"vietnamese-voice.wav"
    with wave.open(str(voice_wav),"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(canvas.tobytes())

    run(["ffmpeg","-y","-v","error","-i",str(voice_wav),
         "-af","loudnorm=I=-17.5:TP=-2.0:LRA=7",
         "-c:a","libmp3lame","-b:a","96k","-ac","1",str(out_path)])

    sha=hashlib.sha256(out_path.read_bytes()).hexdigest()
    meta["timing_overflow_segments"]=overflow_count
    meta["mean_tempo"]=round(sum(speed_values)/max(1,len(speed_values)),4)
    meta["voice_bytes"]=out_path.stat().st_size
    meta["voice_sha256"]=sha
    meta["voice_render_device"]="github-actions-cpu"
    meta["voice_render_seconds"]=round(time.monotonic()-started,2)
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"CPU_TTS_DONE bytes={out_path.stat().st_size} sha256={sha}",flush=True)


if __name__=="__main__":
    main()

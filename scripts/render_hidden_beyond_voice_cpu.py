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


def wav_format(path:Path)->tuple[int,int,int]:
    with wave.open(str(path),"rb") as w:
        return w.getframerate(),w.getsampwidth(),w.getnchannels()


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
    max_tempo=float(style.get("max_tempo",1.16))
    max_extra_gap=float(style.get("max_extra_gap",0.65))
    min_pause_between_cues=float(style.get("min_pause_between_cues",0.20))
    voice_name=str(meta.get("voice_name") or "vi_VN-vais1000-medium")

    work=out_path.parent/"cpu-tts"
    segdir=work/"segments"
    voices=work/"voices"
    segdir.mkdir(parents=True,exist_ok=True)
    voices.mkdir(parents=True,exist_ok=True)

    run([sys.executable,"-m","piper.download_voices",voice_name,"--download-dir",str(voices)])
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    voice=PiperVoice.load(str(voices/(voice_name+".onnx")),use_cuda=False)
    syn=SynthesisConfig()

    filters=subprocess.run(
        ["ffmpeg","-hide_banner","-filters"],
        check=True,capture_output=True,text=True
    ).stdout
    has_rubberband="rubberband" in filters
    print("CPU_TTS_FILTER","rubberband" if has_rubberband else "atempo",flush=True)

    fitted=[]
    overflow_count=0
    retimed_count=0
    hard_trim_count=0
    started=time.monotonic()

    for i,s in enumerate(segments,1):
        rawwav=segdir/f"raw-{i:05d}.wav"
        with wave.open(str(rawwav),"wb") as w:
            voice.synthesize_wav(clean(s["vi"]),w,syn_config=syn)

        rate,width,channels=wav_format(rawwav)
        if (rate,width,channels)!=(22050,2,1):
            normalized=segdir/f"norm-{i:05d}.wav"
            run([
                "ffmpeg","-y","-v","error","-i",str(rawwav),
                "-ar","22050","-ac","1","-c:a","pcm_s16le",str(normalized)
            ])
            rawwav=normalized

        original=max(0.01,wav_duration(rawwav))
        next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+max_extra_gap+min_pause_between_cues
        available_end=min(next_start-min_pause_between_cues,float(s["end"])+max_extra_gap)
        slot=max(0.25,available_end-float(s["start"]))

        # Fast path: most Piper lines already fit their cue. Do not spawn ffmpeg
        # for those lines. Old code spawned one process for every segment.
        fitwav=rawwav
        if original>slot+0.08:
            retimed_count+=1
            tempo=min(max_tempo,max(1.0,original/slot))
            fitwav=segdir/f"fit-{i:05d}.wav"
            if has_rubberband:
                speed_filter=f"rubberband=tempo={tempo:.6f}:pitch=1.0:formant=preserved"
            else:
                speed_filter=f"atempo={tempo:.6f}"
            run([
                "ffmpeg","-y","-v","error","-i",str(rawwav),
                "-af",f"{speed_filter},afade=t=out:st={max(0.0,slot-0.05):.3f}:d=0.05",
                "-ar","22050","-ac","1","-c:a","pcm_s16le",str(fitwav)
            ])
            fitted_duration=wav_duration(fitwav)
            if fitted_duration>slot+0.08:
                overflow_count+=1
                # Last-resort trim is explicit and measurable. Production quality
                # gate can reject runs with too many trims instead of hiding them.
                trimmed=segdir/f"trim-{i:05d}.wav"
                run([
                    "ffmpeg","-y","-v","error","-i",str(fitwav),
                    "-af",f"atrim=duration={slot:.3f},afade=t=out:st={max(0.0,slot-0.06):.3f}:d=0.06",
                    "-ar","22050","-ac","1","-c:a","pcm_s16le",str(trimmed)
                ])
                fitwav=trimmed
                hard_trim_count+=1

        fitted.append((s,fitwav))
        if i%100==0 or i==len(segments):
            elapsed=max(0.001,time.monotonic()-started)
            rate_seg=i/elapsed
            remain=(len(segments)-i)/rate_seg if rate_seg>0 else 0
            print(
                f"CPU_TTS_PROGRESS {i}/{len(segments)} eta={remain/60:.1f}m "
                f"retimed={retimed_count} overflow={overflow_count}",
                flush=True
            )

    duration=max(float(s["end"]) for s in segments)+1.0
    with wave.open(str(fitted[0][1]),"rb") as w:
        rate=w.getframerate()
        width=w.getsampwidth()
        channels=w.getnchannels()
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
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(canvas.tobytes())

    # One global audio cleanup pass instead of repeating filters hundreds of times.
    run([
        "ffmpeg","-y","-v","error","-i",str(voice_wav),
        "-af","highpass=f=70,lowpass=f=11000,acompressor=threshold=-21dB:ratio=2.0:attack=10:release=140:makeup=1.15,loudnorm=I=-19.0:TP=-3.0:LRA=6",
        "-c:a","libmp3lame","-b:a","96k","-ac","1",str(out_path)
    ])

    sha=hashlib.sha256(out_path.read_bytes()).hexdigest()
    render_seconds=round(time.monotonic()-started,2)
    meta["timing_overflow_segments"]=overflow_count
    meta["retimed_segments"]=retimed_count
    meta["hard_trim_segments"]=hard_trim_count
    meta["voice_loudness_target_lufs"]=-19.0
    meta["min_pause_between_cues"]=min_pause_between_cues
    meta["max_extra_gap"]=max_extra_gap
    meta["voice_bytes"]=out_path.stat().st_size
    meta["voice_sha256"]=sha
    meta["voice_render_device"]="github-actions-cpu"
    meta["voice_render_seconds"]=render_seconds
    meta["voice_efficiency_mode"]="piper-fast-path-no-per-segment-ffmpeg-v3"
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(
        f"CPU_TTS_DONE seconds={render_seconds:.1f} retimed={retimed_count}/{len(segments)} "
        f"hard_trim={hard_trim_count} bytes={out_path.stat().st_size} sha256={sha}",
        flush=True
    )


if __name__=="__main__":
    main()

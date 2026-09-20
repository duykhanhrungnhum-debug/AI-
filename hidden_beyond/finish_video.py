#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
import wave
from pathlib import Path


def ts(sec: float) -> str:
    ms=max(0,int(round(float(sec)*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ffprobe_duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1",str(path)
    ],text=True).strip())


def has_audio(path: Path) -> bool:
    probe=subprocess.run([
        "ffprobe","-v","error","-select_streams","a:0",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(path)
    ],text=True,capture_output=True,check=False)
    return probe.returncode==0 and "audio" in probe.stdout


def atempo_chain(speed: float) -> str:
    speed=max(0.5,min(2.0,float(speed)))
    return f"atempo={speed:.6f}"


def fit_segment(src: Path, dst: Path, target_seconds: float) -> dict:
    original=max(0.01,ffprobe_duration(src))
    target=max(0.18,float(target_seconds))
    raw_speed=original/target

    # Keep speech natural. Only compress/expand moderately.
    speed=max(0.88,min(1.28,raw_speed))
    filt=[
        atempo_chain(speed),
        "highpass=f=70",
        "lowpass=f=11500",
        "afade=t=in:st=0:d=0.025",
    ]
    predicted=original/speed
    fade_out=max(0.0,predicted-0.035)
    filt.append(f"afade=t=out:st={fade_out:.3f}:d=0.035")
    r=subprocess.run([
        "ffmpeg","-y","-v","error","-i",str(src),
        "-af",",".join(filt),"-ar","22050","-ac","1","-c:a","pcm_s16le",str(dst)
    ],text=True,capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2500:])
    fitted=ffprobe_duration(dst)
    return {
        "source_seconds":round(original,3),
        "slot_seconds":round(target,3),
        "speed":round(speed,3),
        "fitted_seconds":round(fitted,3),
        "overflow_seconds":round(max(0.0,fitted-target),3),
    }


def build_voice_track(base: Path, segments: list[dict], total: float, out: Path) -> list[dict]:
    fitted_dir=base/"fitted"
    fitted_dir.mkdir(parents=True,exist_ok=True)
    pieces=[]
    timing=[]
    rate=width=channels=None

    for item in segments:
        src=base/item["audio_file"]
        slot=max(0.18,float(item["end"])-float(item["start"])-0.06)
        dst=fitted_dir/f"{int(item['index']):05d}.wav"
        info=fit_segment(src,dst,slot)
        with wave.open(str(dst),"rb") as w:
            params=(w.getframerate(),w.getsampwidth(),w.getnchannels())
            if rate is None:
                rate,width,channels=params
            if params!=(rate,width,channels):
                raise ValueError("Fitted TTS WAV formats differ")
            data=w.readframes(w.getnframes())
        # Center short speech inside its dialogue window. Long speech starts near cue.
        fitted=info["fitted_seconds"]
        spare=max(0.0,slot-fitted)
        start=float(item["start"])+min(0.12,spare*0.35)
        pieces.append((start,data))
        timing.append({
            "index":item["index"],
            "start":round(start,3),
            "end":round(min(total,start+fitted),3),
            **info,
        })

    if not pieces:
        raise ValueError("No TTS segments")
    frames=max(1,int(total*rate))
    canvas=bytearray(b"\0"*(frames*width*channels))
    for start,data in pieces:
        pos=int(start*rate)*width*channels
        end=min(len(canvas),pos+len(data))
        if pos<len(canvas):
            canvas[pos:end]=data[:end-pos]
    with wave.open(str(out),"wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        w.writeframes(bytes(canvas))
    return timing


def mux_with_original(input_path: Path, voice: Path, output: Path) -> str:
    if has_audio(input_path):
        # Keep ambience/SFX. Vietnamese voice acts as a side-chain to duck the source audio
        # only while speech is present, then the original track returns naturally.
        filt=(
            "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            "highpass=f=35[orig];"
            "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            "volume=1.20[voice];"
            "[orig][voice]sidechaincompress="
            "threshold=0.012:ratio=9:attack=12:release=280:makeup=1[ducked];"
            "[ducked]volume=0.72[bed];"
            "[bed][voice]amix=inputs=2:weights='1 1':normalize=0,"
            "alimiter=limit=0.95[outa]"
        )
        cmd=[
            "ffmpeg","-y","-v","error","-i",str(input_path),"-i",str(voice),
            "-filter_complex",filt,
            "-map","0:v:0","-map","[outa]","-c:v","copy",
            "-c:a","aac","-b:a","192k","-movflags","+faststart",
            "-shortest",str(output)
        ]
        mode="ducked_original_plus_vietnamese_voice"
    else:
        cmd=[
            "ffmpeg","-y","-v","error","-i",str(input_path),"-i",str(voice),
            "-map","0:v:0","-map","1:a:0","-c:v","copy",
            "-c:a","aac","-b:a","192k","-movflags","+faststart",
            "-shortest",str(output)
        ]
        mode="vietnamese_voice_only_no_source_audio"

    r=subprocess.run(cmd,text=True,capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return mode


def main():
    p=argparse.ArgumentParser()
    for x in ("input","result","output","srt","metadata"):
        p.add_argument("--"+x,required=True)
    a=p.parse_args()
    input_path=Path(a.input)
    result_path=Path(a.result)
    output_path=Path(a.output)
    base=result_path.parent
    d=json.loads(result_path.read_text(encoding="utf-8"))
    segments=d["segments"]
    total=ffprobe_duration(input_path)

    with tempfile.TemporaryDirectory(prefix="hidden-beyond-finish-") as td:
        work=Path(td)
        voice=work/"vietnamese-voice.wav"
        timing=build_voice_track(base,segments,total,voice)
        mix_mode=mux_with_original(input_path,voice,output_path)

    lines=[]
    for i,(item,t) in enumerate(zip(segments,timing,strict=True),1):
        lines += [
            str(i),
            f"{ts(t['start'])} --> {ts(max(t['start']+0.15,t['end']))}",
            "\n".join(textwrap.wrap(item["translation"],48)),
            "",
        ]
    Path(a.srt).write_text("\n".join(lines),encoding="utf-8")

    overflow=max((x["overflow_seconds"] for x in timing),default=0.0)
    avg_speed=sum(x["speed"] for x in timing)/len(timing)
    metadata={
        "status":"ok",
        "translated_title":d["translated_title"],
        "subtitle_language":"vi",
        "audio_language":"vi",
        "original_visuals_preserved":True,
        "original_audio_preserved":has_audio(input_path),
        "audio_mix_mode":mix_mode,
        "timing_strategy":"duration_fit_per_segment",
        "average_tempo":round(avg_speed,3),
        "max_overflow_seconds":round(overflow,3),
        "segments":len(segments),
        "request_id":d["request_id"],
        "timing":timing,
    }
    Path(a.metadata).write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print(
        "HIDDEN_BEYOND_FINISH_OK "
        f"segments={len(segments)} mix={mix_mode} avg_tempo={avg_speed:.3f} "
        f"max_overflow={overflow:.3f}"
    )


if __name__=="__main__":
    main()

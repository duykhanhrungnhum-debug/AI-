#!/usr/bin/env python3
import argparse,json,subprocess,textwrap,wave
from pathlib import Path

def ts(sec):
    ms=max(0,int(round(float(sec)*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    p=argparse.ArgumentParser()
    for x in ("input","result","output","srt","metadata"): p.add_argument("--"+x,required=True)
    a=p.parse_args(); base=Path(a.result).parent
    d=json.loads(Path(a.result).read_text(encoding="utf-8"))
    seg=d["segments"]; rate=width=channels=None; pieces=[]
    for x in seg:
        with wave.open(str(base/x["audio_file"]),"rb") as w:
            params=(w.getframerate(),w.getsampwidth(),w.getnchannels())
            if rate is None: rate,width,channels=params
            if params!=(rate,width,channels): raise ValueError("TTS WAV formats differ")
            pieces.append((x,w.readframes(w.getnframes())))
    if not pieces: raise ValueError("No TTS segments")
    total=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",a.input],text=True).strip())
    frames=max(1,int(total*rate)); canvas=bytearray(b"\0"*(frames*width*channels))
    for x,data in pieces:
        start=int(float(x["start"])*rate)*width*channels
        end=min(len(canvas),start+len(data))
        if start<len(canvas): canvas[start:end]=data[:end-start]
    dubbed=Path("dubbed.wav")
    with wave.open(str(dubbed),"wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate); w.writeframes(bytes(canvas))
    lines=[]
    for i,x in enumerate(seg,1):
        lines += [str(i),f"{ts(x['start'])} --> {ts(x['end'])}","\n".join(textwrap.wrap(x["translation"],48)),""]
    Path(a.srt).write_text("\n".join(lines),encoding="utf-8")
    cmd=["ffmpeg","-y","-i",a.input,"-i",str(dubbed),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",a.output]
    r=subprocess.run(cmd,text=True,capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr[-2500:])
    meta={"status":"ok","translated_title":d["translated_title"],"subtitle_language":"vi","audio_language":"vi","original_visuals_preserved":True,"segments":len(seg),"request_id":d["request_id"]}
    Path(a.metadata).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()

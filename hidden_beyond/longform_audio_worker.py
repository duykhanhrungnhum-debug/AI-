#!/usr/bin/env python3
from __future__ import annotations

# __JOB_CONFIG_INJECT__

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path("/kaggle/working")
AUDIO=WORK/"source-audio.mp3"
VOICE_WAV=WORK/"vietnamese-voice.wav"
VOICE_MP3=WORK/"voice.mp3"
SRT=WORK/"vi.srt"
META=WORK/"metadata.json"
SEGDIR=WORK/"tts"
SEGDIR.mkdir(parents=True,exist_ok=True)

API=JOB["callback_base"].rstrip("/")
JOB_ID=JOB["job_id"]
JOB_TOKEN=JOB["job_token"]
_stage={"name":"starting","message":"GPU worker starting"}
_stop=threading.Event()

def post(path:str,payload:dict,*,token:bool=True)->dict:
    data=json.dumps(payload,ensure_ascii=False).encode()
    headers={"content-type":"application/json"}
    if token:
        headers["x-job-token"]=JOB_TOKEN
    req=Request(API+path,data=data,headers=headers,method="POST")
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def heartbeat(stage:str,message:str):
    _stage["name"]=stage; _stage["message"]=message
    try:
        post("/heartbeat",{"job_id":JOB_ID,"stage":stage,"message":message})
        print("HB_HEARTBEAT",stage,message,flush=True)
    except Exception as exc:
        print("HB_HEARTBEAT_ERROR",stage,repr(exc),flush=True)

def heartbeat_loop():
    while not _stop.wait(60):
        heartbeat(_stage["name"],_stage["message"])

def download(url:str,path:Path):
    req=Request(url,headers={"User-Agent":"Hidden-Beyond-AI/2.0"})
    with urlopen(req,timeout=180) as src,path.open("wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk: break
            dst.write(chunk)

def upload(url:str,path:Path,mime:str):
    with path.open("rb") as f:
        data=f.read()
    req=Request(url,data=data,headers={"content-type":mime,"x-upsert":"true"},method="PUT")
    with urlopen(req,timeout=300) as r:
        if r.status not in (200,201):
            raise RuntimeError(f"upload failed {r.status}")

def run(cmd):
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def clean(s:str)->str:
    return re.sub(r"\s+"," ",s).strip()

def wav_duration(path:Path)->float:
    with wave.open(str(path),"rb") as w:
        return w.getnframes()/w.getframerate()

def ts(sec:float)->str:
    ms=max(0,int(round(sec*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    heartbeat("installing","Installing local AI runtime")
    run([sys.executable,"-m","pip","install","--quiet",
         "faster-whisper>=1.1,<2","transformers<5","accelerate<2",
         "sentencepiece","sacremoses","piper-tts>=1.3,<2"])

    import torch
    from faster_whisper import WhisperModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    heartbeat("fetching_input","Fetching compressed source audio")
    cfg=post("/worker-config",{"job_id":JOB_ID})
    download(cfg["input_audio_url"],AUDIO)
    if AUDIO.stat().st_size<100000:
        raise RuntimeError("input audio is unexpectedly small")

    device="cuda" if torch.cuda.is_available() else "cpu"
    compute="float16" if device=="cuda" else "int8"
    heartbeat("transcribing",f"Speech recognition on {device}")
    whisper=WhisperModel("large-v3-turbo",device=device,compute_type=compute)
    seg_iter,info=whisper.transcribe(str(AUDIO),language="zh",vad_filter=True,beam_size=5,condition_on_previous_text=True)
    raw=[{"start":float(s.start),"end":float(s.end),"text":clean(s.text)} for s in seg_iter if clean(s.text)]
    if len(raw)<10: raise RuntimeError(f"too few transcript segments: {len(raw)}")

    merged=[]
    for s in raw:
        if not merged:
            merged.append(dict(s)); continue
        cur=merged[-1]
        candidate=(cur["text"]+" "+s["text"]).strip()
        if s["start"]-cur["end"]<=0.45 and s["end"]-cur["start"]<=7.0 and len(candidate)<=72:
            cur["end"]=s["end"]; cur["text"]=candidate
        else:
            merged.append(dict(s))
    segments=merged
    heartbeat("transcribed",f"Transcript ready: {len(segments)} segments")
    del whisper; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    heartbeat("translating",f"Translating {len(segments)} segments to Vietnamese")
    model_name="facebook/nllb-200-distilled-600M"
    tok=AutoTokenizer.from_pretrained(model_name,src_lang="zho_Hans")
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
    model.eval()
    target_id=tok.convert_tokens_to_ids("vie_Latn")
    texts=[s["text"] for s in segments]+[str(cfg.get("title") or "")]
    translations=[]
    bs=18 if device=="cuda" else 4
    with torch.inference_mode():
        for i in range(0,len(texts),bs):
            inp=tok(texts[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=256)
            inp={k:v.to(device) for k,v in inp.items()}
            out=model.generate(**inp,forced_bos_token_id=target_id,max_new_tokens=180,num_beams=4,repetition_penalty=1.08,no_repeat_ngram_size=3)
            translations.extend(tok.batch_decode(out,skip_special_tokens=True))
            if i and i%(bs*8)==0:
                heartbeat("translating",f"Translated {min(i+bs,len(texts))}/{len(texts)}")
    translations=[clean(x) for x in translations]
    if len(translations)!=len(texts) or any(not x for x in translations): raise RuntimeError("translation output mismatch")
    for s,vi in zip(segments,translations[:-1],strict=True): s["vi"]=vi
    translated_title=translations[-1]
    del model,tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    heartbeat("tts","Starting fast Piper Vietnamese TTS")
    voice_mode="piper-vais1000-fast"
    voice_name="vi_VN-vais1000-medium"
    voices=WORK/"piper-voices"
    voices.mkdir(exist_ok=True)
    run([sys.executable,"-m","piper.download_voices",voice_name,"--download-dir",str(voices)])
    from piper import PiperVoice
    piper_voice=PiperVoice.load(str(voices/(voice_name+".onnx")),use_cuda=False)

    def synth(text:str,path:Path):
        with wave.open(str(path),"wb") as w:
            piper_voice.synthesize_wav(clean(text),w)

    fitted=[]
    tts_started=time.monotonic()
    for i,s in enumerate(segments,1):
        rawwav=SEGDIR/f"raw-{i:05d}.wav"
        fitwav=SEGDIR/f"fit-{i:05d}.wav"
        synth(s["vi"],rawwav)
        original=max(0.01,wav_duration(rawwav))
        slot=max(0.25,float(s["end"])-float(s["start"])-0.04)
        speed=max(0.80,min(1.60,original/slot))
        run(["ffmpeg","-y","-v","error","-i",str(rawwav),"-af",
             f"atempo={speed:.6f},atrim=duration={slot:.3f},highpass=f=70,lowpass=f=11500",
             "-ar","22050","-ac","1","-c:a","pcm_s16le",str(fitwav)])
        s["audio"]=fitwav
        fitted.append((s,fitwav))
        if i%50==0 or i==len(segments):
            elapsed=max(0.001,time.monotonic()-tts_started)
            rate=i/elapsed
            remaining=(len(segments)-i)/rate if rate>0 else 0
            heartbeat("tts",f"Piper {i}/{len(segments)} segments; eta≈{remaining/60:.1f} min")

    duration=max(float(s["end"]) for s in segments)+1.0
    with wave.open(str(fitted[0][1]),"rb") as w:
        rate=w.getframerate(); width=w.getsampwidth(); channels=w.getnchannels()
    frames=max(1,int(duration*rate)); canvas=bytearray(b"\0"*(frames*width*channels))
    for s,path in fitted:
        with wave.open(str(path),"rb") as w:
            data=w.readframes(w.getnframes())
        pos=int(float(s["start"])*rate)*width*channels
        end=min(len(canvas),pos+len(data))
        if pos<len(canvas): canvas[pos:end]=data[:end-pos]
    with wave.open(str(VOICE_WAV),"wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate); w.writeframes(bytes(canvas))
    run(["ffmpeg","-y","-v","error","-i",str(VOICE_WAV),"-c:a","libmp3lame","-b:a","64k","-ac","1",str(VOICE_MP3)])

    lines=[]
    for i,s in enumerate(segments,1):
        lines += [str(i),f"{ts(s['start'])} --> {ts(s['end'])}",s["vi"],""]
    SRT.write_text("\n".join(lines),encoding="utf-8")

    sha=hashlib.sha256(VOICE_MP3.read_bytes()).hexdigest()
    meta={
        "ok":True,"job_id":JOB_ID,"source_video_id":cfg["source_video_id"],
        "series_id":cfg["series_id"],"episode_number":cfg["episode_number"],
        "translated_title":translated_title,"segments":len(segments),
        "translation_model":model_name,"tts_voice":voice_mode,
        "voice_bytes":VOICE_MP3.stat().st_size,"voice_sha256":sha,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    heartbeat("uploading_ai_output","Uploading Vietnamese voice and subtitles")
    upload(cfg["voice_upload_url"],VOICE_MP3,"audio/mpeg")
    upload(cfg["subtitle_upload_url"],SRT,"text/plain")
    upload(cfg["metadata_upload_url"],META,"application/json")
    result=post("/ai-complete",{"job_id":JOB_ID,"translated_title":translated_title,"output_sha256":sha,"output_bytes":VOICE_MP3.stat().st_size})
    print("HB_AI_COMPLETE",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    thread=threading.Thread(target=heartbeat_loop,daemon=True)
    thread.start()
    try:
        main()
    except Exception as exc:
        print("HB_AI_FAIL",repr(exc),flush=True)
        try: post("/fail",{"job_id":JOB_ID,"error":repr(exc)})
        except Exception as fail_exc: print("HB_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise
    finally:
        _stop.set()

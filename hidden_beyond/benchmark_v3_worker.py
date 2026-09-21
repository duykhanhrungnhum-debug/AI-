#!/usr/bin/env python3
from __future__ import annotations

# __BENCH_CONFIG_INJECT__

import gc
import html
import json
import math
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path("/kaggle/working")
AUDIO=WORK/"benchmark-source.mp3"
META=WORK/"translated-metadata.json"
REPORT=WORK/"benchmark.json"

CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TAG_RE=re.compile(r"<[^>]+>")
REPEAT_RE=re.compile(r"\b([\wÀ-ỹ]+)(?:\s+\1)+\b",re.IGNORECASE)
GLOSSARY={
    "修仙":"tu tiên","修士":"tu sĩ","灵气":"linh khí","靈氣":"linh khí","灵力":"linh lực","靈力":"linh lực",
    "灵根":"linh căn","靈根":"linh căn","炼气":"Luyện Khí","練氣":"Luyện Khí","筑基":"Trúc Cơ","築基":"Trúc Cơ",
    "金丹":"Kim Đan","元婴":"Nguyên Anh","元嬰":"Nguyên Anh","渡劫":"Độ Kiếp","宗门":"tông môn","宗門":"tông môn",
    "师尊":"sư tôn","師尊":"sư tôn","师兄":"sư huynh","師兄":"sư huynh","师姐":"sư tỷ","師姐":"sư tỷ",
    "师弟":"sư đệ","師弟":"sư đệ","师妹":"sư muội","師妹":"sư muội","掌门":"chưởng môn","掌門":"chưởng môn",
    "长老":"trưởng lão","長老":"trưởng lão","道友":"đạo hữu","法宝":"pháp bảo","法寶":"pháp bảo",
    "丹药":"đan dược","丹藥":"đan dược","功法":"công pháp","秘境":"bí cảnh","洞府":"động phủ",
    "魔修":"ma tu","正道":"chính đạo","天道":"thiên đạo","飞升":"phi thăng","飛升":"phi thăng","境界":"cảnh giới",
}
STYLE={"max_tempo":1.22,"min_tempo":0.90,"pitch_ratio":1.025,"max_extra_gap":0.24,"words_per_second":3.25}

def callback_post(path:str,payload:dict)->dict:
    data=json.dumps(payload,ensure_ascii=False).encode()
    req=Request(
        BENCH["callback_base"].rstrip("/")+path,
        data=data,
        headers={"content-type":"application/json","x-benchmark-token":BENCH["run_token"]},
        method="POST",
    )
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def beat(stage:str,message:str)->None:
    try:
        callback_post("/heartbeat",{"run_id":BENCH["run_id"],"stage":stage,"message":message})
        print("BENCH_HEARTBEAT",stage,message,flush=True)
    except Exception as exc:
        print("BENCH_HEARTBEAT_ERROR",stage,repr(exc),flush=True)

def _alarm(_signum,_frame):
    raise TimeoutError("benchmark hard deadline exceeded")

def run(cmd:list[str])->None:
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def clean(value:str)->str:
    value=html.unescape(TAG_RE.sub("",str(value))).replace("\u200b"," ")
    value=re.sub(r"\s+"," ",value).strip().strip("“”\"")
    previous=None
    while previous!=value:
        previous=value
        value=REPEAT_RE.sub(r"\1",value)
    return re.sub(r"\s+([,.;:!?])",r"\1",value).strip()

def download(url:str,path:Path)->None:
    req=Request(url,headers={"User-Agent":"Hidden-Beyond-Benchmark/1.0"})
    with urlopen(req,timeout=180) as src,path.open("wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk:
                break
            dst.write(chunk)

def glossary_pairs(text:str)->list[tuple[str,str]]:
    out=[]
    seen=set()
    for zh,vi in GLOSSARY.items():
        if zh in text and vi not in seen:
            out.append((zh,vi)); seen.add(vi)
    return out[:8]

def validate(text:str,source:str)->str:
    value=clean(text)
    if not value or CJK_RE.search(value):
        raise ValueError("empty_or_cjk")
    for number in re.findall(r"\d+",source):
        if number not in value:
            raise ValueError("lost_number")
    return value

def merge(raw:list[dict])->list[dict]:
    out=[]
    for x in raw:
        text=clean(x.get("text",""))
        if not text:
            continue
        s={"start":float(x["start"]),"end":float(x["end"]),"text":text}
        if not out:
            out.append(s); continue
        cur=out[-1]
        candidate=clean(cur["text"]+" "+s["text"])
        if s["start"]-cur["end"]<=0.28 and s["end"]-cur["start"]<=7.5 and len(candidate)<=82:
            cur["end"]=s["end"]; cur["text"]=candidate
        else:
            out.append(s)
    return out

def choose_track(meta:dict):
    pref=("zh-Hans","zh-CN","zh","zh-Hant","zh-TW")
    for kind,key in (("manual","subtitles"),("auto","automatic_captions")):
        table=meta.get(key) or {}
        langs=list(pref)+[k for k in table if str(k).startswith("zh") and k not in pref]
        for lang in langs:
            choices=table.get(lang) or []
            for ext in ("json3","vtt"):
                for item in choices:
                    if item.get("ext")==ext and item.get("url"):
                        return kind,lang,ext,item["url"]
    return None

def parse_json3(path:Path)->list[dict]:
    data=json.loads(path.read_text(encoding="utf-8"))
    out=[]
    for e in data.get("events") or []:
        segs=e.get("segs") or []
        if not segs:
            continue
        text=clean("".join(str(s.get("utf8") or "") for s in segs))
        if not text:
            continue
        start=float(e.get("tStartMs") or 0)/1000.0
        dur=max(0.2,float(e.get("dDurationMs") or 1000)/1000.0)
        out.append({"start":start,"end":start+dur,"text":text})
    return out

def parse_vtt(path:Path)->list[dict]:
    def sec(v:str)->float:
        vals=[float(x) for x in v.replace(",",".").split(":")]
        return vals[0]*3600+vals[1]*60+vals[2] if len(vals)==3 else vals[0]*60+vals[1]
    lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
    out=[]; i=0
    while i<len(lines):
        if "-->" not in lines[i]:
            i+=1; continue
        left,right=lines[i].split("-->",1)
        start=sec(left.strip().split()[0]); end=sec(right.strip().split()[0])
        i+=1; buf=[]
        while i<len(lines) and lines[i].strip():
            buf.append(lines[i].strip()); i+=1
        text=clean(" ".join(buf))
        if text:
            out.append({"start":start,"end":max(start+0.2,end),"text":text})
        i+=1
    return out

def get_caption(meta:dict):
    track=choose_track(meta)
    if not track:
        return None
    kind,lang,ext,url=track
    p=WORK/f"caption.{ext}"
    download(url,p)
    cues=parse_json3(p) if ext=="json3" else parse_vtt(p)
    segs=merge(cues)
    cjk=sum(1 for s in segs if CJK_RE.search(s["text"]))
    if len(segs)<10 or cjk<max(5,int(len(segs)*0.35)):
        return None
    return segs,kind,lang

def main()->None:
    signal.signal(signal.SIGALRM,_alarm)
    signal.alarm(18*60)
    total_started=time.monotonic()
    beat("starting","Benchmark worker started")
    run([
        sys.executable,"-m","pip","install","--quiet",
        "yt-dlp>=2026.1","faster-whisper>=1.1,<2",
        "transformers==4.56.0","accelerate<2","sentencepiece","sacremoses",
    ])
    beat("installed","Optimized translation runtime installed")

    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    url=BENCH["source_url"]
    beat("source_probe","Checking source captions before ASR")
    probe_started=time.monotonic()
    meta=json.loads(subprocess.check_output(
        [sys.executable,"-m","yt_dlp","--no-playlist","--skip-download","--dump-single-json",url],
        text=True,timeout=120,
    ))
    caption=get_caption(meta)
    source_probe_seconds=time.monotonic()-probe_started

    transcript_started=time.monotonic()
    if caption:
        raw,kind,lang=caption
        transcript_source=f"youtube_{kind}_{lang}"
        asr_model="skipped-caption-available"
        source_fetch_seconds=source_probe_seconds
    else:
        fetch_started=time.monotonic()
        run([
            sys.executable,"-m","yt_dlp","--no-playlist",
            "--retries","5","--fragment-retries","5",
            "-x","--audio-format","mp3","--audio-quality","64K",
            "-o",str(AUDIO),url,
        ])
        source_fetch_seconds=source_probe_seconds+(time.monotonic()-fetch_started)
        asr_model="large-v3-turbo"
        whisper=WhisperModel(asr_model,device="cuda",compute_type="float16")
        batched=BatchedInferencePipeline(model=whisper)
        seg_iter,info=batched.transcribe(
            str(AUDIO),language="zh",vad_filter=True,batch_size=16,beam_size=1,
            condition_on_previous_text=False,word_timestamps=False,
            vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
        )
        raw=[{"start":float(s.start),"end":float(s.end),"text":clean(s.text)} for s in seg_iter if clean(s.text)]
        transcript_source="faster_whisper_large-v3-turbo_batched"
        del batched,whisper
        gc.collect(); torch.cuda.empty_cache()

    segments=merge(raw)
    if len(segments)<10:
        raise RuntimeError(f"too few segments: {len(segments)}")
    for i,s in enumerate(segments,1):
        s["index"]=i
        s["slot"]=max(0.3,s["end"]-s["start"])
        s["max_words"]=max(2,int(math.floor(s["slot"]*STYLE["words_per_second"]+0.5)))
    transcript_seconds=time.monotonic()-transcript_started
    beat("transcribed",f"{transcript_source}: {len(segments)} segments in {transcript_seconds:.1f}s")

    translation_started=time.monotonic()
    model_name="tencent/Hy-MT2-1.8B"
    load_started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(model_name)
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        model_name,torch_dtype=torch.float16,low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    model_load_seconds=time.monotonic()-load_started
    beat("translating",f"Hy-MT2 loaded in {model_load_seconds:.1f}s; translating {len(segments)} segments")

    def prompt(s:dict)->str:
        pairs=glossary_pairs(s["text"])
        terms=("参考下面的翻译：\n"+"\n".join(f"{a} 翻译成 {b}" for a,b in pairs)+"\n\n") if pairs else ""
        return (
            terms+"将以下文本翻译为越南语。译文自然、简洁，适合中国仙侠动画越南语配音；"
            "保持人物称谓、境界和专有名词一致；不要解释，不要输出中文。"
            f"尽量不超过 {s['max_words']+2} 个越南语词。\n\n{s['text']}"
        )

    translated={}
    invalid=[]
    bs=20
    inference_started=time.monotonic()
    with torch.inference_mode():
        cursor=0
        while cursor<len(segments):
            batch=segments[cursor:cursor+bs]
            chats=[
                tok.apply_chat_template(
                    [{"role":"user","content":prompt(s)}],
                    tokenize=False,add_generation_prompt=False
                ) for s in batch
            ]
            try:
                inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=512)
                inp={k:v.to("cuda") for k,v in inp.items()}
                out=model.generate(
                    **inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,
                    use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs<=5:
                    raise
                bs=max(5,bs//2)
                print("BENCH_BATCH_REDUCE",bs,flush=True)
                continue
            gen=out[:,inp["input_ids"].shape[1]:]
            vis=tok.batch_decode(gen,skip_special_tokens=True)
            for s,vi in zip(batch,vis,strict=True):
                try:
                    translated[s["index"]]=validate(vi,s["text"])
                except Exception:
                    invalid.append(s["index"])
                    translated[s["index"]]=clean(vi)
            cursor+=len(batch)
            elapsed=max(0.001,time.monotonic()-inference_started)
            print(
                f"BENCH_TRANSLATE {cursor}/{len(segments)} "
                f"pct={cursor*100/len(segments):.1f} rate={cursor/elapsed:.2f} invalid={len(invalid)}",
                flush=True
            )
            beat(
                "translating",
                f"Hy-MT2 {cursor}/{len(segments)} ({cursor*100/len(segments):.1f}%); "
                f"{cursor/elapsed:.2f} seg/s; invalid={len(invalid)}"
            )

    inference_seconds=time.monotonic()-inference_started
    translation_seconds=time.monotonic()-translation_started
    for s in segments:
        s["vi"]=translated[s["index"]]

    report={
        "ok":True,
        "source_video_id":BENCH["source_video_id"],
        "duration_seconds":meta.get("duration"),
        "transcript_source":transcript_source,
        "segments":len(segments),
        "source_fetch_seconds":round(source_fetch_seconds,2),
        "transcript_seconds":round(transcript_seconds,2),
        "translation_model":model_name,
        "translation_model_load_seconds":round(model_load_seconds,2),
        "translation_inference_seconds":round(inference_seconds,2),
        "translation_seconds":round(translation_seconds,2),
        "invalid_segments":len(invalid),
        "invalid_ids":invalid[:50],
        "gpu":torch.cuda.get_device_name(0),
        "gpu_benchmark_seconds":round(time.monotonic()-total_started,2),
        "target_translation_seconds":300,
        "target_compute_seconds":1200,
    }
    timed={
        "ok":True,
        "source_video_id":BENCH["source_video_id"],
        "episode_number":1,
        "translated_title":"Hidden Beyond Benchmark",
        "segments":len(segments),
        "translation_model":model_name,
        "transcript_source":transcript_source,
        "style":STYLE,
        "voice_name":"vi_VN-vais1000-medium",
        "gpu_translation_seconds":round(translation_seconds,2),
        "timed_segments":[{
            "index":s["index"],"start":round(s["start"],3),"end":round(s["end"],3),
            "vi":s["vi"],"max_words":s["max_words"]
        } for s in segments],
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    META.write_text(json.dumps(timed,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    callback_post("/complete",{
        "run_id":BENCH["run_id"],
        "report":report,
        "metadata":timed,
    })
    signal.alarm(0)
    print("HB_V3_BENCHMARK_DONE",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("HB_V3_BENCHMARK_FAIL",repr(exc),flush=True)
        try:
            callback_post("/fail",{"run_id":BENCH["run_id"],"error":repr(exc)})
        except Exception as report_exc:
            print("HB_V3_BENCHMARK_FAIL_REPORT_ERROR",repr(report_exc),flush=True)
        raise
    finally:
        signal.alarm(0)

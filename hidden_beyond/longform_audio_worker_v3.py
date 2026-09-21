#!/usr/bin/env python3
from __future__ import annotations

# __JOB_CONFIG_INJECT__

import gc
import html
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path("/kaggle/working")
AUDIO=WORK/"source-audio.mp3"
SRT=WORK/"vi.srt"
META=WORK/"metadata.json"

API=JOB["callback_base"].rstrip("/")
JOB_ID=JOB["job_id"]
JOB_TOKEN=JOB["job_token"]
_stage={"name":"starting","message":"GPU worker starting"}
_stop=threading.Event()

CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REPEAT_RE=re.compile(r"\b([\wÀ-ỹ]+)(?:\s+\1)+\b",re.IGNORECASE)

def has_ngram_loop(value:str)->bool:
    words=re.findall(r"[\wÀ-ỹ]+",value.lower())
    for n in (2,3,4):
        for i in range(0,max(0,len(words)-n*3+1)):
            gram=words[i:i+n]
            if gram and words[i+n:i+2*n]==gram and words[i+2*n:i+3*n]==gram:
                return True
    return False
TAG_RE=re.compile(r"<[^>]+>")
FANTASY_GLOSSARY={
    "修仙":"tu tiên","修士":"tu sĩ","灵气":"linh khí","靈氣":"linh khí","灵力":"linh lực","靈力":"linh lực",
    "灵根":"linh căn","靈根":"linh căn","炼气":"Luyện Khí","練氣":"Luyện Khí","筑基":"Trúc Cơ","築基":"Trúc Cơ",
    "金丹":"Kim Đan","元婴":"Nguyên Anh","元嬰":"Nguyên Anh","渡劫":"Độ Kiếp","宗门":"tông môn","宗門":"tông môn",
    "师尊":"sư tôn","師尊":"sư tôn","师兄":"sư huynh","師兄":"sư huynh","师姐":"sư tỷ","師姐":"sư tỷ",
    "师弟":"sư đệ","師弟":"sư đệ","师妹":"sư muội","師妹":"sư muội","掌门":"chưởng môn","掌門":"chưởng môn",
    "长老":"trưởng lão","長老":"trưởng lão","道友":"đạo hữu","法宝":"pháp bảo","法寶":"pháp bảo",
    "丹药":"đan dược","丹藥":"đan dược","功法":"công pháp","秘境":"bí cảnh","洞府":"động phủ",
    "魔修":"ma tu","正道":"chính đạo","天道":"thiên đạo","飞升":"phi thăng","飛升":"phi thăng","境界":"cảnh giới",
}
STYLE={
    "max_tempo":1.16,
    "min_tempo":0.94,
    "pitch_ratio":1.0,
    "max_extra_gap":0.65,
    "words_per_second":3.0,
    "target_segment_seconds":3.2,
    "hard_max_segment_seconds":5.5,
    "min_pause_between_cues":0.20,
}

def post(path:str,payload:dict)->dict:
    data=json.dumps(payload,ensure_ascii=False).encode()
    req=Request(
        API+path,
        data=data,
        headers={"content-type":"application/json","x-job-token":JOB_TOKEN},
        method="POST",
    )
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def heartbeat(stage:str,message:str)->None:
    _stage["name"]=stage
    _stage["message"]=message
    try:
        post("/heartbeat",{"job_id":JOB_ID,"stage":stage,"message":message})
        print("HB_HEARTBEAT",stage,message,flush=True)
    except Exception as exc:
        print("HB_HEARTBEAT_ERROR",stage,repr(exc),flush=True)

def heartbeat_loop()->None:
    while not _stop.wait(60):
        heartbeat(_stage["name"],_stage["message"])

def download(url:str,path:Path)->None:
    req=Request(url,headers={"User-Agent":"Hidden-Beyond-AI/4.0"})
    with urlopen(req,timeout=180) as src,path.open("wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk:
                break
            dst.write(chunk)

def upload(url:str,path:Path,mime:str)->None:
    with path.open("rb") as f:
        data=f.read()
    req=Request(url,data=data,headers={"content-type":mime,"x-upsert":"true"},method="PUT")
    with urlopen(req,timeout=300) as r:
        if r.status not in (200,201):
            raise RuntimeError(f"upload failed {r.status}")

def run(cmd:list[str])->None:
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def clean(value:str)->str:
    value=html.unescape(TAG_RE.sub("",str(value)))
    value=value.replace("\u200b"," ")
    value=re.sub(r"\s+"," ",value).strip().strip("“”\"")
    previous=None
    while previous!=value:
        previous=value
        value=REPEAT_RE.sub(r"\1",value)
    value=re.sub(r"\s+([,.;:!?])",r"\1",value)
    return value.strip()

def ts(sec:float)->str:
    ms=max(0,int(round(sec*1000)))
    h,r=divmod(ms,3600000)
    m,r=divmod(r,60000)
    s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def validate_vi(text:str,source:str,*,field:str)->str:
    value=clean(text)
    if not value:
        raise ValueError(f"{field} empty")
    if CJK_RE.search(value):
        raise ValueError(f"{field} still contains CJK")
    if REPEAT_RE.search(value) or has_ngram_loop(value):
        raise ValueError(f"{field} has repetition loop")
    for number in re.findall(r"\d+",source):
        if number not in value:
            raise ValueError(f"{field} lost number {number}")
    return value

def vi_word_count(value:str)->int:
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+",clean(value)))

def fit_word_limit(segment:dict)->int:
    slot=max(0.25,float(segment.get("tts_slot") or (float(segment["end"])-float(segment["start"]))))
    return max(2,int(math.ceil(slot*5.2)))

def validate_segment_fit(text:str,segment:dict,*,field:str)->str:
    value=validate_vi(text,segment["text"],field=field)
    limit=int(segment.get("fit_words") or fit_word_limit(segment))
    if vi_word_count(value)>limit:
        raise ValueError(f"{field} too long for slot")
    return value

def glossary_pairs(text:str)->list[tuple[str,str]]:
    pairs=[]
    seen=set()
    for zh,vi in FANTASY_GLOSSARY.items():
        if zh in text and vi not in seen:
            pairs.append((zh,vi))
            seen.add(vi)
    return pairs[:8]

def split_words_to_dialogue(words:list[dict])->list[dict]:
    """Create short dubbing cues from Whisper word timestamps.

    The target cadence mirrors the supplied reference: sentence-sized lines,
    natural pauses, and no minute-long cues.
    """
    usable=[]
    for w in words:
        text=str(w.get("text") or "")
        if not clean(text):
            continue
        start=max(0.0,float(w.get("start") or 0))
        end=max(start+0.04,float(w.get("end") or start+0.08))
        if end-start>1.8:
            end=start+1.8
        usable.append({"start":start,"end":end,"text":text})
    if not usable:
        return []

    strong=set("。！？!?；;")
    soft=set("，,、：:")
    out=[]
    cur=[]
    hard=float(STYLE["hard_max_segment_seconds"])
    for i,w in enumerate(usable):
        if cur:
            gap_before=w["start"]-cur[-1]["end"]
            cur_duration=cur[-1]["end"]-cur[0]["start"]
            if gap_before>=0.55 or (gap_before>=0.32 and cur_duration>=1.40):
                out.append({
                    "start":cur[0]["start"],"end":cur[-1]["end"],
                    "text":clean("".join(x["text"] for x in cur)),
                })
                cur=[]
        if cur and w["end"]-cur[0]["start"]>hard:
            out.append({
                "start":cur[0]["start"],"end":cur[-1]["end"],
                "text":clean("".join(x["text"] for x in cur)),
            })
            cur=[]
        cur.append(w)
        start=cur[0]["start"]; end=w["end"]
        text=clean("".join(x["text"] for x in cur))
        duration=end-start
        cjk_count=len(CJK_RE.findall(text))
        next_gap=(usable[i+1]["start"]-end) if i+1<len(usable) else 9.0
        last=text[-1] if text else ""
        target=float(STYLE["target_segment_seconds"])
        boundary=(
            (last in strong and duration>=1.20)
            or (last in soft and duration>=2.20)
            or (next_gap>=0.55 and duration>=0.70)
            or (next_gap>=0.32 and duration>=1.40)
            or (duration>=target and (next_gap>=0.18 or last in strong or last in soft))
            or duration>=hard
            or cjk_count>=32
            or len(text)>=64
        )
        if boundary:
            out.append({"start":start,"end":end,"text":text})
            cur=[]
    if cur:
        out.append({
            "start":cur[0]["start"],
            "end":cur[-1]["end"],
            "text":clean("".join(x["text"] for x in cur)),
        })

    fixed=[]
    for s in out:
        dur=s["end"]-s["start"]
        if fixed:
            prev=fixed[-1]
            prev_dur=prev["end"]-prev["start"]
            gap=s["start"]-prev["end"]
            candidate=clean(prev["text"]+s["text"])
            if (
                gap<=0.35
                and s["end"]-prev["start"]<=hard
                and (dur<0.90 or prev_dur<0.90)
                and len(candidate)<=64
            ):
                prev["end"]=s["end"]; prev["text"]=candidate
                continue
        fixed.append(s)
    return fixed

def split_timed_text(raw:list[dict])->list[dict]:
    """Normalize source captions without joining them into long cues."""
    out=[]
    punct=re.compile(r"(?<=[。！？!?；;])")
    for item in raw:
        text=clean(item.get("text",""))
        if not text:
            continue
        start=float(item["start"]); end=float(item["end"])
        duration=max(0.2,end-start)
        parts=[clean(p) for p in punct.split(text) if clean(p)] or [text]
        expanded=[]
        for part in parts:
            while len(part)>36:
                cut=max(part.rfind("，",0,30),part.rfind(",",0,30),part.rfind("、",0,30))
                if cut<10:
                    cut=30
                expanded.append(clean(part[:cut+1]))
                part=clean(part[cut+1:])
            if part:
                expanded.append(part)
        total=max(1,sum(max(1,len(p)) for p in expanded))
        cursor=start
        for n,part in enumerate(expanded):
            frac=max(1,len(part))/total
            part_end=end if n==len(expanded)-1 else min(end,cursor+duration*frac)
            if part_end-cursor<0.25:
                part_end=min(end,cursor+0.25)
            out.append({"start":cursor,"end":max(cursor+0.2,part_end),"text":part})
            cursor=part_end
    return out

def collect_whisper_segments(seg_iter)->tuple[list[dict],int]:
    words=[]
    fallback=[]
    word_count=0
    for s in seg_iter:
        text=clean(s.text)
        if not text:
            continue
        sw=getattr(s,"words",None) or []
        if sw:
            for w in sw:
                wt=str(getattr(w,"word","") or "")
                if not clean(wt):
                    continue
                ws=getattr(w,"start",None)
                we=getattr(w,"end",None)
                words.append({
                    "start":float(s.start if ws is None else ws),
                    "end":float(s.end if we is None else we),
                    "text":wt,
                })
                word_count+=1
        else:
            fallback.append({"start":float(s.start),"end":float(s.end),"text":text})
    if words:
        return split_words_to_dialogue(words),word_count
    return split_timed_text(fallback),word_count

def select_caption_track(meta:dict):
    preferred=("zh-Hans","zh-CN","zh","zh-Hant","zh-TW")
    for kind,key in (("manual","subtitles"),("auto","automatic_captions")):
        table=meta.get(key) or {}
        ordered=list(preferred)+[k for k in table if str(k).startswith("zh") and k not in preferred]
        for lang in ordered:
            choices=table.get(lang) or []
            for ext in ("json3","vtt"):
                for item in choices:
                    if item.get("ext")==ext and item.get("url"):
                        return kind,lang,ext,item["url"]
    return None

def parse_json3_caption(path:Path)->list[dict]:
    data=json.loads(path.read_text(encoding="utf-8"))
    out=[]
    for event in data.get("events") or []:
        segs=event.get("segs") or []
        if not segs:
            continue
        text=clean("".join(str(s.get("utf8") or "") for s in segs))
        if not text:
            continue
        start=float(event.get("tStartMs") or 0)/1000.0
        dur=max(0.20,float(event.get("dDurationMs") or 1000)/1000.0)
        out.append({"start":start,"end":start+dur,"text":text})
    return out

def parse_vtt_caption(path:Path)->list[dict]:
    def sec(v:str)->float:
        p=v.replace(",",".").split(":")
        nums=[float(x) for x in p]
        if len(nums)==3:
            return nums[0]*3600+nums[1]*60+nums[2]
        return nums[0]*60+nums[1]
    lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
    out=[]
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if "-->" not in line:
            i+=1
            continue
        left,right=line.split("-->",1)
        start=sec(left.strip().split()[0])
        end=sec(right.strip().split()[0])
        i+=1
        text_lines=[]
        while i<len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i+=1
        text=clean(" ".join(text_lines))
        if text:
            out.append({"start":start,"end":max(start+0.2,end),"text":text})
        i+=1
    return out

def try_youtube_captions(source_video_id:str):
    url=f"https://www.youtube.com/watch?v={source_video_id}"
    try:
        raw=subprocess.check_output(
            [sys.executable,"-m","yt_dlp","--no-playlist","--skip-download","--dump-single-json",url],
            text=True,timeout=120,stderr=subprocess.DEVNULL,
        )
        meta=json.loads(raw)
        track=select_caption_track(meta)
        if not track:
            return None
        kind,lang,ext,track_url=track
        path=WORK/f"source-caption.{ext}"
        download(track_url,path)
        cues=parse_json3_caption(path) if ext=="json3" else parse_vtt_caption(path)
        merged=split_timed_text(cues)
        cjk=sum(1 for s in merged if CJK_RE.search(s["text"]))
        if len(merged)<10 or cjk<max(5,int(len(merged)*0.35)):
            return None
        return merged,kind,lang,float(meta.get("duration") or 0)
    except Exception as exc:
        print("HB_CAPTION_FALLBACK",repr(exc),flush=True)
        return None

def main()->None:
    heartbeat("installing","Installing optimized local AI runtime")
    run([
        sys.executable,"-m","pip","install","--quiet",
        "yt-dlp>=2026.1",
        "faster-whisper>=1.1,<2",
        "transformers>=5.6,<6",
        "accelerate<2",
        "sentencepiece",
        "sacremoses",
    ])

    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    cfg=post("/worker-config",{"job_id":JOB_ID})
    device="cuda" if torch.cuda.is_available() else "cpu"
    compute="float16" if device=="cuda" else "int8"
    source_video_id=str(cfg["source_video_id"])

    pipeline_started=time.monotonic()
    transcript_started=time.monotonic()
    heartbeat("caption_probe","Checking source captions before ASR")
    caption_result=try_youtube_captions(source_video_id)

    if caption_result:
        raw,caption_kind,caption_lang,transcript_media_duration=caption_result
        transcript_source=f"youtube_{caption_kind}_{caption_lang}"
        whisper_name="skipped-caption-available"
        detected_language="zh"
        language_probability=None
        asr_word_count=0
        heartbeat("transcribed",f"Source captions ready: {len(raw)} cues; ASR skipped")
    else:
        heartbeat("fetching_input","No usable Chinese captions; fetching compressed source audio")
        download(cfg["input_audio_url"],AUDIO)
        if AUDIO.stat().st_size<100000:
            raise RuntimeError("input audio is unexpectedly small")
        whisper_name="large-v3-turbo" if device=="cuda" else "small"
        transcript_source=f"faster_whisper_{whisper_name}"
        heartbeat("transcribing",f"Batched speech recognition on {device} with {whisper_name}")
        whisper=WhisperModel(whisper_name,device=device,compute_type=compute)
        if device=="cuda":
            transcriber=BatchedInferencePipeline(model=whisper)
            seg_iter,info=transcriber.transcribe(
                str(AUDIO),
                language="zh",
                vad_filter=True,
                batch_size=16,
                beam_size=1,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
            )
        else:
            seg_iter,info=whisper.transcribe(
                str(AUDIO),
                language="zh",
                vad_filter=True,
                beam_size=2,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
            )
        raw,asr_word_count=collect_whisper_segments(seg_iter)
        detected_language=info.language
        language_probability=float(info.language_probability or 0)
        transcript_media_duration=float(getattr(info,"duration",0) or 0)
        del whisper
        if device=="cuda":
            del transcriber
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        heartbeat("transcribed",f"ASR ready: {len(raw)} raw cues")

    segments=raw
    if len(segments)<10:
        raise RuntimeError(f"too few transcript segments: {len(segments)}")
    for i,s in enumerate(segments,1):
        s["index"]=i
        s["slot"]=max(0.30,float(s["end"])-float(s["start"]))
        next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+float(STYLE["max_extra_gap"])+float(STYLE["min_pause_between_cues"])
        available_end=min(
            next_start-float(STYLE["min_pause_between_cues"]),
            float(s["end"])+float(STYLE["max_extra_gap"]),
        )
        s["tts_slot"]=max(0.25,available_end-float(s["start"]))
        s["max_words"]=max(2,int(math.floor(s["tts_slot"]*STYLE["words_per_second"]+0.5)))
        s["fit_words"]=fit_word_limit(s)
    transcript_seconds=round(time.monotonic()-transcript_started,2)
    transcript_last_end=max(float(s["end"]) for s in segments)
    transcript_coverage_pct=(transcript_last_end*100.0/transcript_media_duration) if transcript_media_duration>0 else 0.0
    segments_per_minute=(len(segments)/(transcript_media_duration/60.0)) if transcript_media_duration>0 else 0.0
    heartbeat(
        "transcribed",
        f"Transcript ready: {len(segments)} segments in {transcript_seconds:.1f}s via {transcript_source}; "
        f"coverage={transcript_coverage_pct:.1f}% density={segments_per_minute:.2f}/min"
    )
    if transcript_media_duration>=600 and transcript_coverage_pct<95.0:
        raise RuntimeError(
            f"transcript coverage too low: {transcript_coverage_pct:.1f}% "
            f"({transcript_last_end:.1f}s/{transcript_media_duration:.1f}s)"
        )
    if transcript_media_duration>=600 and segments_per_minute<3.0:
        raise RuntimeError(
            f"dialogue segmentation too coarse: {segments_per_minute:.2f} segments/min "
            f"for {transcript_media_duration/60.0:.1f} minutes"
        )
    too_long=sum(1 for s in segments if float(s["end"])-float(s["start"])>6.2)
    if too_long:
        raise RuntimeError(f"dialogue segmentation has {too_long} cues longer than 6.2s")

    translation_started=time.monotonic()
    translated={}
    invalid_ids=[]
    qwen_reviewed=0
    fallback_segments=0
    primary_model=""
    fast_path_used=False

    def opus_translate(items:list[dict], *, on_device:str)->dict[int,str]:
        nonlocal fallback_segments
        model_name="Helsinki-NLP/opus-mt-zh-vi"
        otok=AutoTokenizer.from_pretrained(model_name)
        omodel=AutoModelForSeq2SeqLM.from_pretrained(model_name,low_cpu_mem_usage=True).to(on_device)
        omodel.eval()
        result={}
        bs=24 if on_device=="cuda" else 8
        with torch.inference_mode():
            for off in range(0,len(items),bs):
                batch=items[off:off+bs]
                inp=otok([x["text"] for x in batch],return_tensors="pt",padding=True,truncation=True,max_length=256)
                inp={k:v.to(on_device) for k,v in inp.items()}
                out=omodel.generate(**inp,max_new_tokens=128,num_beams=1,repetition_penalty=1.05)
                texts=otok.batch_decode(out,skip_special_tokens=True)
                for s,vi in zip(batch,texts,strict=True):
                    try:
                        result[s["index"]]=validate_vi(vi,s["text"],field=f"fallback segment {s['index']}")
                    except Exception:
                        result[s["index"]]=clean(vi)
                fallback_segments+=len(batch)
        del omodel,otok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    if device=="cuda":
        primary_model="tencent/Hy-MT2-1.8B"
        heartbeat("translating",f"Loading {primary_model}; batch translation for {len(segments)} segments")
        tok=AutoTokenizer.from_pretrained(primary_model,trust_remote_code=True)
        tok.padding_side="left"
        if tok.pad_token_id is None:
            tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(
            primary_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        by_index={s["index"]:s for s in segments}
        def prompt_for(s:dict)->str:
            terms=glossary_pairs(s["text"])
            term_text=""
            if terms:
                term_text="参考下面的翻译：\n"+"\n".join(f"{zh} 翻译成 {vi}" for zh,vi in terms)+"\n\n"
            prev=by_index.get(s["index"]-1,{}).get("text","")
            nxt=by_index.get(s["index"]+1,{}).get("text","")
            context=clean(prev+" "+nxt)
            context_text=(context+"\n参考上面的信息，") if context else ""
            return (
                term_text+context_text+
                "把下面的文本翻译成越南语，注意只需要输出翻译后的结果，不要翻译上文，也不要额外解释：\n"
                +s["text"]
            )

        batch_size=20
        cursor=0
        budget_seconds=8*60
        with torch.inference_mode():
            while cursor<len(segments):
                if time.monotonic()-translation_started>=budget_seconds:
                    fast_path_used=True
                    heartbeat(
                        "translating_fast_path",
                        f"Hy-MT2 budget reached at {cursor}/{len(segments)}; switching remaining to fast fallback",
                    )
                    break
                batch=segments[cursor:cursor+batch_size]
                chats=[
                    tok.apply_chat_template(
                        [{"role":"user","content":prompt_for(s)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for s in batch
                ]
                try:
                    inp=tok(
                        chats,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
                    out=model.generate(
                        **inp,
                        max_new_tokens=96,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.6,
                        top_k=20,
                        repetition_penalty=1.05,
                        use_cache=True,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size<=5:
                        raise
                    batch_size=max(5,batch_size//2)
                    heartbeat("translating",f"GPU memory guard: reducing translation batch to {batch_size}")
                    continue
                generated=out[:,inp["input_ids"].shape[1]:]
                vis=tok.batch_decode(generated,skip_special_tokens=True)
                for s,vi in zip(batch,vis,strict=True):
                    try:
                        translated[s["index"]]=validate_vi(vi,s["text"],field=f"segment {s['index']}")
                    except Exception:
                        invalid_ids.append(s["index"])
                cursor+=len(batch)
                elapsed=max(0.001,time.monotonic()-translation_started)
                rate=cursor/elapsed
                pct=cursor*100.0/len(segments)
                heartbeat(
                    "translating",
                    f"Hy-MT2 {cursor}/{len(segments)} ({pct:.1f}%); {rate:.1f} seg/s; invalid={len(invalid_ids)}",
                )

        title_item={
            "index":0,
            "text":str(cfg.get("title") or ""),
            "max_words":20,
        }
        try:
            title_chat=tok.apply_chat_template(
                [{"role":"user","content":(
                    "将以下标题翻译为越南语，保持仙侠作品风格和专有名词，只输出越南语标题，不要解释：\n\n"
                    +title_item["text"]
                )}],
                tokenize=False,
                add_generation_prompt=False,
            )
            inp=tok([title_chat],return_tensors="pt",padding=True,truncation=True,max_length=256)
            inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
            out=model.generate(
                **inp,max_new_tokens=64,do_sample=False,repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
            )
            generated=out[:,inp["input_ids"].shape[1]:]
            translated_title=clean(tok.batch_decode(generated,skip_special_tokens=True)[0])
            if not translated_title or CJK_RE.search(translated_title):
                raise ValueError("invalid translated title")
        except Exception:
            translated_title=clean(str(cfg.get("series_title") or cfg.get("title") or "Hidden Beyond"))

        del model,tok
        gc.collect()
        torch.cuda.empty_cache()

        if cursor<len(segments):
            remaining=segments[cursor:]
            translated.update(opus_translate(remaining,on_device="cuda"))

        overlong_ids=[
            s["index"] for s in segments
            if translated.get(s["index"]) and vi_word_count(translated[s["index"]])>s["fit_words"]
        ]
        review_ids=sorted(set(invalid_ids)|set(overlong_ids))
        if review_ids:
            # Qwen is an editor only. It is never the bulk translator.
            review_set=set(review_ids)
            review_items=[s for s in segments if s["index"] in review_set]
            heartbeat(
                "translation_review",
                f"Reviewing {len(review_items)} segments: invalid={len(invalid_ids)} overlong={len(overlong_ids)}",
            )
            editor_name="Qwen/Qwen2.5-1.5B-Instruct"
            qtok=AutoTokenizer.from_pretrained(editor_name)
            qtok.padding_side="left"
            if qtok.pad_token_id is None:
                qtok.pad_token=qtok.eos_token
            qmodel=AutoModelForCausalLM.from_pretrained(
                editor_name,torch_dtype=torch.float16,low_cpu_mem_usage=True
            ).to(device)
            qmodel.eval()
            unresolved=[]
            with torch.inference_mode():
                for off in range(0,len(review_items),8):
                    batch=review_items[off:off+8]
                    prompts=[]
                    for s in batch:
                        gloss=", ".join(vi for _,vi in glossary_pairs(s["text"]))
                        prev=by_index.get(s["index"]-1,{}).get("text","")
                        nxt=by_index.get(s["index"]+1,{}).get("text","")
                        prompts.append(
                            "Dịch TARGET từ tiếng Trung sang tiếng Việt tự nhiên để lồng tiếng phim tiên hiệp. "
                            "Ngữ cảnh chỉ để hiểu, không được dịch vào câu trả lời. "
                            "Không để chữ Hán, giữ số liệu, tên riêng và xưng hô; không giải thích. "
                            +(f"Ưu tiên thuật ngữ: {gloss}. " if gloss else "")
                            +f"Viết gọn, tự nhiên, tối đa {s['fit_words']} từ để khớp thời lượng. "
                            +f"NGỮ CẢNH: {prev} {nxt}\nTARGET: {s['text']}"
                        )
                    chats=[
                        qtok.apply_chat_template(
                            [{"role":"user","content":p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for p in prompts
                    ]
                    inp=qtok(chats,return_tensors="pt",padding=True,truncation=True,max_length=384)
                    inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
                    out=qmodel.generate(
                        **inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,
                        pad_token_id=qtok.pad_token_id,eos_token_id=qtok.eos_token_id,
                    )
                    gen=out[:,inp["input_ids"].shape[1]:]
                    vis=qtok.batch_decode(gen,skip_special_tokens=True)
                    for s,vi in zip(batch,vis,strict=True):
                        try:
                            translated[s["index"]]=validate_segment_fit(
                                vi,s,field=f"review segment {s['index']}"
                            )
                            qwen_reviewed+=1
                        except Exception:
                            unresolved.append(s)
            del qmodel,qtok
            gc.collect()
            torch.cuda.empty_cache()
            if unresolved:
                heartbeat("translating_fast_path",f"{len(unresolved)} review failures; deterministic MT fallback")
                translated.update(opus_translate(unresolved,on_device="cpu"))
    else:
        primary_model="Helsinki-NLP/opus-mt-zh-vi"
        translated.update(opus_translate(segments,on_device="cpu"))
        title_model=AutoTokenizer.from_pretrained(primary_model)
        title_llm=AutoModelForSeq2SeqLM.from_pretrained(primary_model).to("cpu")
        inp=title_model([str(cfg.get("title") or "")],return_tensors="pt")
        out=title_llm.generate(**inp,max_new_tokens=96,num_beams=2)
        translated_title=clean(title_model.batch_decode(out,skip_special_tokens=True)[0])
        del title_llm,title_model
        gc.collect()

    missing=[s for s in segments if not translated.get(s["index"])]
    if missing:
        raise RuntimeError(f"translation missing {len(missing)} segments")

    final_invalid=[]
    for s in segments:
        try:
            translated[s["index"]]=validate_segment_fit(
                translated[s["index"]],s,field=f"final segment {s['index']}"
            )
        except Exception:
            final_invalid.append(s["index"])
    if final_invalid:
        raise RuntimeError(
            f"translation quality unresolved {len(final_invalid)} segments: "
            +",".join(map(str,final_invalid[:30]))
        )

    for s in segments:
        s["vi"]=translated[s["index"]]

    translation_seconds=round(time.monotonic()-translation_started,2)
    heartbeat(
        "packaging_translation",
        f"Translation complete: {len(segments)}/{len(segments)} (100%) in {translation_seconds:.1f}s; packaging for CPU TTS",
    )

    voice_mode="piper-vais1000-fast-cpu-v3"
    voice_name="vi_VN-vais1000-medium"
    lines=[]
    for i,s in enumerate(segments,1):
        lines += [str(i),f"{ts(s['start'])} --> {ts(s['end'])}",s["vi"],""]
    SRT.write_text("\n".join(lines),encoding="utf-8")

    meta={
        "ok":True,
        "job_id":JOB_ID,
        "source_video_id":cfg["source_video_id"],
        "series_id":cfg["series_id"],
        "episode_number":cfg["episode_number"],
        "translated_title":translated_title,
        "segments":len(segments),
        "translation_model":primary_model,
        "translation_editor":"Qwen/Qwen2.5-1.5B-Instruct" if qwen_reviewed else "not_used",
        "transcript_source":transcript_source,
        "whisper_model":whisper_name,
        "detected_language":detected_language,
        "language_probability":language_probability,
        "tts_voice":voice_mode,
        "translation_mode":"hy-mt2-batched-cultivation-dubbing-v3",
        "timing_mode":"speech-segment-sync",
        "style":STYLE,
        "voice_name":voice_name,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_efficiency_mode":"caption-first-batched-asr-hymt2-batch-qwen-editor-only-cpu-tts-v3",
        "transcript_seconds":transcript_seconds,
        "transcript_media_duration_seconds":round(transcript_media_duration,2),
        "transcript_last_end_seconds":round(transcript_last_end,2),
        "transcript_coverage_pct":round(transcript_coverage_pct,2),
        "segments_per_minute":round(segments_per_minute,3),
        "asr_word_count":asr_word_count,
        "max_segment_seconds":round(max(float(s["end"])-float(s["start"]) for s in segments),3),
        "avg_segment_seconds":round(sum(float(s["end"])-float(s["start"]) for s in segments)/len(segments),3),
        "gpu_translation_seconds":translation_seconds if device=="cuda" else 0,
        "gpu_worker_seconds_to_package":round(time.monotonic()-pipeline_started,2),
        "qwen_reviewed_segments":qwen_reviewed,
        "overlong_review_segments":len(overlong_ids) if device=="cuda" else 0,
        "fallback_segments":fallback_segments,
        "fast_path_used":fast_path_used,
        "timed_segments":[{
            "index":s["index"],
            "start":round(float(s["start"]),3),
            "end":round(float(s["end"]),3),
            "vi":s["vi"],
            "max_words":s["max_words"],
            "fit_words":s["fit_words"],
        } for s in segments],
    }
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    heartbeat("uploading_translation","Uploading translation artifacts; GPU work is done")
    upload(cfg["subtitle_upload_url"],SRT,"text/plain")
    upload(cfg["metadata_upload_url"],META,"application/json")
    result=post("/ai-complete",{
        "job_id":JOB_ID,
        "translated_title":translated_title,
        "output_sha256":"",
        "output_bytes":META.stat().st_size,
        "voice_generated":False,
    })
    print("HB_AI_COMPLETE_GPU_RELEASE",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    thread=threading.Thread(target=heartbeat_loop,daemon=True)
    thread.start()
    try:
        main()
    except Exception as exc:
        print("HB_AI_FAIL",repr(exc),flush=True)
        try:
            post("/fail",{"job_id":JOB_ID,"error":repr(exc)})
        except Exception as fail_exc:
            print("HB_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise
    finally:
        _stop.set()

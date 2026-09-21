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
    "max_tempo":1.22,
    "min_tempo":0.90,
    "pitch_ratio":1.025,
    "max_extra_gap":0.24,
    "words_per_second":3.25,
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
    if REPEAT_RE.search(value):
        raise ValueError(f"{field} has repeated words")
    for number in re.findall(r"\d+",source):
        if number not in value:
            raise ValueError(f"{field} lost number {number}")
    return value

def glossary_pairs(text:str)->list[tuple[str,str]]:
    pairs=[]
    seen=set()
    for zh,vi in FANTASY_GLOSSARY.items():
        if zh in text and vi not in seen:
            pairs.append((zh,vi))
            seen.add(vi)
    return pairs[:8]

def merge_timed_segments(raw:list[dict])->list[dict]:
    merged=[]
    for item in raw:
        text=clean(item.get("text",""))
        if not text:
            continue
        s={"start":float(item["start"]),"end":float(item["end"]),"text":text}
        if not merged:
            merged.append(s)
            continue
        cur=merged[-1]
        gap=s["start"]-cur["end"]
        candidate=clean(cur["text"]+" "+s["text"])
        span=s["end"]-cur["start"]
        if gap<=0.28 and span<=7.5 and len(candidate)<=82:
            cur["end"]=s["end"]
            cur["text"]=candidate
        else:
            merged.append(s)
    return merged

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
        merged=merge_timed_segments(cues)
        cjk=sum(1 for s in merged if CJK_RE.search(s["text"]))
        if len(merged)<10 or cjk<max(5,int(len(merged)*0.35)):
            return None
        return merged,kind,lang
    except Exception as exc:
        print("HB_CAPTION_FALLBACK",repr(exc),flush=True)
        return None

def main()->None:
    heartbeat("installing","Installing optimized local AI runtime")
    run([
        sys.executable,"-m","pip","install","--quiet",
        "yt-dlp>=2026.1",
        "faster-whisper>=1.1,<2",
        "transformers==4.56.0",
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
        raw,caption_kind,caption_lang=caption_result
        transcript_source=f"youtube_{caption_kind}_{caption_lang}"
        whisper_name="skipped-caption-available"
        detected_language="zh"
        language_probability=None
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
                word_timestamps=False,
                vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
            )
        else:
            seg_iter,info=whisper.transcribe(
                str(AUDIO),
                language="zh",
                vad_filter=True,
                beam_size=2,
                condition_on_previous_text=False,
                word_timestamps=False,
                vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
            )
        raw=[
            {"start":float(s.start),"end":float(s.end),"text":clean(s.text)}
            for s in seg_iter if clean(s.text)
        ]
        detected_language=info.language
        language_probability=float(info.language_probability or 0)
        del whisper
        if device=="cuda":
            del transcriber
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        heartbeat("transcribed",f"ASR ready: {len(raw)} raw cues")

    segments=merge_timed_segments(raw)
    if len(segments)<10:
        raise RuntimeError(f"too few transcript segments: {len(segments)}")
    for i,s in enumerate(segments,1):
        s["index"]=i
        s["slot"]=max(0.30,float(s["end"])-float(s["start"]))
        s["max_words"]=max(2,int(math.floor(s["slot"]*STYLE["words_per_second"]+0.5)))
    transcript_seconds=round(time.monotonic()-transcript_started,2)
    heartbeat("transcribed",f"Transcript ready: {len(segments)} timed segments in {transcript_seconds:.1f}s via {transcript_source}")

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
        tok=AutoTokenizer.from_pretrained(primary_model)
        tok.padding_side="left"
        if tok.pad_token_id is None:
            tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(
            primary_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        def prompt_for(s:dict)->str:
            terms=glossary_pairs(s["text"])
            term_text=""
            if terms:
                term_text="参考下面的翻译：\n"+"\n".join(f"{zh} 翻译成 {vi}" for zh,vi in terms)+"\n\n"
            return (
                term_text+
                "将以下文本翻译为越南语。译文必须自然、简洁，适合中国仙侠动画越南语配音；"
                "保持人物称谓、境界和专有名词一致；不要解释，不要输出中文。"
                f"尽量不超过 {s['max_words']+2} 个越南语词。\n\n{s['text']}"
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
                        add_generation_prompt=False,
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
                    inp={k:v.to(device) for k,v in inp.items()}
                    out=model.generate(
                        **inp,
                        max_new_tokens=96,
                        do_sample=False,
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
            inp={k:v.to(device) for k,v in inp.items()}
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

        if invalid_ids:
            # Qwen is an editor only. It is never the bulk translator.
            review_items=[s for s in segments if s["index"] in set(invalid_ids)]
            heartbeat("translation_review",f"Rule check flagged {len(review_items)} segments; bounded Qwen editor")
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
                        prompts.append(
                            "Dịch câu tiếng Trung sang tiếng Việt tự nhiên để lồng tiếng phim tiên hiệp. "
                            "Không để chữ Hán, giữ số liệu, tên riêng và xưng hô. "
                            +(f"Ưu tiên thuật ngữ: {gloss}. " if gloss else "")
                            +f"Tối đa {s['max_words']+2} từ. Chỉ trả câu tiếng Việt.\n{s['text']}"
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
                    inp={k:v.to(device) for k,v in inp.items()}
                    out=qmodel.generate(
                        **inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,
                        pad_token_id=qtok.pad_token_id,eos_token_id=qtok.eos_token_id,
                    )
                    gen=out[:,inp["input_ids"].shape[1]:]
                    vis=qtok.batch_decode(gen,skip_special_tokens=True)
                    for s,vi in zip(batch,vis,strict=True):
                        try:
                            translated[s["index"]]=validate_vi(vi,s["text"],field=f"review segment {s['index']}")
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
        "gpu_translation_seconds":translation_seconds if device=="cuda" else 0,
        "gpu_worker_seconds_to_package":round(time.monotonic()-pipeline_started,2),
        "qwen_reviewed_segments":qwen_reviewed,
        "fallback_segments":fallback_segments,
        "fast_path_used":fast_path_used,
        "timed_segments":[{
            "index":s["index"],
            "start":round(float(s["start"]),3),
            "end":round(float(s["end"]),3),
            "vi":s["vi"],
            "max_words":s["max_words"],
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

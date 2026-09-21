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

def has_ngram_loop(value:str)->bool:
    words=re.findall(r"[\wÀ-ỹ]+",value.lower())
    for n in (2,3,4):
        for i in range(0,max(0,len(words)-n*3+1)):
            gram=words[i:i+n]
            if gram and words[i+n:i+2*n]==gram and words[i+2*n:i+3*n]==gram:
                return True
    return False
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
STYLE={"max_tempo":1.16,"min_tempo":0.94,"pitch_ratio":1.0,"max_extra_gap":0.65,"words_per_second":3.0,"target_segment_seconds":3.2,"hard_max_segment_seconds":5.5,"min_pause_between_cues":0.20}

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
    if REPEAT_RE.search(value) or has_ngram_loop(value):
        raise ValueError("repetition_loop")
    for number in re.findall(r"\d+",source):
        if number not in value:
            raise ValueError("lost_number")
    return value

def vi_word_count(value:str)->int:
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+",clean(value)))

def fit_word_limit(segment:dict)->int:
    slot=max(0.25,float(segment.get("tts_slot") or (float(segment["end"])-float(segment["start"]))))
    return max(2,int(math.ceil(slot*5.2)))

def validate_segment(text:str,segment:dict)->str:
    value=validate(text,segment["text"])
    limit=int(segment.get("fit_words") or fit_word_limit(segment))
    if vi_word_count(value)>limit:
        raise ValueError("too_long_for_slot")
    return value

def _join_text(parts:list[str])->str:
    raw="".join(str(x) for x in parts)
    # Preserve natural Latin spacing while keeping Chinese compact.
    raw=re.sub(r"(?<=[A-Za-z0-9À-ỹ])(?=[A-Za-z0-9À-ỹ])"," ",raw) if False else raw
    return clean(raw)

def split_words_to_dialogue(words:list[dict])->list[dict]:
    """Turn Whisper word timestamps into subtitle/dubbing-size utterances.

    Quality target is based on the supplied sample: prefer punctuation/silence
    boundaries, ~1-5.5 second cues, and never minute-long ASR chunks.
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
        start=cur[0]["start"]
        end=w["end"]
        text=clean("".join(x["text"] for x in cur))
        duration=end-start
        cjk_count=len(CJK_RE.findall(text))
        next_gap=(usable[i+1]["start"]-end) if i+1<len(usable) else 9.0
        last=(text[-1] if text else "")
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

    # Merge micro-fragments so Vietnamese dubbing has enough speaking time.
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
                prev["end"]=s["end"]
                prev["text"]=candidate
                continue
        fixed.append(s)
    return fixed

def split_timed_text(raw:list[dict])->list[dict]:
    """Normalize captions/fallback segments without merging them into long cues."""
    out=[]
    punct=re.compile(r"(?<=[。！？!?；;])")
    for x in raw:
        text=clean(x.get("text",""))
        if not text:
            continue
        start=float(x["start"]); end=float(x["end"])
        duration=max(0.2,end-start)
        parts=[clean(p) for p in punct.split(text) if clean(p)]
        if not parts:
            parts=[text]
        # Further split very long caption text by punctuation/size.
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
    segs=split_timed_text(cues)
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
        "transformers>=5.6,<6","accelerate<2","sentencepiece","sacremoses",
    ])
    beat("installed","Optimized translation runtime installed")

    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    url=BENCH["source_url"]
    source_audio_url=str(BENCH.get("source_audio_url") or "").strip()
    if not source_audio_url:
        raise RuntimeError("staged source audio URL missing")

    beat("source_probe","Checking source captions; staged audio is ready as fallback")
    probe_started=time.monotonic()
    meta={}
    caption=None
    try:
        probe=subprocess.run(
            [sys.executable,"-m","yt_dlp","--no-playlist","--skip-download","--dump-single-json",url],
            text=True,capture_output=True,timeout=45
        )
        if probe.returncode==0 and probe.stdout.strip():
            meta=json.loads(probe.stdout)
            caption=get_caption(meta)
        else:
            msg=clean((probe.stderr or probe.stdout or "")[-500:])
            print("BENCH_CAPTION_PROBE_FALLBACK",msg,flush=True)
    except Exception as exc:
        print("BENCH_CAPTION_PROBE_FALLBACK",repr(exc),flush=True)

    source_duration_seconds=float(meta.get("duration") or 0)
    source_probe_seconds=time.monotonic()-probe_started

    transcript_started=time.monotonic()
    if caption:
        raw,kind,lang=caption
        transcript_source=f"youtube_{kind}_{lang}"
        asr_model="skipped-caption-available"
        source_fetch_seconds=source_probe_seconds
        transcript_media_duration=source_duration_seconds
        asr_word_count=0
    else:
        fetch_started=time.monotonic()
        beat("fetching_input","Caption unavailable; downloading staged source audio")
        download(source_audio_url,AUDIO)
        if AUDIO.stat().st_size<100000:
            raise RuntimeError("staged source audio is unexpectedly small")
        source_fetch_seconds=source_probe_seconds+(time.monotonic()-fetch_started)
        asr_model="large-v3-turbo"
        whisper=WhisperModel(asr_model,device="cuda",compute_type="float16")
        batched=BatchedInferencePipeline(model=whisper)
        seg_iter,info=batched.transcribe(
            str(AUDIO),language="zh",vad_filter=True,batch_size=16,beam_size=1,
            condition_on_previous_text=False,word_timestamps=True,
            vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
        )
        raw,asr_word_count=collect_whisper_segments(seg_iter)
        transcript_source="faster_whisper_large-v3-turbo_batched_word_timestamps"
        transcript_media_duration=float(getattr(info,"duration",0) or source_duration_seconds or 0)
        del batched,whisper
        gc.collect(); torch.cuda.empty_cache()

    segments=raw
    if len(segments)<10:
        raise RuntimeError(f"too few segments: {len(segments)}")
    for i,s in enumerate(segments,1):
        s["index"]=i
        s["slot"]=max(0.3,s["end"]-s["start"])
        next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+float(STYLE["max_extra_gap"])+float(STYLE["min_pause_between_cues"])
        available_end=min(
            next_start-float(STYLE["min_pause_between_cues"]),
            float(s["end"])+float(STYLE["max_extra_gap"]),
        )
        s["tts_slot"]=max(0.25,available_end-float(s["start"]))
        s["max_words"]=max(2,int(math.floor(s["tts_slot"]*STYLE["words_per_second"]+0.5)))
        s["fit_words"]=fit_word_limit(s)
    transcript_seconds=time.monotonic()-transcript_started
    transcript_last_end=max(float(s["end"]) for s in segments)
    transcript_coverage_pct=(transcript_last_end*100.0/transcript_media_duration) if transcript_media_duration>0 else 0.0
    segments_per_minute=(len(segments)/(transcript_media_duration/60.0)) if transcript_media_duration>0 else 0.0
    cjk_chars=sum(len(CJK_RE.findall(s["text"])) for s in segments)
    cjk_chars_per_minute=(cjk_chars/(transcript_media_duration/60.0)) if transcript_media_duration>0 else 0.0
    beat(
        "transcribed",
        f"{transcript_source}: {len(segments)} segments in {transcript_seconds:.1f}s; "
        f"coverage={transcript_coverage_pct:.1f}% density={segments_per_minute:.2f}/min cjk={cjk_chars_per_minute:.1f}/min"
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
    model_name="tencent/Hy-MT2-1.8B"
    load_started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        model_name,torch_dtype=torch.float16,low_cpu_mem_usage=True,trust_remote_code=True
    ).to("cuda")
    model.eval()
    model_load_seconds=time.monotonic()-load_started
    beat("translating",f"Hy-MT2 loaded in {model_load_seconds:.1f}s; translating {len(segments)} segments")

    by_index={s["index"]:s for s in segments}
    def prompt(s:dict)->str:
        pairs=glossary_pairs(s["text"])
        terms=("参考下面的翻译：\n"+"\n".join(f"{a} 翻译成 {b}" for a,b in pairs)+"\n\n") if pairs else ""
        prev=by_index.get(s["index"]-1,{}).get("text","")
        nxt=by_index.get(s["index"]+1,{}).get("text","")
        context=clean(prev+" "+nxt)
        context_text=(context+"\n参考上面的信息，") if context else ""
        return (
            terms+context_text+
            "把下面的文本翻译成越南语，注意只需要输出翻译后的结果，不要翻译上文，也不要额外解释：\n"
            +s["text"]
        )

    translated={}
    invalid=[]
    bs=20
    inference_started=time.monotonic()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.inference_mode():
        cursor=0
        while cursor<len(segments):
            batch=segments[cursor:cursor+bs]
            chats=[
                tok.apply_chat_template(
                    [{"role":"user","content":prompt(s)}],
                    tokenize=False,add_generation_prompt=True
                ) for s in batch
            ]
            try:
                inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=512)
                inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
                out=model.generate(
                    **inp,max_new_tokens=96,do_sample=True,temperature=0.7,top_p=0.6,top_k=20,
                    repetition_penalty=1.05,use_cache=True,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
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

    primary_invalid_count=len(invalid)
    overlong_primary=[
        s["index"] for s in segments
        if translated.get(s["index"]) and vi_word_count(translated[s["index"]])>s["fit_words"]
    ]
    review_ids=sorted(set(invalid)|set(overlong_primary))
    qwen_reviewed=0
    unresolved=[]
    if review_ids:
        beat(
            "translation_review",
            f"Reviewing {len(review_ids)} segments: invalid={len(invalid)} overlong={len(overlong_primary)}"
        )
        editor_name="Qwen/Qwen2.5-1.5B-Instruct"
        qtok=AutoTokenizer.from_pretrained(editor_name)
        qtok.padding_side="left"
        if qtok.pad_token_id is None:
            qtok.pad_token=qtok.eos_token
        qmodel=AutoModelForCausalLM.from_pretrained(
            editor_name,torch_dtype=torch.float16,low_cpu_mem_usage=True
        ).to("cuda")
        qmodel.eval()
        review_set=set(review_ids)
        review_items=[s for s in segments if s["index"] in review_set]
        with torch.inference_mode():
            for off in range(0,len(review_items),8):
                batch=review_items[off:off+8]
                prompts=[]
                for s in batch:
                    prev=by_index.get(s["index"]-1,{}).get("text","")
                    nxt=by_index.get(s["index"]+1,{}).get("text","")
                    prompts.append(
                        "Dịch TARGET từ tiếng Trung sang tiếng Việt tự nhiên để lồng tiếng phim tiên hiệp. "
                        "Ngữ cảnh chỉ để hiểu, không được dịch vào câu trả lời. Không để chữ Hán, không giải thích. "
                        f"Viết gọn, tự nhiên, tối đa {s['fit_words']} từ để khớp thời lượng. "
                        f"NGỮ CẢNH: {prev} {nxt}\nTARGET: {s['text']}"
                    )
                chats=[
                    qtok.apply_chat_template(
                        [{"role":"user","content":p}],tokenize=False,add_generation_prompt=True
                    ) for p in prompts
                ]
                inp=qtok(chats,return_tensors="pt",padding=True,truncation=True,max_length=384)
                inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
                out=qmodel.generate(
                    **inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,
                    pad_token_id=qtok.pad_token_id,eos_token_id=qtok.eos_token_id,
                )
                gen=out[:,inp["input_ids"].shape[1]:]
                vis=qtok.batch_decode(gen,skip_special_tokens=True)
                for s,vi in zip(batch,vis,strict=True):
                    try:
                        translated[s["index"]]=validate_segment(vi,s)
                        qwen_reviewed+=1
                    except Exception:
                        unresolved.append(s["index"])
        del qmodel,qtok
        gc.collect(); torch.cuda.empty_cache()

    fallback_reviewed=0
    unresolved_set=set(unresolved)
    content_fallback=[]
    for s in segments:
        if s["index"] not in unresolved_set:
            continue
        try:
            validate(translated.get(s["index"],""),s["text"])
        except Exception:
            content_fallback.append(s)
    if content_fallback:
        beat("translation_fallback",f"OPUS fallback for {len(content_fallback)} content-invalid segments")
        oname="Helsinki-NLP/opus-mt-zh-vi"
        otok=AutoTokenizer.from_pretrained(oname)
        omodel=AutoModelForSeq2SeqLM.from_pretrained(oname).to("cpu")
        omodel.eval()
        with torch.inference_mode():
            for off in range(0,len(content_fallback),8):
                batch=content_fallback[off:off+8]
                inp=otok([s["text"] for s in batch],return_tensors="pt",padding=True,truncation=True,max_length=192)
                out=omodel.generate(**inp,max_new_tokens=96,num_beams=2,repetition_penalty=1.05)
                vis=otok.batch_decode(out,skip_special_tokens=True)
                for s,vi in zip(batch,vis,strict=True):
                    translated[s["index"]]=clean(vi)
                    fallback_reviewed+=1
        del omodel,otok
        gc.collect()

    final_invalid=[]
    for s in segments:
        try:
            translated[s["index"]]=validate(translated.get(s["index"],""),s["text"])
        except Exception:
            final_invalid.append(s["index"])
    estimated_overlong_after_review=sum(
        1 for s in segments
        if translated.get(s["index"]) and vi_word_count(translated[s["index"]])>s["fit_words"]
    )

    inference_seconds=time.monotonic()-inference_started
    translation_seconds=time.monotonic()-translation_started
    invalid=final_invalid
    for s in segments:
        s["vi"]=translated.get(s["index"],"")

    report={
        "ok":True,
        "source_video_id":BENCH["source_video_id"],
        "duration_seconds":meta.get("duration"),
        "transcript_source":transcript_source,
        "segments":len(segments),
        "source_duration_seconds":round(source_duration_seconds,2),
        "transcript_media_duration_seconds":round(transcript_media_duration,2),
        "transcript_last_end_seconds":round(transcript_last_end,2),
        "transcript_coverage_pct":round(transcript_coverage_pct,2),
        "segments_per_minute":round(segments_per_minute,3),
        "asr_word_count":asr_word_count,
        "max_segment_seconds":round(max(float(s["end"])-float(s["start"]) for s in segments),3),
        "avg_segment_seconds":round(sum(float(s["end"])-float(s["start"]) for s in segments)/len(segments),3),
        "cjk_chars_per_minute":round(cjk_chars_per_minute,2),
        "source_fetch_seconds":round(source_fetch_seconds,2),
        "transcript_seconds":round(transcript_seconds,2),
        "translation_model":model_name,
        "translation_model_load_seconds":round(model_load_seconds,2),
        "translation_inference_seconds":round(inference_seconds,2),
        "translation_seconds":round(translation_seconds,2),
        "invalid_segments":len(invalid),
        "primary_invalid_segments":primary_invalid_count,
        "primary_overlong_segments":len(overlong_primary),
        "review_segments":len(review_ids),
        "qwen_reviewed_segments":qwen_reviewed,
        "unresolved_after_qwen":len(unresolved),
        "fallback_reviewed_segments":fallback_reviewed,
        "estimated_overlong_after_review":estimated_overlong_after_review,
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

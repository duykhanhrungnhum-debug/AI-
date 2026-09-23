#!/usr/bin/env python3
from __future__ import annotations

# __JOB_CONFIG_INJECT__

import gc
import glob
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path("/kaggle/working")
SOURCE=WORK/"source.mp4"
AUDIO=WORK/"source-audio.mp3"
VOICE=WORK/"vietnamese-voice.wav"
OUT=WORK/"processed.mp4"
SRT=WORK/"vi.srt"
META=WORK/"metadata.json"

API=JOB["callback_base"].rstrip("/")
JOB_ID=JOB["job_id"]
JOB_TOKEN=JOB["job_token"]
BOT2_MODE=str(JOB.get("mode") or "").lower()=="bot2"
WORKER_PHASE=str(JOB.get("phase") or "full").strip().lower()
WORKER_REVISION=str(JOB.get("worker_revision") or "").strip()
CHECKPOINT_REVISION=str(JOB.get("checkpoint_revision") or WORKER_REVISION).strip()
_stage={"name":"starting","message":"worker starting"}
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

def repair_prompt_policy(failure_reason:str)->dict:
    """Keep broken structural output out of the model's repair context.

    A bad previous translation can poison deterministic decoding and reproduce
    the same loop/CJK/meta failure. Soft-review rewrites may still use context.
    """
    reason=str(failure_reason or "")
    structural=(
        "still contains CJK" in reason
        or "repetition loop" in reason
        or " empty" in reason
        or reason.endswith("empty")
    )
    return {
        "include_previous":not structural,
        "include_context":not structural,
        "structural":structural,
    }

def collapse_cjk_asr_repetition(value:str)->str:
    """Collapse obvious consecutive CJK ASR loops while preserving emphasis twice."""
    value=clean(value)
    for _ in range(3):
        before=value
        for width in range(8,1,-1):
            pattern=re.compile(rf"((?:[\u3400-\u4dbf\u4e00-\u9fff]{{{width}}}))\1{{2,}}")
            value=pattern.sub(lambda m:m.group(1)*2,value)
        if value==before:
            break
    return value
TAG_RE=re.compile(r"<[^>]+>")
DIALOGUE_GLOSSARY={
    "自寻死路":"tự tìm đường chết",
    "别给脸不要脸":"đừng có không biết điều",
    "一成":"một thành",
}
ACTIVE_PROFILE={}
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
    token_header="x-bot2-token" if BOT2_MODE else "x-job-token"
    req=Request(
        API+path,
        data=data,
        headers={"content-type":"application/json",token_header:JOB_TOKEN},
        method="POST",
    )
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def bot2_stage_name(stage:str)->str:
    if stage in {"installing","starting"}:
        return "gpu_submitted"
    if stage in {"caption_probe","extracting_audio","transcribing","transcribed"}:
        return "asr"
    if stage=="translation_complete":
        return "translation_complete"
    if stage in {"translating","packaging_translation"}:
        return "translating"
    if stage=="translation_repair":
        return "translation_repair"
    if stage in {"tts_loading","tts"}:
        return "tts"
    if stage in {"mixing","evidence"}:
        return "mixing"
    if stage in {"youtube_upload","youtube_uploaded"}:
        return stage
    if stage=="source_ready":
        return "source_ready"
    return "gpu_submitted"

def infer_translation_profile(cfg:dict,segments:list[dict],existing:dict)->dict:
    profile=dict(existing or {})
    if str(profile.get("genre") or "auto").lower()!="auto":
        return profile
    sample=" ".join([
        str(cfg.get("series_title") or ""),
        str(cfg.get("title") or ""),
        *[str(x.get("text") or "") for x in segments[:220]],
    ])
    families={
        "xianxia":("修仙","修士","灵气","灵根","炼气","筑基","金丹","元婴","渡劫","宗门","师尊","法宝","飞升","天道"),
        "historical":("皇上","皇帝","王爷","将军","公主","朝廷","江湖","武林","门派","掌门","大侠","少侠"),
        "modern":("公司","老板","总裁","手机","微信","学校","医院","办公室","同事","经理","互联网"),
        "crime":("警察","凶手","案件","尸体","侦探","证据","嫌疑人","法医","调查","犯罪"),
    }
    scores={name:sum(sample.count(term) for term in terms) for name,terms in families.items()}
    genre,max_score=max(scores.items(),key=lambda kv:kv[1])
    if max_score<2:
        genre="general"
    rules={
        "xianxia":[
            "Giữ sắc thái tiên hiệp/cổ phong nhưng tiếng Việt phải tự nhiên.",
            "Suy luận xưng hô theo vai vế; ưu tiên ta/ngươi/nàng/hắn khi đúng ngữ cảnh.",
        ],
        "historical":[
            "Dùng thoại Việt cổ trang tự nhiên, không hiện đại hóa xưng hô tùy tiện.",
            "Giữ chức tước, vai vế và quan hệ nhân vật nhất quán.",
        ],
        "modern":[
            "Dùng tiếng Việt hiện đại tự nhiên; xưng hô theo tuổi, quan hệ và hoàn cảnh.",
            "Không mang cách xưng hô cổ trang sang bối cảnh hiện đại.",
        ],
        "crime":[
            "Ưu tiên chính xác thông tin, bằng chứng, thời gian, số liệu và quan hệ nhân vật.",
            "Giữ nhịp thoại hiện đại, rõ ràng và căng thẳng khi nguồn có sắc thái đó.",
        ],
        "general":[
            "Dùng tiếng Việt điện ảnh tự nhiên và trung tính.",
            "Suy luận xưng hô theo ngữ cảnh, không áp một phong cách cố định.",
        ],
    }
    profile.update({
        "profile_version":int(profile.get("profile_version") or 1),
        "profile_key":"auto-"+genre,
        "genre":genre,
        "register":"natural cinematic Vietnamese",
        "pronoun_policy":"infer_from_relationship_and_context",
        "style_rules":rules[genre],
        "detected_from":"title_plus_first_220_segments",
    })
    return profile

def profile_prompt_rule(profile:dict)->str:
    genre=str(profile.get("genre") or "general").lower()
    if genre=="xianxia":
        return "使用自然、专业的仙侠/修仙影视越南语对白；按人物关系保持古风称谓和辈分一致，不得把所有人物机械翻成同一种称呼。"
    if genre=="historical":
        return "使用自然的越南语古装影视对白；保持身份、官职、辈分和称谓一致，不得随意现代化。"
    if genre=="modern":
        return "使用自然、口语化的现代越南语；根据年龄、关系和场景选择 tôi/anh/em/bạn 等称谓，不得套用古装称谓。"
    if genre=="crime":
        return "使用准确、简洁的现代越南语悬疑/刑侦对白；证据、时间、数字、身份和因果关系必须精确。"
    return "使用自然、专业、适合影视对白的越南语；根据人物关系和场景自动选择称谓，不得套用固定题材风格。"

def heartbeat(stage:str,message:str)->None:
    _stage["name"]=stage
    _stage["message"]=message
    try:
        if BOT2_MODE:
            post("/worker-stage",{
                "source_video_id":str(JOB["source_video_id"]),
                "stage":bot2_stage_name(stage),
                "message":message,
                "gpu_kernel_ref":str(JOB.get("gpu_kernel_ref") or ""),
            })
        else:
            post("/heartbeat",{"job_id":JOB_ID,"stage":stage,"message":message})
        print("HB_HEARTBEAT",stage,message,flush=True)
    except Exception as exc:
        print("HB_HEARTBEAT_ERROR",stage,repr(exc),flush=True)

def heartbeat_loop()->None:
    while not _stop.wait(60):
        heartbeat(_stage["name"],_stage["message"])

def translation_segment_signature(segments:list[dict])->str:
    material="\n".join(
        f"{int(s.get('index') or 0)}|{clean(s.get('text',''))}"
        for s in segments
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def fetch_translation_checkpoint()->dict:
    if not BOT2_MODE or not CHECKPOINT_REVISION:
        return {}
    try:
        return post("/translation-checkpoint-get",{
            "source_video_id":str(JOB["source_video_id"]),
            "worker_revision":CHECKPOINT_REVISION,
        })
    except Exception as exc:
        print("HB_CHECKPOINT_LOAD_ERROR",repr(exc),flush=True)
        return {}

def load_translation_checkpoint(segment_signature:str,segment_count:int)->dict:
    result=fetch_translation_checkpoint()
    if not result.get("found"):
        return {}
    payload=result.get("payload") or {}
    stored_signature=str(payload.get("segment_signature") or "")
    if stored_signature==segment_signature:
        return payload

    # Same source video can produce tiny ASR text differences between runs.
    # A completed checkpoint is safe to reuse when the segment cardinality is
    # identical and every translated index is present; each reused line is
    # validated again before CPU handoff.
    translated=dict(payload.get("translated") or {})
    complete_compatible=(
        str(result.get("phase") or "")=="translation_complete"
        and int(result.get("cursor") or 0)==int(segment_count)
        and len(translated)==int(segment_count)
        and all(str(i) in translated for i in range(1,int(segment_count)+1))
    )
    if complete_compatible:
        print(
            "HB_CHECKPOINT_COMPATIBLE same_video_same_count",
            segment_count,
            "signature_mismatch",
            flush=True,
        )
        return payload

    print(
        "HB_CHECKPOINT_IGNORED signature_mismatch",
        "stored_cursor="+str(result.get("cursor") or 0),
        "current_count="+str(segment_count),
        flush=True,
    )
    return {}

def save_translation_checkpoint(
    phase:str,
    cursor:int,
    segment_signature:str,
    translated:dict,
    review_ids:set,
    hard_reasons:dict,
    translation_profile:dict,
    extra_payload:dict|None=None,
)->None:
    if not BOT2_MODE or not CHECKPOINT_REVISION:
        return
    try:
        payload={
            "segment_signature":segment_signature,
            "translated":{str(k):v for k,v in translated.items()},
            "review_ids":sorted(int(x) for x in review_ids),
            "hard_reasons":{str(k):str(v) for k,v in hard_reasons.items()},
            "translation_profile":translation_profile,
        }
        if extra_payload:
            payload.update(extra_payload)
        post("/translation-checkpoint-save",{
            "source_video_id":str(JOB["source_video_id"]),
            "worker_revision":CHECKPOINT_REVISION,
            "phase":phase,
            "cursor":int(cursor),
            "payload":payload,
        })
        print("HB_CHECKPOINT_SAVED",phase,cursor,flush=True)
    except Exception as exc:
        print("HB_CHECKPOINT_SAVE_ERROR",repr(exc),flush=True)

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
    """Deterministic hard gate only: defects that are unambiguously unusable for Vietnamese dubbing."""
    value=clean(text)
    if not value:
        raise ValueError(f"{field} empty")
    if CJK_RE.search(value):
        raise ValueError(f"{field} still contains CJK")
    if REPEAT_RE.search(value) or has_ngram_loop(value):
        raise ValueError(f"{field} has repetition loop")
    return value

def vi_word_count(value:str)->int:
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+",clean(value)))

META_HALLUCINATION_VI=(
    "không thể thực hiện yêu cầu","không thể đáp ứng yêu cầu","vi phạm bản quyền",
    "vấn đề bản quyền","chính sách nội dung","đội làm phim","đội phim","đoàn phim",
    "quay phim","quá trình quay",
)
META_SOURCE_ZH=("版权","著作权","拍摄","摄制","剧组","影片","电影","政策","请求","要求")
NEGATION_ZH=("不","没","沒有","没有","未","無","无","别","別","莫")
NEGATION_VI=("không","chẳng","chưa","đừng","khỏi","không có","chớ")
QUESTION_ZH=("吗","嗎","呢","？","?")
QUESTION_VI=("không","à","ư","sao","gì","nào","chứ")

def has_vi_question_marker(value:str)->bool:
    low=clean(value).casefold()
    if "?" in low:
        return True
    words=set(re.findall(r"[A-Za-zÀ-ỹ]+",low))
    return any(marker in words for marker in QUESTION_VI)

def validate_translation_pair(text:str,source:str,*,field:str,enforce_intent:bool=True)->str:
    """Hard production gate: only deterministic translation defects."""
    return validate_vi(text,source,field=field)

def translation_review_reasons(text:str,source:str)->list[str]:
    """Heuristic quality signals. They request one repair pass but never hard-stop by themselves."""
    value=clean(text)
    reasons=[]
    src_cjk=len(CJK_RE.findall(source))
    words=vi_word_count(value)
    if src_cjk and words>max(10,math.ceil(src_cjk*2.6)+3):
        reasons.append("extreme_expansion")
    if src_cjk>=10 and words<max(2,math.floor(src_cjk/6)):
        reasons.append("extreme_omission")
    low=value.casefold()
    if not any(term in source for term in META_SOURCE_ZH):
        if any(term in low for term in META_HALLUCINATION_VI):
            reasons.append("meta_hallucination")
    if any(term in source for term in NEGATION_ZH) and not any(term in low for term in NEGATION_VI):
        reasons.append("negation")
    if any(term in source for term in QUESTION_ZH) and not has_vi_question_marker(value):
        reasons.append("question")
    source_numbers=re.findall(r"\d+",source)
    target_numbers=re.findall(r"\d+",value)
    if source_numbers and source_numbers!=target_numbers:
        reasons.append("number_surface_mismatch")
    for zh,vi in glossary_pairs(source):
        if vi.casefold() not in low:
            reasons.append("glossary_"+zh)
    return reasons

def fit_word_limit(segment:dict)->int:
    slot=max(0.25,float(segment.get("tts_slot") or (float(segment["end"])-float(segment["start"]))))
    return max(2,int(math.ceil(slot*4.8)))

def validate_segment_fit(text:str,segment:dict,*,field:str)->str:
    value=validate_vi(text,segment["text"],field=field)
    limit=int(segment.get("fit_words") or fit_word_limit(segment))
    if vi_word_count(value)>limit:
        raise ValueError(f"{field} too long for slot")
    return value

def glossary_pairs(text:str)->list[tuple[str,str]]:
    pairs=[]
    seen=set()
    profile_glossary=ACTIVE_PROFILE.get("glossary") or {}
    tables=[DIALOGUE_GLOSSARY]
    if str(ACTIVE_PROFILE.get("genre") or "").lower()=="xianxia":
        tables.append(FANTASY_GLOSSARY)
    tables.append(profile_glossary)
    for table in tables:
        for zh,vi in table.items():
            if zh in text and vi not in seen:
                pairs.append((zh,vi))
                seen.add(vi)
    return pairs[:12]

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

def wav_duration(path:Path)->float:
    with wave.open(str(path),"rb") as w:
        return w.getnframes()/w.getframerate()

def render_voice_and_video(segments:list[dict],meta:dict)->dict:
    import numpy as np
    import soundfile as sf
    from vieneu import Vieneu

    voice_name=str(meta.get("voice_name") or "Ngọc Linh")
    segdir=WORK/"tts"
    segdir.mkdir(exist_ok=True)
    max_tempo=float((meta.get("style") or {}).get("max_tempo",1.16))
    min_pause=float((meta.get("style") or {}).get("min_pause_between_cues",0.20))

    heartbeat("tts_loading",f"Loading VieNeu-TTS v3 Turbo ONNX int8 voice={voice_name}")
    tts=Vieneu(mode="v3turbo",backend="onnx",precision="int8")
    available={voice_id for _,voice_id in tts.list_preset_voices()}
    if voice_name not in available:
        raise RuntimeError(f"VieNeu preset voice not available: {voice_name}")

    fitted=[]
    retimed=0
    hard_trim=0
    started=time.monotonic()
    batch_size=24

    for off in range(0,len(segments),batch_size):
        batch=segments[off:off+batch_size]
        texts=[clean(s["vi"]) for s in batch]
        audios=tts.infer_batch(
            texts,
            voice=voice_name,
            max_batch_size=batch_size,
            apply_watermark=False,
        )
        if len(audios)!=len(batch):
            raise RuntimeError(f"VieNeu batch output mismatch {len(audios)} != {len(batch)}")

        for s,audio in zip(batch,audios,strict=True):
            i=int(s["index"])
            raw=segdir/f"raw-{i:05d}.wav"
            arr=np.asarray(audio,dtype=np.float32)
            if arr.size<100:
                raise RuntimeError(f"VieNeu empty audio segment {i}")
            sf.write(str(raw),arr,tts.sample_rate,subtype="PCM_16")

            original=max(0.01,wav_duration(raw))
            next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+1.2
            slot_end=min(next_start-min_pause,float(s["end"])+0.18)
            slot=max(0.25,slot_end-float(s["start"]))
            fit=raw
            if original>slot+0.08:
                retimed+=1
                tempo=min(max_tempo,max(1.0,original/slot))
                fit=segdir/f"fit-{i:05d}.wav"
                run([
                    "ffmpeg","-y","-v","error","-i",str(raw),
                    "-af",f"atempo={tempo:.6f},afade=t=out:st={max(0.0,slot-0.05):.3f}:d=0.05",
                    "-ar","48000","-ac","1","-c:a","pcm_s16le",str(fit),
                ])
                if wav_duration(fit)>slot+0.08:
                    trimmed=segdir/f"trim-{i:05d}.wav"
                    run([
                        "ffmpeg","-y","-v","error","-i",str(fit),
                        "-af",f"atrim=duration={slot:.3f},afade=t=out:st={max(0.0,slot-0.06):.3f}:d=0.06",
                        "-ar","48000","-ac","1","-c:a","pcm_s16le",str(trimmed),
                    ])
                    fit=trimmed
                    hard_trim+=1
            fitted.append((s,fit))

        done=min(off+len(batch),len(segments))
        elapsed=max(0.001,time.monotonic()-started)
        rate=done/elapsed
        remain=(len(segments)-done)/rate if rate>0 else 0
        heartbeat(
            "tts",
            f"VieNeu {done}/{len(segments)} eta={remain/60:.1f}m retimed={retimed} hard_trim={hard_trim}",
        )

    del tts
    gc.collect()

    with wave.open(str(fitted[0][1]),"rb") as wf:
        rate=wf.getframerate()
    duration=max(float(s["end"]) for s in segments)+1.0
    canvas=np.zeros(max(1,int(math.ceil(duration*rate))),dtype=np.float32)
    for s,path in fitted:
        with wave.open(str(path),"rb") as wf:
            data=np.frombuffer(wf.readframes(wf.getnframes()),dtype="<i2").astype(np.float32)
        pos=max(0,int(round(float(s["start"])*rate)))
        stop=min(len(canvas),pos+len(data))
        if pos<len(canvas):
            canvas[pos:stop]+=data[:stop-pos]
    canvas=np.clip(canvas,-32768,32767).astype("<i2")
    with wave.open(str(VOICE),"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate); wf.writeframes(canvas.tobytes())

    heartbeat("mixing","Mixing Vietnamese voice with original video; copying video stream")
    mix=(
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,highpass=f=35,volume=0.62[orig];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.06,asplit=2[sc][voice];"
        "[orig][sc]sidechaincompress=threshold=0.010:ratio=14:attack=8:release=240:makeup=1[duck];"
        "[duck][voice]amix=inputs=2:weights='0.68 1':normalize=0,loudnorm=I=-18.0:TP=-2.5:LRA=6,"
        "alimiter=limit=0.94[outa]"
    )
    run([
        "ffmpeg","-y","-v","error","-i",str(SOURCE),"-i",str(VOICE),
        "-filter_complex",mix,"-map","0:v:0","-map","[outa]",
        "-c:v","copy","-c:a","aac","-b:a","160k","-movflags","+faststart","-shortest",str(OUT),
    ])
    if not OUT.exists() or OUT.stat().st_size<1_000_000:
        raise RuntimeError("final video missing or unexpectedly small")

    meta["tts_engine"]="VieNeu-TTS-v3-Turbo"
    meta["tts_backend"]="onnx-int8"
    meta["tts_voice"]=voice_name
    meta["tts_sample_rate"]=48000
    meta["retimed_segments"]=retimed
    meta["hard_trim_segments"]=hard_trim
    meta["voice_render_seconds"]=round(time.monotonic()-started,2)
    meta["final_video_bytes"]=OUT.stat().st_size
    return meta

def upload_to_youtube(url:str,path:Path)->str:
    heartbeat("youtube_upload",f"Uploading final video directly from worker bytes={path.stat().st_size}")
    p=subprocess.run([
        "curl","--fail-with-body","--retry","3","--retry-all-errors","--retry-delay","3",
        "-sS","-X","PUT",url,"-H","content-type: video/mp4","--data-binary","@"+str(path),
    ],capture_output=True,text=True,check=True)
    body=json.loads(p.stdout or "{}")
    video_id=str(body.get("id") or "").strip()
    if not video_id:
        raise RuntimeError("youtube upload response missing video id: "+(p.stdout or "")[:1000])
    return video_id

def main()->None:
    if BOT2_MODE:
        mounted_pattern=str(JOB.get("mounted_source_glob") or "/kaggle/input/**/source.mp4")
        deadline=time.monotonic()+60.0
        matches=[]
        while time.monotonic()<deadline:
            matches=[Path(p) for p in glob.glob(mounted_pattern,recursive=True) if Path(p).is_file()]
            if len(matches)==1:
                break
            time.sleep(5)
        if len(matches)!=1:
            visible=[str(p) for p in Path("/kaggle/input").rglob("*") if p.is_file()][:80]
            raise RuntimeError(
                f"expected exactly one mounted source, found {len(matches)} for {mounted_pattern}; "
                f"visible_files={visible}"
            )
        mounted=matches[0]
        if mounted.stat().st_size<1_000_000:
            raise RuntimeError(f"mounted source unexpectedly small: {mounted.stat().st_size}")
        if SOURCE.exists() or SOURCE.is_symlink():
            SOURCE.unlink()
        SOURCE.symlink_to(mounted)
        heartbeat("source_ready",f"Mounted source ready bytes={mounted.stat().st_size} path={mounted}")

    heartbeat("installing","Installing optimized local runtime")
    if BOT2_MODE and WORKER_PHASE=="tts_only":
        deps=[
            "vieneu>=3.8.1,<4",
            "soundfile>=0.13,<1",
            "numpy",
        ]
    else:
        deps=[
            "faster-whisper>=1.1,<2",
            "transformers>=5.6,<6",
            "accelerate<2",
            "bitsandbytes>=0.45,<1",
            "sentencepiece",
            "sacremoses",
            "soundfile>=0.13,<1",
            "numpy",
        ]
    if not BOT2_MODE:
        deps.insert(0,"yt-dlp>=2026.1")
    run([sys.executable,"-m","pip","install","--quiet",*deps])

    cfg=dict(JOB.get("config") or {}) if BOT2_MODE else post("/worker-config",{"job_id":JOB_ID})
    source_video_id=str(cfg["source_video_id"])
    pipeline_started=time.monotonic()

    if BOT2_MODE and WORKER_PHASE=="tts_only":
        checkpoint=fetch_translation_checkpoint()
        if not checkpoint.get("found"):
            raise RuntimeError("translation checkpoint missing for CPU TTS")
        payload=checkpoint.get("payload") or {}
        segments=list(payload.get("segments_data") or [])
        if not segments:
            raise RuntimeError("translation checkpoint has no timed segments for CPU TTS")
        for s in segments:
            s["index"]=int(s["index"])
            s["start"]=float(s["start"])
            s["end"]=float(s["end"])
            s["text"]=clean(s.get("text",""))
            s["vi"]=validate_vi(s.get("vi",""),s["text"],field=f"cpu tts segment {s['index']}")
            s["tts_slot"]=float(s.get("tts_slot") or max(0.25,s["end"]-s["start"]))
            s["max_words"]=int(s.get("max_words") or max(2,math.ceil(s["tts_slot"]*3.0)))
            s["fit_words"]=int(s.get("fit_words") or fit_word_limit(s))
        translated_title=clean(str(payload.get("translated_title") or cfg.get("series_title") or cfg.get("title") or "Hidden Beyond"))
        translation_profile=dict(payload.get("translation_profile") or {})
        voice_name=str(payload.get("voice_name") or "Ngọc Linh")
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
            "translation_profile":translation_profile,
            "translation_checkpoint_revision":CHECKPOINT_REVISION,
            "tts_voice":"vieneu-v3-turbo-onnx-int8",
            "style":STYLE,
            "voice_name":voice_name,
            "compute_stage":"cpu_tts_mix_upload",
            "timed_segments":[{
                "index":s["index"],"start":round(s["start"],3),"end":round(s["end"],3),
                "zh":s["text"],"vi":s["vi"],"max_words":s["max_words"],"fit_words":s["fit_words"],
            } for s in segments],
        }
        heartbeat("tts",f"CPU TTS resume: {len(segments)} translated segments")
        meta=render_voice_and_video(segments,meta)
        META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        video_id=upload_to_youtube(str(cfg["youtube_upload_url"]),OUT)
        heartbeat("youtube_uploaded",f"YouTube accepted video id={video_id}")
        result=post("/worker-complete",{
            "source_video_id":source_video_id,
            "youtube_video_id":video_id,
        })
        print("HB_BOT2_CPU_COMPLETE",json.dumps(result,ensure_ascii=False),flush=True)
        return

    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    device="cuda" if torch.cuda.is_available() else "cpu"
    compute="float16" if device=="cuda" else "int8"
    if not BOT2_MODE:
        heartbeat("source_download","Downloading source once on the direct worker")
        client_sets=[None,"android_vr,web_safari","tv,mweb"]
        last_source_error=""
        for attempt,clients in enumerate(client_sets,1):
            if SOURCE.exists():
                SOURCE.unlink()
            cmd=[
                sys.executable,"-m","yt_dlp","--no-playlist",
                "--retries","5","--fragment-retries","5",
                "--remote-components","ejs:github",
            ]
            if shutil.which("node"):
                cmd += ["--js-runtimes","node"]
            if clients:
                cmd += ["--extractor-args",f"youtube:player_client={clients}"]
            cmd += [
                "-f","bv*[height<=720]+ba/b[height<=720]/b",
                "--merge-output-format","mp4","-o",str(SOURCE),str(cfg["source_url"]),
            ]
            proc=subprocess.run(cmd,text=True,capture_output=True)
            if proc.returncode==0 and SOURCE.exists() and SOURCE.stat().st_size>=1_000_000:
                heartbeat(
                    "source_ready",
                    f"Source ready attempt={attempt} clients={clients or 'default'} bytes={SOURCE.stat().st_size}",
                )
                break
            tail=(proc.stderr or proc.stdout or "").strip()[-1200:]
            last_source_error=tail or f"yt-dlp exit={proc.returncode}"
            heartbeat(
                "source_retry",
                f"Source attempt {attempt}/{len(client_sets)} failed clients={clients or 'default'}; retrying bounded fallback",
            )
            if attempt<len(client_sets):
                time.sleep(12*attempt)
        else:
            raise RuntimeError("source download failed after bounded retries: "+last_source_error)

    transcript_started=time.monotonic()
    if BOT2_MODE:
        caption_result=None
        heartbeat("extracting_audio","Bot2 mounted source verified; using local ASR")
    else:
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
        heartbeat("extracting_audio","No usable Chinese captions; extracting compact audio locally")
        run([
            "ffmpeg","-y","-v","error","-i",str(SOURCE),
            "-vn","-ac","1","-ar","16000","-c:a","libmp3lame","-b:a","48k",str(AUDIO),
        ])
        if AUDIO.stat().st_size<100000:
            raise RuntimeError("extracted source audio is unexpectedly small")
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
    for s in segments:
        s["text"]=collapse_cjk_asr_repetition(s.get("text",""))
    if len(segments)<10:
        raise RuntimeError(f"too few transcript segments: {len(segments)}")
    for i,s in enumerate(segments,1):
        s["index"]=i
        s["slot"]=max(0.30,float(s["end"])-float(s["start"]))
        next_start=float(segments[i]["start"]) if i<len(segments) else float(s["end"])+max(1.2,float(STYLE["max_extra_gap"]))+float(STYLE["min_pause_between_cues"])
        speech_tail=float(s["end"])+0.18
        available_end=min(next_start-float(STYLE["min_pause_between_cues"]),speech_tail)
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

    translation_profile=infer_translation_profile(cfg,segments,cfg.get("translation_profile") or {})
    global ACTIVE_PROFILE
    ACTIVE_PROFILE=translation_profile
    pending_segments=segments
    heartbeat(
        "translation_profile",
        f"Profile ready: {translation_profile.get('genre','general')}; translating {len(segments)} segments",
    )

    translation_started=time.monotonic()
    segment_signature=translation_segment_signature(segments)
    checkpoint=load_translation_checkpoint(segment_signature,len(segments))
    translated={}
    review_ids=set()
    hard_reasons={}
    resume_cursor=0
    if checkpoint:
        try:
            translated={
                int(k):clean(v)
                for k,v in dict(checkpoint.get("translated") or {}).items()
                if clean(v)
            }
            review_ids={int(x) for x in (checkpoint.get("review_ids") or [])}
            hard_reasons={int(k):str(v) for k,v in dict(checkpoint.get("hard_reasons") or {}).items()}
            resume_cursor=max(0,min(len(segments),int(checkpoint.get("cursor") or 0)))
            if any(i not in translated for i in range(1,resume_cursor+1)):
                raise ValueError("checkpoint translated prefix incomplete")
            if checkpoint.get("translation_profile"):
                translation_profile=dict(checkpoint["translation_profile"])
                ACTIVE_PROFILE=translation_profile
            heartbeat(
                "translating",
                f"Resuming translation checkpoint {resume_cursor}/{len(segments)}; "
                f"review={len(review_ids)} hard={len(hard_reasons)}",
            )
        except Exception as exc:
            print("HB_CHECKPOINT_INVALID",repr(exc),flush=True)
            translated={}
            review_ids=set()
            hard_reasons={}
            resume_cursor=0
    qwen_reviewed=0
    fallback_segments=0
    structural_rescue_segments=0
    fast_path_used=False
    primary_model="tencent/Hy-MT2-7B"

    if WORKER_PHASE=="translation_only" and resume_cursor==len(segments) and len(translated)==len(segments):
        for seg in segments:
            idx=seg["index"]
            translated[idx]=validate_translation_pair(
                translated[idx],seg["text"],field=f"checkpoint segment {idx}",enforce_intent=False
            )
            seg["vi"]=translated[idx]
        translated_title=clean(str(checkpoint.get("translated_title") or cfg.get("series_title") or cfg.get("title") or "Hidden Beyond"))
        save_translation_checkpoint(
            "translation_complete",len(segments),segment_signature,translated,set(),{},
            translation_profile,
            extra_payload={
                "translated_title":translated_title,
                "voice_name":"Ngọc Linh",
                "segments_data":[{
                    "index":seg["index"],"start":round(float(seg["start"]),3),"end":round(float(seg["end"]),3),
                    "text":seg["text"],"vi":seg["vi"],"tts_slot":round(float(seg["tts_slot"]),3),
                    "max_words":int(seg["max_words"]),"fit_words":int(seg["fit_words"]),
                } for seg in segments],
            },
        )
        heartbeat("translation_complete",f"Translation checkpoint ready: {len(segments)}/{len(segments)}; GPU stage complete")
        return

    if device!="cuda":
        raise RuntimeError("Hidden Beyond production translation requires Kaggle GPU")

    heartbeat("translating",f"Loading {primary_model}; translating {len(segments)} segments")
    tok=AutoTokenizer.from_pretrained(primary_model,trust_remote_code=True)
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    quant_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model=AutoModelForCausalLM.from_pretrained(
        primary_model,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    by_index={s["index"]:s for s in segments}

    def context_for(s:dict)->str:
        idx=s["index"]
        before=[]
        after=[]
        for j in range(max(1,idx-4),idx):
            if j in by_index:
                before.append(f"{j}:{clean(by_index[j].get('text',''))}")
        for j in range(idx+1,min(len(segments),idx+4)+1):
            if j in by_index:
                after.append(f"{j}:{clean(by_index[j].get('text',''))}")
        prior_vi=[]
        for j in range(max(1,idx-3),idx):
            if translated.get(j):
                prior_vi.append(f"{j}:{clean(translated[j])}")
        title=clean(str(cfg.get("series_title") or cfg.get("title") or ""))
        parts=[]
        if title:
            parts.append("作品="+title)
        if before:
            parts.append("前文="+" | ".join(before))
        if prior_vi:
            parts.append("已译前文="+" | ".join(prior_vi))
        if after:
            parts.append("后文="+" | ".join(after))
        return "\n".join(parts)

    def prompt_for(s:dict, *, repair:bool=False, current:str="", failure_reason:str="")->str:
        terms=glossary_pairs(s["text"])
        term_text=""
        if terms:
            term_text="固定术语："+"；".join(f"{zh}={vi}" for zh,vi in terms)+"\n"
        limit=int(s.get("fit_words") or 18)
        if repair:
            structural_rule=""
            policy=repair_prompt_policy(failure_reason)
            include_previous=bool(policy["include_previous"])
            include_context=bool(policy["include_context"])
            if "still contains CJK" in failure_reason:
                structural_rule=(
                    "上一版含中文或解释性内容。现在重新独立翻译。"
                    "输出必须只有一行越南语对白；禁止汉字、禁止解释、禁止前缀、禁止讨论翻译过程。\n"
                )
            elif "repetition loop" in failure_reason:
                structural_rule=(
                    "上一版出现重复循环。不要参考上一版措辞，必须从原文重新独立翻译；"
                    "修正版不得重复词句。\n"
                )
            elif " empty" in failure_reason or failure_reason.endswith("empty"):
                structural_rule="上一版为空；必须从原文重新独立翻译并输出非空越南语。\n"
            return (
                term_text
                +"请重新翻译下面一句中文为自然、准确、简洁的越南语影视对白。\n"
                +"必须忠实原意，不添加信息，不遗漏否定、疑问、数字、人物关系和专有名词。\n"
                +structural_rule
                +profile_prompt_rule(translation_profile)+"\n"
                +f"尽量控制在 {limit} 个越南语词以内，但不要为了缩短而改变原意。\n"
                +(f"上下文：{context_for(s)}\n" if include_context else "")
                +(f"上一版：{current}\n" if include_previous and current else "")
                +"只输出修正后的越南语一句话，不解释。\n"
                +f"原文：{s['text']}"
            )
        return (
            term_text
            +"〖背景信息〗\n"+context_for(s)+"\n"
            +"〖翻译要求〗\n"
            +"1. 忠实传达原意，不添加原文没有的信息，不遗漏关键含义。\n"
            +"2. 必须先结合前后文判断谁对谁说话、人物身份、辈分、关系和当前事件，再翻译本句；不得只按字面孤立翻译。\n"
            +"3. 保持人物关系、否定、数字、疑问、因果和情绪强度；称呼、专有名词和术语在同一系列中保持一致。\n"
            +"4. "+profile_prompt_rule(translation_profile)+"\n"
            +f"5. 尽量简洁，目标不超过 {limit} 个越南语词，适合配音。\n"
            +"6. 只输出越南语译文，不解释。\n"
            +"〖待翻译文本〗\n"+s["text"]
        )

    def rescue_prompt_for(s:dict, failure_reason:str)->str:
        """Last bounded structural rescue: source-only, no poisoned history."""
        terms=glossary_pairs(s["text"])
        term_text=""
        if terms:
            term_text="固定术语："+"；".join(f"{zh}={vi}" for zh,vi in terms)+"\n"
        limit=int(s.get("fit_words") or 18)
        return (
            term_text
            +"这是结构性故障恢复翻译。上一版不可参考。\n"
            +"请只根据原文，从零翻译成一行自然、准确、简洁的越南语影视对白。\n"
            +"禁止汉字、解释、前缀、重复词句或循环；不得添加原文没有的信息。\n"
            +profile_prompt_rule(translation_profile)+"\n"
            +f"尽量控制在 {limit} 个越南语词以内，但不得改变原意。\n"
            +f"故障类型：{failure_reason or 'structural_invalid'}\n"
            +f"原文：{s['text']}"
        )

    def soft_intent_review(source:str,value:str)->bool:
        return bool(translation_review_reasons(value,source))

    hard_invalid=set(hard_reasons)
    batch_size=6
    cursor=resume_cursor
    last_checkpoint_cursor=resume_cursor
    with torch.inference_mode():
        while cursor<len(segments):
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
                inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=512)
                inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
                out=model.generate(
                    **inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,
                    use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size<=4:
                    raise
                batch_size=max(4,batch_size//2)
                heartbeat("translating",f"GPU memory guard: translation batch={batch_size}")
                continue

            generated=out[:,inp["input_ids"].shape[1]:]
            vis=tok.batch_decode(generated,skip_special_tokens=True)
            for s,vi in zip(batch,vis,strict=True):
                idx=s["index"]
                candidate=clean(vi)
                translated[idx]=candidate
                try:
                    translated[idx]=validate_translation_pair(
                        candidate,s["text"],field=f"segment {idx}",enforce_intent=False
                    )
                except Exception as exc:
                    hard_invalid.add(idx)
                    hard_reasons[idx]=str(exc)
                    review_ids.add(idx)
                if soft_intent_review(s["text"],candidate):
                    review_ids.add(idx)
                if vi_word_count(candidate)>s["fit_words"]:
                    review_ids.add(idx)

            cursor+=len(batch)
            if cursor==len(segments) or cursor-last_checkpoint_cursor>=300:
                save_translation_checkpoint(
                    "translating",cursor,segment_signature,translated,
                    review_ids,hard_reasons,translation_profile,
                )
                last_checkpoint_cursor=cursor
            elapsed=max(0.001,time.monotonic()-translation_started)
            heartbeat(
                "translating",
                f"Hy-MT2 {cursor}/{len(segments)} ({cursor*100.0/len(segments):.1f}%); "
                f"{cursor/elapsed:.1f} seg/s; review={len(review_ids)} hard={len(hard_invalid)}",
            )

    # Bounded two-stage repair using the same already-loaded Hy-MT2:
    # 1) normal targeted repair for review items;
    # 2) source-only structural rescue only for the rare items still invalid.
    # No OPUS/Qwen/model handoff and no second model load in production.
    repair_items=[s for s in segments if s["index"] in review_ids]
    if repair_items:
        heartbeat(
            "translation_repair",
            f"Single targeted repair pass: {len(repair_items)} of {len(segments)} segments",
        )
        hard_after=[]
        hard_after_details={}
        with torch.inference_mode():
            for off in range(0,len(repair_items),4):
                batch=repair_items[off:off+4]
                chats=[
                    tok.apply_chat_template(
                        [{"role":"user","content":prompt_for(
                            s,repair=True,current=translated.get(s["index"],""),
                            failure_reason=hard_reasons.get(s["index"],"")
                        )}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for s in batch
                ]
                inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=512)
                inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
                out=model.generate(
                    **inp,max_new_tokens=64,do_sample=False,repetition_penalty=1.08,
                    use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                )
                generated=out[:,inp["input_ids"].shape[1]:]
                vis=tok.batch_decode(generated,skip_special_tokens=True)
                for s,vi in zip(batch,vis,strict=True):
                    idx=s["index"]
                    candidate=clean(vi)
                    translated[idx]=candidate
                    try:
                        translated[idx]=validate_translation_pair(
                            candidate,s["text"],field=f"repair segment {idx}",enforce_intent=False
                        )
                    except Exception as exc:
                        hard_after.append(idx)
                        hard_after_details[idx]={
                            "reason":str(exc),
                            "source":clean(s["text"])[:160],
                            "output":candidate[:160],
                        }

        if hard_after:
            rescue_items=[by_index[idx] for idx in hard_after if idx in by_index]
            structural_rescue_segments=len(rescue_items)
            heartbeat(
                "translation_repair",
                f"Structural rescue pass: {len(rescue_items)} segments; source-only deterministic decode",
            )
            rescue_failed=[]
            rescue_details={}
            with torch.inference_mode():
                for off in range(0,len(rescue_items),2):
                    batch=rescue_items[off:off+2]
                    chats=[
                        tok.apply_chat_template(
                            [{"role":"user","content":rescue_prompt_for(
                                s,hard_after_details.get(s["index"],{}).get("reason","")
                            )}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for s in batch
                    ]
                    inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=384)
                    inp={k:v.to(device) for k,v in inp.items() if k!="token_type_ids"}
                    out=model.generate(
                        **inp,max_new_tokens=48,do_sample=False,repetition_penalty=1.20,
                        no_repeat_ngram_size=3,use_cache=True,
                        pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                    )
                    generated=out[:,inp["input_ids"].shape[1]:]
                    vis=tok.batch_decode(generated,skip_special_tokens=True)
                    for s,vi in zip(batch,vis,strict=True):
                        idx=s["index"]
                        candidate=clean(vi)
                        translated[idx]=candidate
                        try:
                            translated[idx]=validate_translation_pair(
                                candidate,s["text"],field=f"rescue segment {idx}",enforce_intent=False
                            )
                        except Exception as exc:
                            rescue_failed.append(idx)
                            rescue_details[idx]={
                                "reason":str(exc),
                                "source":clean(s["text"])[:160],
                                "output":candidate[:160],
                            }

            if rescue_failed:
                detail=" | ".join(
                    f"{idx}:{rescue_details.get(idx,{})}"
                    for idx in rescue_failed[:3]
                )
                raise RuntimeError(
                    f"translation hard quality unresolved after bounded structural rescue "
                    f"{len(rescue_failed)} segments: "
                    +",".join(map(str,rescue_failed[:30]))+"; details="+detail
                )

    title_item={
        "index":0,
        "text":str(cfg.get("title") or ""),
        "fit_words":20,
    }
    try:
        title_chat=tok.apply_chat_template(
            [{"role":"user","content":(
                "将以下标题翻译为自然越南语，保持原作品题材风格和专有名词，只输出越南语标题，不解释：\n\n"
                +title_item["text"]
            )}],
            tokenize=False,
            add_generation_prompt=True,
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

    overlong_ids=[
        s["index"] for s in segments
        if translated.get(s["index"]) and vi_word_count(translated[s["index"]])>s["fit_words"]
    ]
    invalid_ids=[]
    del model,tok
    gc.collect()
    torch.cuda.empty_cache()

    missing=[s for s in segments if not translated.get(s["index"])]
    if missing:
        raise RuntimeError(f"translation missing {len(missing)} segments")

    final_invalid=[]
    for s in segments:
        try:
            translated[s["index"]]=validate_translation_pair(
                translated[s["index"]],s["text"],field=f"final semantic segment {s['index']}",
                enforce_intent=False,
            )
            # Segment duration is a review signal, not a production hard-fail.
        except Exception:
            final_invalid.append(s["index"])
    if final_invalid:
        raise RuntimeError(
            f"translation quality unresolved {len(final_invalid)} segments: "
            +",".join(map(str,final_invalid[:30]))
        )
    estimated_overlong_after_review=sum(
        1 for s in segments
        if vi_word_count(translated[s["index"]])>s["fit_words"]
    )

    for s in segments:
        s["vi"]=translated[s["index"]]

    save_translation_checkpoint(
        "translation_complete",len(segments),segment_signature,translated,set(),{},
        translation_profile,
        extra_payload={
            "translated_title":translated_title,
            "voice_name":"Ngọc Linh",
            "segments_data":[{
                "index":seg["index"],"start":round(float(seg["start"]),3),"end":round(float(seg["end"]),3),
                "text":seg["text"],"vi":seg["vi"],"tts_slot":round(float(seg["tts_slot"]),3),
                "max_words":int(seg["max_words"]),"fit_words":int(seg["fit_words"]),
            } for seg in segments],
        },
    )

    translation_seconds=round(time.monotonic()-translation_started,2)
    heartbeat(
        "translation_complete",
        f"Translation complete: {len(segments)}/{len(segments)} (100%) in {translation_seconds:.1f}s; checkpoint ready",
    )
    if BOT2_MODE and WORKER_PHASE=="translation_only":
        return

    voice_mode="vieneu-v3-turbo-onnx-int8"
    voice_name="Ngọc Linh"
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
        "translation_profile":translation_profile,
        "translation_editor":"not_used",
        "transcript_source":transcript_source,
        "whisper_model":whisper_name,
        "detected_language":detected_language,
        "language_probability":language_probability,
        "tts_voice":voice_mode,
        "translation_mode":"hy-mt2-context-window-bounded-structural-rescue-v11",
        "timing_mode":"source-speech-window-sync-v2",
        "translation_quality_profile":"asr-dedup-source-only-structural-rescue-v8",
        "translation_quality_rules":["fidelity","no_addition","no_omission","terminology_consistency","negation_preserved","question_intent_preserved","number_preserved","context_aware"],
        "style":STYLE,
        "voice_name":voice_name,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_efficiency_mode":"single-worker-caption-first-batched-asr-hymt2-context-v10-vieneu-remux",
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
        "estimated_overlong_after_review":estimated_overlong_after_review,
        "fallback_segments":fallback_segments,
        "structural_rescue_segments":structural_rescue_segments,
        "fast_path_used":fast_path_used,
        "timed_segments":[{
            "index":s["index"],
            "start":round(float(s["start"]),3),
            "end":round(float(s["end"]),3),
            "zh":s["text"],
            "vi":s["vi"],
            "max_words":s["max_words"],
            "fit_words":s["fit_words"],
        } for s in segments],
    }
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    heartbeat("tts","Rendering Vietnamese voice on the same worker")
    meta=render_voice_and_video(segments,meta)
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    if not BOT2_MODE:
        heartbeat("evidence","Uploading small subtitle and metadata evidence")
        upload(cfg["subtitle_upload_url"],SRT,"text/plain")
        upload(cfg["metadata_upload_url"],META,"application/json")

    video_id=upload_to_youtube(str(cfg["youtube_upload_url"]),OUT)
    heartbeat("youtube_uploaded",f"YouTube accepted video id={video_id}")
    if BOT2_MODE:
        result=post("/worker-complete",{
            "source_video_id":source_video_id,
            "youtube_video_id":video_id,
        })
        print("HB_BOT2_COMPLETE",json.dumps(result,ensure_ascii=False),flush=True)
    else:
        result=post("/complete",{
            "job_id":JOB_ID,
            "youtube_video_id":video_id,
            "translated_title":translated_title,
            "translation_profile":translation_profile,
        })
        print("HB_DIRECT_COMPLETE",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    thread=threading.Thread(target=heartbeat_loop,daemon=True)
    thread.start()
    try:
        main()
    except Exception as exc:
        print("HB_AI_FAIL",repr(exc),flush=True)
        try:
            if BOT2_MODE:
                post("/worker-fail",{
                    "source_video_id":str(JOB["source_video_id"]),
                    "stage":bot2_stage_name(_stage["name"]),
                    "error":repr(exc),
                })
            else:
                post("/fail",{"job_id":JOB_ID,"error":repr(exc)})
        except Exception as fail_exc:
            print("HB_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise
    finally:
        _stop.set()

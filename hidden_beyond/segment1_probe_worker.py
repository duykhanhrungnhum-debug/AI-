#!/usr/bin/env python3
from __future__ import annotations
import gc,json,math,re,shutil,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen

WORK=Path("/kaggle/working")
RESULT=WORK/"segment1-probe-result.json"
VIDEO_URL="https://www.youtube.com/watch?v=vaY4URFw0cg"

def run(cmd,check=True):
    print("+"," ".join(map(str,cmd)),flush=True)
    return subprocess.run(cmd,check=check,text=True,capture_output=not check)

run([
    sys.executable,"-m","pip","install","--quiet",
    "yt-dlp>=2026.1","faster-whisper>=1.1,<2","transformers>=5.6,<6",
    "accelerate<2","bitsandbytes>=0.45,<1","sentencepiece","sacremoses"
])

import torch
from faster_whisper import BatchedInferencePipeline,WhisperModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REPEAT_RE=re.compile(r"\b([\wÀ-ỹ]+)(?:\s+\1)+\b",re.IGNORECASE)
TAG_RE=re.compile(r"<[^>]+>")
STYLE={"target_segment_seconds":3.2,"hard_max_segment_seconds":5.5,"min_pause_between_cues":0.20}
DIALOGUE_GLOSSARY={"自寻死路":"tự tìm đường chết","别给脸不要脸":"đừng có không biết điều","一成":"một thành"}
FANTASY_GLOSSARY={
"修仙":"tu tiên","修士":"tu sĩ","灵气":"linh khí","灵根":"linh căn","炼气":"Luyện Khí","筑基":"Trúc Cơ",
"金丹":"Kim Đan","元婴":"Nguyên Anh","渡劫":"Độ Kiếp","宗门":"tông môn","师尊":"sư tôn","师兄":"sư huynh",
"师姐":"sư tỷ","师弟":"sư đệ","师妹":"sư muội","掌门":"chưởng môn","长老":"trưởng lão","道友":"đạo hữu",
"法宝":"pháp bảo","丹药":"đan dược","功法":"công pháp","秘境":"bí cảnh","洞府":"động phủ","魔修":"ma tu",
"正道":"chính đạo","天道":"thiên đạo","飞升":"phi thăng","境界":"cảnh giới"
}

def clean(value):
    value=TAG_RE.sub("",str(value)).replace("\u200b"," ")
    value=re.sub(r"\s+"," ",value).strip().strip('“”"')
    previous=None
    while previous!=value:
        previous=value
        value=REPEAT_RE.sub(r"\1",value)
    return re.sub(r"\s+([,.;:!?])",r"\1",value).strip()

def has_ngram_loop(value):
    words=re.findall(r"[\wÀ-ỹ]+",value.lower())
    for n in (2,3,4):
        for i in range(0,max(0,len(words)-n*3+1)):
            gram=words[i:i+n]
            if gram and words[i+n:i+2*n]==gram and words[i+2*n:i+3*n]==gram:
                return True
    return False

def hard_validate(text):
    value=clean(text)
    if not value: return value,"empty"
    if CJK_RE.search(value): return value,"still contains CJK"
    if REPEAT_RE.search(value) or has_ngram_loop(value): return value,"repetition loop"
    return value,""

def split_words_to_dialogue(words):
    usable=[]
    for w in words:
        text=str(w.get("text") or "")
        if not clean(text): continue
        start=max(0.0,float(w.get("start") or 0))
        end=max(start+0.04,float(w.get("end") or start+0.08))
        if end-start>1.8: end=start+1.8
        usable.append({"start":start,"end":end,"text":text})
    strong=set("。！？!?；;"); soft=set("，,、：:")
    out=[]; cur=[]; hard=float(STYLE["hard_max_segment_seconds"])
    for i,w in enumerate(usable):
        if cur:
            gap=w["start"]-cur[-1]["end"]; dur=cur[-1]["end"]-cur[0]["start"]
            if gap>=0.55 or (gap>=0.32 and dur>=1.40):
                out.append({"start":cur[0]["start"],"end":cur[-1]["end"],"text":clean("".join(x["text"] for x in cur))}); cur=[]
        if cur and w["end"]-cur[0]["start"]>hard:
            out.append({"start":cur[0]["start"],"end":cur[-1]["end"],"text":clean("".join(x["text"] for x in cur))}); cur=[]
        cur.append(w)
        start=cur[0]["start"]; end=w["end"]; text=clean("".join(x["text"] for x in cur)); duration=end-start
        next_gap=(usable[i+1]["start"]-end) if i+1<len(usable) else 9.0
        last=text[-1] if text else ""; target=float(STYLE["target_segment_seconds"])
        boundary=((last in strong and duration>=1.20) or (last in soft and duration>=2.20)
            or (next_gap>=0.55 and duration>=0.70) or (next_gap>=0.32 and duration>=1.40)
            or (duration>=target and (next_gap>=0.18 or last in strong or last in soft))
            or duration>=hard or len(CJK_RE.findall(text))>=32 or len(text)>=64)
        if boundary: out.append({"start":start,"end":end,"text":text}); cur=[]
    if cur: out.append({"start":cur[0]["start"],"end":cur[-1]["end"],"text":clean("".join(x["text"] for x in cur))})
    fixed=[]
    for s in out:
        dur=s["end"]-s["start"]
        if fixed:
            prev=fixed[-1]; prev_dur=prev["end"]-prev["start"]; gap=s["start"]-prev["end"]; candidate=clean(prev["text"]+s["text"])
            if gap<=0.35 and s["end"]-prev["start"]<=hard and (dur<0.90 or prev_dur<0.90) and len(candidate)<=64:
                prev["end"]=s["end"]; prev["text"]=candidate; continue
        fixed.append(s)
    return fixed

audio=WORK/"probe.mp3"
clients=[None,"android_vr,web_safari","tv,mweb"]
err=""
for attempt,client in enumerate(clients,1):
    if audio.exists(): audio.unlink()
    cmd=[sys.executable,"-m","yt_dlp","--no-playlist","--retries","3","--fragment-retries","3","--remote-components","ejs:github"]
    if shutil.which("node"): cmd += ["--js-runtimes","node"]
    if client: cmd += ["--extractor-args",f"youtube:player_client={client}"]
    cmd += ["--download-sections","*0-90","-f","ba/b","-x","--audio-format","mp3","--audio-quality","5","-o",str(audio),VIDEO_URL]
    p=subprocess.run(cmd,text=True,capture_output=True)
    if p.returncode==0 and audio.exists() and audio.stat().st_size>10000: break
    err=(p.stderr or p.stdout or "")[-1500:]
else:
    raise RuntimeError("probe source failed: "+err)

whisper=WhisperModel("large-v3-turbo",device="cuda",compute_type="float16")
transcriber=BatchedInferencePipeline(model=whisper)
seg_iter,info=transcriber.transcribe(
    str(audio),language="zh",vad_filter=True,batch_size=16,beam_size=1,
    condition_on_previous_text=False,word_timestamps=True,
    vad_parameters={"min_silence_duration_ms":240,"speech_pad_ms":80},
)
words=[]
for s in seg_iter:
    for w in (getattr(s,"words",None) or []):
        wt=str(getattr(w,"word","") or "")
        if not clean(wt): continue
        words.append({"start":float(getattr(w,"start",s.start) or s.start),"end":float(getattr(w,"end",s.end) or s.end),"text":wt})
segments=split_words_to_dialogue(words)
if not segments: raise RuntimeError("probe ASR produced no segments")
s=segments[0]
s["index"]=1
next_start=segments[1]["start"] if len(segments)>1 else s["end"]+1.2
s["tts_slot"]=max(0.25,next_start-STYLE["min_pause_between_cues"]-s["start"])
s["fit_words"]=max(2,int(math.ceil(s["tts_slot"]*4.8)))

del transcriber,whisper
gc.collect(); torch.cuda.empty_cache()

model_name="tencent/Hy-MT2-7B"
tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
model=AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quant,device_map="auto",low_cpu_mem_usage=True,trust_remote_code=True)
model.eval()

def terms_for(text):
    out=[]
    for table in (DIALOGUE_GLOSSARY,FANTASY_GLOSSARY):
        for zh,vi in table.items():
            if zh in text: out.append((zh,vi))
    return out[:12]

def generate(prompt):
    chat=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inp=tok([chat],return_tensors="pt",padding=True,truncation=True,max_length=512)
    inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
    with torch.inference_mode():
        out=model.generate(**inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    gen=out[:,inp["input_ids"].shape[1]:]
    return clean(tok.batch_decode(gen,skip_special_tokens=True)[0])

terms=terms_for(s["text"])
term_text=("固定术语："+"；".join(f"{a}={b}" for a,b in terms)+"\n") if terms else ""
base=(term_text+"请把下面中文准确、自然、简洁地翻译成越南语影视对白。"
      "只输出越南语，不解释，不添加信息。尽量适合配音。\n原文："+s["text"])
first=generate(base)
first_clean,first_reason=hard_validate(first)

repair=first_clean
repair_reason=""
if first_reason:
    rule=""
    if first_reason=="still contains CJK": rule="上一版仍含中文字符；修正版只能输出越南语，不得包含任何中文汉字。\n"
    elif first_reason=="repetition loop": rule="上一版出现重复循环；修正版不得重复词句。\n"
    elif first_reason=="empty": rule="上一版为空；必须输出非空的越南语译文。\n"
    prompt=(term_text+"请重新翻译下面一句中文为自然、准确、简洁的越南语影视对白。\n"
            +"必须忠实原意，不添加信息。\n"+rule
            +f"尽量控制在 {s['fit_words']} 个越南语词以内，但不要为了缩短而改变原意。\n"
            +f"上一版：{first_clean}\n只输出修正后的越南语一句话，不解释。\n原文：{s['text']}")
    repair=generate(prompt)
    repair,repair_reason=hard_validate(repair)

result={
    "ok": repair_reason=="",
    "segment_index":1,
    "source_start":round(s["start"],3),
    "source_end":round(s["end"],3),
    "source":s["text"],
    "fit_words":s["fit_words"],
    "first_translation":first_clean,
    "first_failure_reason":first_reason or None,
    "repair_translation":repair,
    "repair_failure_reason":repair_reason or None,
    "segments_seen_in_first_90s":len(segments),
    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
}
RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SEGMENT1_PROBE_RESULT",json.dumps(result,ensure_ascii=False),flush=True)

callback="https://rlqqcuuphjmwksanbfml.supabase.co/functions/v1/hb-segment-probe-callback"
payload=json.dumps({"probe_id":"ec2ee722-9494-4067-a931-a53ba67ef534","result":result},ensure_ascii=False).encode("utf-8")
req=Request(callback,data=payload,headers={"content-type":"application/json"},method="POST")
with urlopen(req,timeout=60) as r:
    body=r.read().decode("utf-8",errors="replace")
    print("SEGMENT1_CALLBACK",r.status,body,flush=True)

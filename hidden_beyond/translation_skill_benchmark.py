#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

# __BENCH_CONFIG_INJECT__

def post(path:str,payload:dict)->dict:
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
        post("/heartbeat",{"run_id":BENCH["run_id"],"stage":stage,"message":message})
    except Exception as exc:
        print("TRANSLATION_BENCH_HEARTBEAT_ERROR",repr(exc),flush=True)

CASES=[
    {"source":"看不懂","must_any":["không hiểu","đọc không hiểu"],"forbid":["bản quyền","yêu cầu","đội phim","quay phim"]},
    {"source":"这不就是太阳底下修炼吗","must_all":["tu luyện"],"must_any":["mặt trời","ánh nắng","dưới nắng"],"forbid":["bản quyền","đội phim","quay phim"]},
    {"source":"我果然是天选之人","must_any":["trời chọn","thiên tuyển"],"forbid":["máu trời","bản quyền","đội phim"]},
    {"source":"你已经练气三层了","must_all":["luyện khí"],"must_any":["ba","3"]},
    {"source":"师兄小心","must_all":["sư huynh"],"must_any":["cẩn thận","coi chừng"]},
    {"source":"灵气太稀薄了","must_all":["linh khí"],"must_any":["loãng","mỏng","ít"]},
    {"source":"我们去宗门","must_all":["tông môn"]},
    {"source":"这是筑基丹","must_all":["trúc cơ"],"must_any":["đan","đan dược"]},
    {"source":"他突破到金丹境了","must_all":["kim đan"],"must_any":["đột phá","cảnh"]},
    {"source":"我没有灵根","must_all":["linh căn"],"must_any":["không","chẳng"]},
    {"source":"你想飞升吗","must_all":["phi thăng"]},
    {"source":"魔修来了","must_all":["ma tu"],"must_any":["đến","tới"]},
    {"source":"这是我的法宝","must_all":["pháp bảo"]},
    {"source":"洞府里有秘境","must_all":["động phủ","bí cảnh"]},
    {"source":"天道不公","must_all":["thiên đạo"],"must_any":["bất công","không công bằng"]},
]
META=("không thể thực hiện yêu cầu","vi phạm bản quyền","đội phim","đoàn phim","quay phim","chính sách nội dung")
CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

def words(s:str)->int:
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+",s))

def validate(case:dict,out:str)->list[str]:
    v=re.sub(r"\s+"," ",out).strip().casefold()
    src=case["source"]
    reasons=[]
    if not v:
        reasons.append("empty")
    if CJK_RE.search(v):
        reasons.append("contains_cjk")
    if any(x in v for x in META) and not any(x in src for x in ("版权","拍摄","剧组","政策","请求")):
        reasons.append("meta_hallucination")
    cjk=len(CJK_RE.findall(src))
    if cjk and words(v)>max(10,math.ceil(cjk*2.6)+3):
        reasons.append("extreme_expansion")
    for x in case.get("must_all",[]):
        if x.casefold() not in v:
            reasons.append("missing:"+x)
    any_terms=[x.casefold() for x in case.get("must_any",[])]
    if any_terms and not any(x in v for x in any_terms):
        reasons.append("missing_any")
    for x in case.get("forbid",[]):
        if x.casefold() in v:
            reasons.append("forbidden:"+x)
    return reasons

def main():
    beat("installing","Installing translation benchmark runtime")
    subprocess.run([
        sys.executable,"-m","pip","install","--quiet",
        "transformers>=5.6,<6","accelerate<2","bitsandbytes>=0.45,<1","sentencepiece"
    ],check=True)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    model_name="tencent/Hy-MT2-7B"
    beat("loading_model",f"Loading {model_name} in 4-bit")
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    q=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(
        model_name,quantization_config=q,device_map="auto",
        low_cpu_mem_usage=True,trust_remote_code=True,
    )
    model.eval()
    load_seconds=time.monotonic()-started

    rows=[]
    beat("translating",f"Running {len(CASES)} focused Chinese-Vietnamese regression cases")
    with torch.inference_mode():
        for off in range(0,len(CASES),4):
            batch=CASES[off:off+4]
            prompts=[
                "将以下文本翻译为越南语，注意只需要输出翻译后的结果，不要额外解释：\n"+x["source"]
                for x in batch
            ]
            chats=[
                tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
                for p in prompts
            ]
            inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=256)
            inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
            out=model.generate(
                **inp,max_new_tokens=64,do_sample=False,repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
            )
            gen=out[:,inp["input_ids"].shape[1]:]
            texts=tok.batch_decode(gen,skip_special_tokens=True)
            for case,translated in zip(batch,texts,strict=True):
                translated=re.sub(r"\s+"," ",translated).strip()
                reasons=validate(case,translated)
                row={"source":case["source"],"translated":translated,"ok":not reasons,"reasons":reasons}
                rows.append(row)
                print("TRANSLATION_CASE",json.dumps(row,ensure_ascii=False),flush=True)

    failed=[x for x in rows if not x["ok"]]
    report={
        "ok":not failed,
        "model":model_name,
        "gpu":torch.cuda.get_device_name(0),
        "load_seconds":round(load_seconds,2),
        "cases":len(rows),
        "passed":len(rows)-len(failed),
        "failed":len(failed),
        "rows":rows,
    }
    Path("/kaggle/working/translation-skill-report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("TRANSLATION_SKILL_REPORT",json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False),flush=True)
    if failed:
        compact=" | ".join(
            f"{x['source']}=>{x['translated']}[{','.join(x['reasons'])}]"
            for x in failed
        )
        raise RuntimeError("translation regression failed: "+compact[:1200])
    post("/complete",{
        "run_id":BENCH["run_id"],
        "report":report,
        "metadata":{"model":model_name,"cases":len(rows),"passed":len(rows),"failed":len(failed)}
    })
    print("TRANSLATION_SKILL_VERIFIED",flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("TRANSLATION_SKILL_FAIL",repr(exc),flush=True)
        try:
            post("/fail",{"run_id":BENCH["run_id"],"error":repr(exc)})
        except Exception as fail_exc:
            print("TRANSLATION_SKILL_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise

#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

# __BENCH_CONFIG_INJECT__

CASES=[
 {"id":"basic_01","source":"看不懂","reference":"Không hiểu.","must_any":["không hiểu"],"forbid":["bản quyền","đội phim","quay phim"]},
 {"id":"basic_02","source":"这不就是太阳底下修炼吗","reference":"Chẳng phải đây là tu luyện dưới ánh mặt trời sao?","must_all":["tu luyện"],"must_any":["mặt trời","ánh nắng"],"question":True},
 {"id":"basic_03","source":"我果然是天选之人","reference":"Quả nhiên ta là người được trời chọn.","must_any":["trời chọn","thiên tuyển"]},
 {"id":"term_01","source":"师兄小心","reference":"Sư huynh, cẩn thận!","must_all":["sư huynh"],"must_any":["cẩn thận","coi chừng"]},
 {"id":"term_02","source":"这是筑基丹","reference":"Đây là đan dược Trúc Cơ.","must_all":["trúc cơ"],"must_any":["đan"]},
 {"id":"term_03","source":"他突破到金丹境了","reference":"Hắn đã đột phá lên cảnh giới Kim Đan.","must_all":["kim đan"],"must_any":["đột phá"]},
 {"id":"term_04","source":"你想飞升吗","reference":"Ngươi muốn phi thăng không?","must_all":["phi thăng"],"question":True},
 {"id":"term_05","source":"洞府里有秘境","reference":"Trong động phủ có một bí cảnh.","must_all":["động phủ","bí cảnh"]},
 {"id":"term_06","source":"这是我的法宝","reference":"Đây là pháp bảo của ta.","must_all":["pháp bảo"]},
 {"id":"term_07","source":"魔修来了","reference":"Ma tu đến rồi.","must_all":["ma tu"],"must_any":["đến","tới"]},
 {"id":"term_08","source":"天道不公","reference":"Thiên đạo bất công.","must_all":["thiên đạo"],"must_any":["bất công","không công bằng"]},
 {"id":"neg_01","source":"我没有灵根","reference":"Ta không có linh căn.","must_all":["linh căn"],"must_any":["không","chẳng"],"negation":True},
 {"id":"neg_02","source":"先别动手，他不是敌人。","reference":"Khoan ra tay, hắn không phải kẻ địch.","must_any":["đừng","khoan","chớ"],"negation":True},
 {"id":"neg_03","source":"她只是我的师妹，不是道侣。","reference":"Nàng chỉ là sư muội của ta, không phải đạo lữ.","must_all":["sư muội","đạo lữ"],"must_any":["không","chẳng"],"negation":True},
 {"id":"neg_04","source":"我从来没见过这么不要命的人。","reference":"Ta chưa từng thấy kẻ nào liều mạng đến vậy.","must_any":["chưa","không"],"negation":True},
 {"id":"q_01","source":"你竟敢骗本座？","reference":"Ngươi lại dám lừa bổn tọa sao?","must_any":["dám","lừa"],"question":True},
 {"id":"q_02","source":"这句话我已经说过三遍了，你还没听懂？","reference":"Câu này ta đã nói ba lần rồi, ngươi vẫn chưa hiểu sao?","must_any":["ba lần","3 lần","ba lượt"],"negation":True,"question":True},
 {"id":"num_01","source":"你已经练气三层了","reference":"Ngươi đã Luyện Khí tầng ba rồi.","must_all":["luyện khí"],"must_any":["ba","3"]},
 {"id":"num_02","source":"一百块下品灵石，一块都不能少。","reference":"Một trăm viên hạ phẩm linh thạch, không được thiếu một viên.","must_all":["linh thạch"],"must_any":["một trăm","100"],"negation":True},
 {"id":"num_03","source":"如果今晚渡劫失败，他至少要闭关十年。","reference":"Nếu tối nay Độ Kiếp thất bại, hắn phải bế quan ít nhất mười năm.","must_all":["độ kiếp"],"must_any":["mười năm","10 năm"]},
 {"id":"style_01","source":"别以为你赢了一次，就能一直赢下去。","reference":"Đừng tưởng thắng một lần là có thể thắng mãi.","must_any":["đừng","chớ"],"negation":True},
 {"id":"style_02","source":"你若真想救她，就别再浪费时间了。","reference":"Nếu thật sự muốn cứu nàng thì đừng lãng phí thời gian nữa.","must_any":["cứu","đừng"],"negation":True},
 {"id":"style_03","source":"就算只有一成机会，我也要试。","reference":"Dù chỉ có một thành cơ hội, ta cũng phải thử.","must_any":["một thành","10%","một phần mười"]},
 {"id":"ctx_01","context":"长老在责备一个年轻弟子。弟子刚刚违反门规。","source":"你还知道回来？","reference":"Ngươi còn biết đường về à?","must_any":["về"],"question":True},
 {"id":"ctx_02","context":"两人正在讨论一名女子。说话者强调她不是恋人。","source":"她只是我的师妹，不是你想的那样。","reference":"Nàng chỉ là sư muội của ta, không phải như ngươi nghĩ.","must_all":["sư muội"],"negation":True},
 {"id":"ctx_03","context":"青云宗是宗门名称，必须固定译名为 Thanh Vân Tông。","source":"三年后，我们在青云宗再见。","reference":"Ba năm sau, chúng ta gặp lại ở Thanh Vân Tông.","must_all":["thanh vân tông"],"must_any":["ba năm","3 năm"]},
 {"id":"ctx_04","context":"说话者是宗门高层，自称本座，对晚辈说话。","source":"本座给你最后一次机会。","reference":"Bổn tọa cho ngươi cơ hội cuối cùng.","must_any":["bổn tọa","bản tọa"],"forbid":["tôi"]},
 {"id":"ctx_05","context":"师尊正在严肃警告弟子。","source":"筑基之前，不可强行开辟丹田。","reference":"Trước khi Trúc Cơ, không được cưỡng ép khai mở đan điền.","must_all":["trúc cơ","đan điền"],"must_any":["không","chớ"],"negation":True},
 {"id":"idiom_01","source":"你这是自寻死路。","reference":"Ngươi đang tự tìm đường chết.","must_any":["tìm đường chết","tự sát","tự chuốc"]},
 {"id":"idiom_02","source":"别给脸不要脸。","reference":"Đừng được đằng chân lân đằng đầu.","must_any":["đừng"],"negation":True},
 {"id":"dialogue_01","context":"A刚刚拒绝帮忙，B很不满。","source":"行，你不帮就算了。","reference":"Được, ngươi không giúp thì thôi.","must_any":["không giúp","chẳng giúp"],"negation":True},
 {"id":"dialogue_02","context":"说话者刚从昏迷中醒来，不知道发生了什么。","source":"这里是什么地方？","reference":"Đây là đâu?","must_any":["đâu"],"question":True},
]

GLOSSARY={
 "修仙":"tu tiên","修士":"tu sĩ","灵气":"linh khí","灵力":"linh lực","灵根":"linh căn",
 "炼气":"Luyện Khí","筑基":"Trúc Cơ","金丹":"Kim Đan","元婴":"Nguyên Anh","渡劫":"Độ Kiếp",
 "宗门":"tông môn","师尊":"sư tôn","师兄":"sư huynh","师姐":"sư tỷ","师弟":"sư đệ","师妹":"sư muội",
 "掌门":"chưởng môn","长老":"trưởng lão","道友":"đạo hữu","法宝":"pháp bảo","丹药":"đan dược",
 "功法":"công pháp","秘境":"bí cảnh","洞府":"động phủ","魔修":"ma tu","正道":"chính đạo",
 "天道":"thiên đạo","飞升":"phi thăng","境界":"cảnh giới","丹田":"đan điền","灵石":"linh thạch",
 "道侣":"đạo lữ","青云宗":"Thanh Vân Tông","本座":"bổn tọa",
}
META=("không thể thực hiện yêu cầu","không thể đáp ứng yêu cầu","vi phạm bản quyền","chính sách nội dung","đội phim","đoàn phim","quay phim")
CJK_RE=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
NEG_VI=("không","chẳng","chưa","đừng","chớ","khỏi")
Q_VI=("không","à","ư","sao","gì","nào","chứ","đâu")

def post(path,payload):
    data=json.dumps(payload,ensure_ascii=False).encode()
    req=Request(BENCH["callback_base"].rstrip("/")+path,data=data,headers={
        "content-type":"application/json","x-benchmark-token":BENCH["run_token"]
    },method="POST")
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def beat(stage,message):
    try: post("/heartbeat",{"run_id":BENCH["run_id"],"stage":stage,"message":message})
    except Exception as exc: print("PRO_BENCH_HEARTBEAT_ERROR",repr(exc),flush=True)

def clean(s):
    return re.sub(r"\s+"," ",str(s)).strip()

def hard_validate(case,out):
    v=clean(out).casefold()
    words=set(re.findall(r"[A-Za-zÀ-ỹ]+",v))
    reasons=[]
    if not v: reasons.append("empty")
    if CJK_RE.search(v): reasons.append("contains_cjk")
    if any(x in v for x in META): reasons.append("meta_hallucination")
    for x in case.get("must_all",[]):
        if x.casefold() not in v: reasons.append("missing:"+x)
    any_terms=[x.casefold() for x in case.get("must_any",[])]
    if any_terms and not any(x in v for x in any_terms): reasons.append("missing_any")
    for x in case.get("forbid",[]):
        if x.casefold() in v: reasons.append("forbidden:"+x)
    if case.get("negation") and not any(x in words for x in NEG_VI):
        reasons.append("lost_negation")
    if case.get("question") and "?" not in out and not any(x in words for x in Q_VI):
        reasons.append("lost_question")
    return reasons

def prompt(case):
    src=case["source"]
    refs=[f"{zh} 翻译成 {vi}" for zh,vi in GLOSSARY.items() if zh in src]
    term=("参考下面的固定术语翻译：\n"+"\n".join(refs)+"\n\n") if refs else ""
    return (
      term+
      "〖背景信息〗\n"+str(case.get("context") or "")+"\n"
      "〖翻译要求〗\n"
      "1. 忠实传达原意，不得添加原文没有的信息，不得遗漏关键含义。\n"
      "2. 先保证准确，再保证越南语自然；不得改变人物关系、否定、数字、疑问或情绪强度。\n"
      "3. 人物称呼、专有名词、修仙境界、功法和术语必须前后一致。\n"
      "4. 使用自然、专业、适合影视对白的越南语；避免生硬逐字翻译。\n"
      "5. 只输出译文，不要解释、注释、免责声明或元话语。\n"
      "〖待翻译文本〗\n"+src
    )

def parse_review(text):
    text=clean(text)
    m=re.search(r"\{.*\}",text)
    if not m: return {"severity":"MAJOR","reason":"review_json_parse_failed"}
    try:
        obj=json.loads(m.group(0))
    except Exception:
        return {"severity":"MAJOR","reason":"review_json_parse_failed"}
    sev=str(obj.get("severity") or "MAJOR").upper()
    if sev not in {"PASS","MINOR","MAJOR","CRITICAL"}: sev="MAJOR"
    return {"severity":sev,"reason":clean(obj.get("reason") or ""),"correction":clean(obj.get("correction") or "")}

def main():
    beat("installing","Installing professional translation benchmark runtime")
    subprocess.run([sys.executable,"-m","pip","install","--quiet",
        "transformers>=5.6,<6","accelerate<2","bitsandbytes>=0.45,<1","sentencepiece"],check=True)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)

    translator="tencent/Hy-MT2-7B"
    beat("loading_translator",f"Loading {translator}")
    tok=AutoTokenizer.from_pretrained(translator,trust_remote_code=True)
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(translator,quantization_config=qcfg,device_map="auto",low_cpu_mem_usage=True,trust_remote_code=True)
    model.eval()
    rows=[]
    beat("translating",f"Translating {len(CASES)} professional cases")
    with torch.inference_mode():
        for off in range(0,len(CASES),4):
            batch=CASES[off:off+4]
            chats=[tok.apply_chat_template([{"role":"user","content":prompt(x)}],tokenize=False,add_generation_prompt=True) for x in batch]
            inp=tok(chats,return_tensors="pt",padding=True,truncation=True,max_length=640)
            inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
            out=model.generate(**inp,max_new_tokens=96,do_sample=False,repetition_penalty=1.05,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            gen=out[:,inp["input_ids"].shape[1]:]
            texts=tok.batch_decode(gen,skip_special_tokens=True)
            for case,target in zip(batch,texts,strict=True):
                target=clean(target)
                rows.append({"case":case,"target":target,"hard_reasons":hard_validate(case,target)})
    del model,tok
    gc.collect(); torch.cuda.empty_cache()

    reviewer="Qwen/Qwen2.5-7B-Instruct"
    beat("loading_reviewer",f"Loading independent reviewer {reviewer}")
    rtok=AutoTokenizer.from_pretrained(reviewer)
    rtok.padding_side="left"
    if rtok.pad_token_id is None: rtok.pad_token=rtok.eos_token
    rmodel=AutoModelForCausalLM.from_pretrained(reviewer,quantization_config=qcfg,device_map="auto",low_cpu_mem_usage=True)
    rmodel.eval()
    beat("reviewing",f"Independent MQM-style review of {len(rows)} translations")
    with torch.inference_mode():
        for row in rows:
            case=row["case"]
            review_prompt=(
              "Bạn là trưởng nhóm biên dịch Trung-Việt. Hãy kiểm định theo kiểu MQM, ưu tiên độ trung thành nghĩa. "
              "CRITICAL: sai nghĩa làm đảo thông tin, hallucination, mất phủ định/số liệu quan trọng. "
              "MAJOR: bỏ/thêm ý đáng kể, sai thuật ngữ/xưng hô/ngữ cảnh. "
              "MINOR: câu hơi cứng hoặc lựa chọn từ chưa tối ưu nhưng nghĩa vẫn đúng. "
              "PASS: đúng nghĩa, tự nhiên, phù hợp ngữ cảnh. "
              "Không phạt chỉ vì khác câu tham khảo nếu nghĩa tương đương. "
              "Chỉ trả JSON một dòng: {\"severity\":\"PASS|MINOR|MAJOR|CRITICAL\",\"reason\":\"...\",\"correction\":\"...\"}.\n"
              f"NGỮ CẢNH: {case.get('context','')}\n"
              f"NGUỒN TRUNG: {case['source']}\n"
              f"BẢN DỊCH: {row['target']}\n"
              f"THAM KHẢO: {case['reference']}"
            )
            chat=rtok.apply_chat_template([{"role":"user","content":review_prompt}],tokenize=False,add_generation_prompt=True)
            inp=rtok([chat],return_tensors="pt",truncation=True,max_length=900)
            inp={k:v.to("cuda") for k,v in inp.items() if k!="token_type_ids"}
            out=rmodel.generate(**inp,max_new_tokens=160,do_sample=False,repetition_penalty=1.03,pad_token_id=rtok.pad_token_id,eos_token_id=rtok.eos_token_id)
            gen=out[:,inp["input_ids"].shape[1]:]
            review=rtok.batch_decode(gen,skip_special_tokens=True)[0]
            row["review"]=parse_review(review)
            print("PRO_CASE",json.dumps({"id":case["id"],"target":row["target"],"hard":row["hard_reasons"],"review":row["review"]},ensure_ascii=False),flush=True)

    hard_failed=[r for r in rows if r["hard_reasons"]]
    critical=[r for r in rows if r["review"]["severity"]=="CRITICAL"]
    major=[r for r in rows if r["review"]["severity"]=="MAJOR"]
    minor=[r for r in rows if r["review"]["severity"]=="MINOR"]
    total=len(rows)
    passed=total-len(hard_failed)-len(critical)-len(major)
    professional_ok=(not hard_failed and not critical and not major and len(minor)<=max(2,math.floor(total*0.10)))
    report={
      "ok":professional_ok,
      "profile":"professional-zh-vi-v1",
      "translator":translator,
      "reviewer":reviewer,
      "gpu":torch.cuda.get_device_name(0),
      "cases":total,
      "hard_failed":len(hard_failed),
      "critical":len(critical),
      "major":len(major),
      "minor":len(minor),
      "professional_passed":passed,
      "acceptance":{"hard_failed":0,"critical":0,"major":0,"minor_max":max(2,math.floor(total*0.10))},
      "rows":[{"id":r["case"]["id"],"source":r["case"]["source"],"reference":r["case"]["reference"],"target":r["target"],"hard_reasons":r["hard_reasons"],"review":r["review"]} for r in rows],
    }
    Path("/kaggle/working/professional-translation-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not professional_ok:
        summary=" | ".join(
            f"{r['case']['id']}:{','.join(r['hard_reasons']) or r['review']['severity']}:{r['review'].get('reason','')}"
            for r in rows if r["hard_reasons"] or r["review"]["severity"] in {"CRITICAL","MAJOR"}
        )
        raise RuntimeError("professional translation benchmark failed: "+summary[:1400])
    post("/complete",{"run_id":BENCH["run_id"],"report":report,"metadata":{
        "profile":report["profile"],"cases":total,"hard_failed":0,"critical":0,"major":0,"minor":len(minor)
    }})
    print("PROFESSIONAL_TRANSLATION_VERIFIED",json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("PROFESSIONAL_TRANSLATION_FAIL",repr(exc),flush=True)
        try: post("/fail",{"run_id":BENCH["run_id"],"error":repr(exc)})
        except Exception as fail_exc: print("PRO_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise

from __future__ import annotations
import http.cookiejar
import json
import os
from pathlib import Path
from urllib.request import Request, build_opener, HTTPCookieProcessor
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
submitted=json.loads((OUT/"submitted.json").read_text(encoding="utf-8"))
actual_url=submitted.get("response",{}).get("url") or submitted["url"]
owner=submitted.get("owner","duykhanhta")
slug=actual_url.rstrip("/").split("/")[-1]

jar=http.cookiejar.CookieJar()
opener=build_opener(HTTPCookieProcessor(jar))
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"}
if token:
    headers["Authorization"]="Bearer "+token
try:
    opener.open(Request(actual_url,headers=headers),timeout=60).read(1000)
except Exception:
    opener.open(Request("https://www.kaggle.com/",headers=headers),timeout=60).read(1000)

xsrf=None
cookies={}
for c in jar:
    cookies[c.name]=c.value
    if c.name in ("XSRF-TOKEN","CSRF-TOKEN"):
        xsrf=c.value

def post(method, body):
    h={"User-Agent":headers["User-Agent"],"Content-Type":"application/json","Accept":"application/json"}
    if token:
        h["Authorization"]="Bearer "+token
    if xsrf:
        h["X-XSRF-TOKEN"]=xsrf
    req=Request("https://www.kaggle.com/api/i/"+method,data=json.dumps(body).encode(),headers=h,method="POST")
    try:
        with opener.open(req,timeout=120) as r:
            raw = r.read()
            text = raw.decode("utf-8", errors="replace")
            ctype = r.headers.get("Content-Type", "")
            try:
                body = json.loads(text)
            except Exception:
                body = {"raw_text": text[:20000], "content_type": ctype, "bytes": len(raw)}
            return {"status":r.status,"body":body}
    except HTTPError as exc:
        text=exc.read().decode("utf-8",errors="replace")
        try: b=json.loads(text)
        except Exception: b=text
        return {"status":exc.code,"body":b}

vm=post("kernels.LegacyKernelsService/GetKernelViewModel",{
    "authorUserName":owner,
    "kernelSlug":slug,
    "tab":"output",
})
result={"url":actual_url,"owner":owner,"slug":slug,"xsrf":bool(xsrf),"cookie_names":sorted(cookies),"view_model":vm}

kernel=None
if vm["status"]==200 and isinstance(vm["body"],dict):
    kernel=vm["body"].get("kernel") or {}
    kid=kernel.get("id")
    if kid:
        versions=post("kernels.KernelsService/ListKernelVersions",{
            "kernelId":kid,
            "sortOption":"VERSION_ID",
            "pageSize":30,
        })
        result["versions"]=versions

(OUT/"internal-view-model.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))

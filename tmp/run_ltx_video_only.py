from __future__ import annotations
import json, os, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse, quote
from urllib.error import HTTPError, URLError

BASE="https://www.kaggle.com/api/v1"
USER=os.environ.get("KAGGLE_USERNAME","duykhanhta").strip()
TOKEN=os.environ.get("KAGGLE_API_TOKEN","").strip()
if not TOKEN: raise SystemExit("missing token")
OUT=Path("video-smoke-results"); OUT.mkdir(exist_ok=True)
KERNEL="from __future__ import annotations\nimport gc, json, os, shutil, subprocess, sys, time, traceback\nfrom hashlib import sha256\nfrom pathlib import Path\n\nWORK=Path(\"/kaggle/working\")\nREPORT=WORK/\"ltx_video_only_report.json\"\nOUTPUT=WORK/\"ltx_video_only.mp4\"\nLTX=WORK/\"LTX-Video\"\n\ndef save(data):\n    REPORT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+\"\\n\",encoding=\"utf-8\")\n\nreport={\"status\":\"running\",\"model\":\"Lightricks/LTX-Video ltxv-2b-0.9.8-distilled\"}\nstarted=time.perf_counter()\ntry:\n    os.environ.setdefault(\"HF_HUB_DISABLE_TELEMETRY\",\"1\")\n    os.environ.setdefault(\"PYTORCH_CUDA_ALLOC_CONF\",\"expandable_segments:True\")\n\n    import torch\n    if not torch.cuda.is_available():\n        raise RuntimeError(\"CUDA GPU is not available\")\n    report[\"gpu\"]=torch.cuda.get_device_name(0)\n    report[\"gpu_count\"]=torch.cuda.device_count()\n    report[\"vram_gb\"]=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)\n    save(report)\n\n    if LTX.exists():\n        shutil.rmtree(LTX)\n    subprocess.check_call([\n        \"git\",\"clone\",\"--quiet\",\"--filter=blob:none\",\n        \"https://github.com/Lightricks/LTX-Video.git\",str(LTX)\n    ])\n    subprocess.check_call([\n        \"git\",\"-C\",str(LTX),\"checkout\",\"--quiet\",\n        \"4b2d053057623ddd4d0a1d3e9cd28890e9ef487f\"\n    ])\n    t0=time.perf_counter()\n    subprocess.check_call([\n        sys.executable,\"-m\",\"pip\",\"install\",\"--quiet\",\n        \"-e\",str(LTX)+\"[inference]\"\n    ])\n    report[\"install_seconds\"]=round(time.perf_counter()-t0,3)\n    save(report)\n\n    sys.path.insert(0,str(LTX))\n    from ltx_video.inference import infer, InferenceConfig\n\n    out_dir=WORK/\"ltx-output\"\n    out_dir.mkdir(parents=True,exist_ok=True)\n    prompt=(\n        \"A beautiful young Vietnamese woman at an outdoor riverside cafe in soft morning light. \"\n        \"She looks toward the camera, blinks naturally, lightly turns her head and smiles. \"\n        \"Hair moves gently in the breeze. Stable face and body proportions, realistic motion, \"\n        \"natural camera, full head visible, no text, no watermark.\"\n    )\n    t0=time.perf_counter()\n    infer(InferenceConfig(\n        pipeline_config=str(LTX/\"configs\"/\"ltxv-2b-0.9.8-distilled.yaml\"),\n        prompt=prompt,\n        height=512,\n        width=768,\n        num_frames=49,\n        frame_rate=24,\n        seed=20261007,\n        output_path=str(out_dir),\n        offload_to_cpu=True,\n        negative_prompt=\"worst quality, blurry, jittery, distorted, deformed anatomy, cropped head, text, watermark\",\n    ))\n    infer_seconds=time.perf_counter()-t0\n\n    candidates=list(out_dir.rglob(\"*.mp4\"))\n    if not candidates:\n        raise FileNotFoundError(\"LTX-Video produced no MP4\")\n    generated=max(candidates,key=lambda p:p.stat().st_mtime)\n    shutil.copy2(generated,OUTPUT)\n    raw=OUTPUT.read_bytes()\n    if len(raw)<20000 or raw[4:8]!=b\"ftyp\":\n        raise RuntimeError(\"invalid MP4 output\")\n\n    report.update({\n        \"status\":\"success\",\n        \"inference_seconds\":round(infer_seconds,3),\n        \"total_seconds\":round(time.perf_counter()-started,3),\n        \"frames\":49,\n        \"fps\":24,\n        \"approx_seconds\":round(49/24,3),\n        \"bytes\":len(raw),\n        \"sha256\":sha256(raw).hexdigest(),\n    })\n    save(report)\nexcept Exception as exc:\n    report.update({\n        \"status\":\"failed\",\n        \"error\":f\"{type(exc).__name__}: {exc}\",\n        \"traceback\":traceback.format_exc()[-16000:],\n        \"total_seconds\":round(time.perf_counter()-started,3),\n    })\n    save(report)\n    raise\n"

def api(method,path,payload=None,timeout=180):
    body=json.dumps(payload).encode() if payload is not None else None
    req=Request(BASE+path,data=body,method=method,headers={
        "Authorization":f"Bearer {TOKEN}","Content-Type":"application/json","Accept":"application/json"
    })
    with urlopen(req,timeout=timeout) as r:
        return json.loads(r.read().decode())

stamp=str(int(time.time()))
slug=f"ltx-video-only-{stamp}"
payload={
    "slug":f"{USER}/{slug}",
    "newTitle":f"LTX Video Only {stamp[-8:]}",
    "text":KERNEL,
    "language":"python",
    "kernelType":"script",
    "isPrivate":False,
    "enableGpu":True,
    "enableTpu":False,
    "enableInternet":True,
    "machineShape":"NvidiaTeslaT4",
}
res=api("POST","/kernels/push",payload)
if res.get("error"): raise RuntimeError(str(res["error"]))
url=res.get("url") or f"https://www.kaggle.com/code/{USER}/{slug}"
parts=[p for p in urlparse(url).path.split("/") if p]
owner,actual_slug=parts[-2],parts[-1]
print("SUBMITTED",url,flush=True)

headers={"Authorization":f"Bearer {TOKEN}","Accept":"*/*"}
def download(name,timeout=60):
    u=f"{BASE}/kernels/output/download/{owner}/{actual_slug}/{quote(name,safe='/')}?version_number=1"
    req=Request(u,headers=headers)
    with urlopen(req,timeout=timeout) as r: return r.read()

deadline=time.time()+5400
report=None
while time.time()<deadline:
    try:
        raw=download("ltx_video_only_report.json",45)
        if raw:
            report=json.loads(raw.decode())
            print("REPORT",json.dumps(report,ensure_ascii=False),flush=True)
            if report.get("status") in {"success","failed"}:
                (OUT/"ltx_video_only_report.json").write_bytes(raw)
                break
    except Exception:
        pass
    time.sleep(30)

if report is None:
    raise TimeoutError("No final report")

if report.get("status")=="success":
    data=download("ltx_video_only.mp4",300)
    (OUT/"ltx_video_only.mp4").write_bytes(data)
    print("VIDEO_BYTES",len(data),flush=True)

if report.get("status")=="failed":
    raise RuntimeError(report.get("error","video failed"))

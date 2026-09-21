#!/usr/bin/env python3
from __future__ import annotations

# __JOB_CONFIG_INJECT__

import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path('/kaggle/working')
AUDIO=WORK/'source-audio.mp3'
VOICE_WAV=WORK/'vietnamese-voice.wav'
VOICE_MP3=WORK/'voice.mp3'
SRT=WORK/'vi.srt'
META=WORK/'metadata.json'
SEGDIR=WORK/'tts'
SEGDIR.mkdir(parents=True,exist_ok=True)

API=JOB['callback_base'].rstrip('/')
JOB_ID=JOB['job_id']
JOB_TOKEN=JOB['job_token']
_stage={'name':'starting','message':'GPU worker starting'}
_stop=threading.Event()

CJK_RE=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
REPEAT_RE=re.compile(r'\b([\wÀ-ỹ]+)(?:\s+\1)+\b',re.IGNORECASE)
FANTASY_GLOSSARY={
    '修仙':'tu tiên','修士':'tu sĩ','灵气':'linh khí','靈氣':'linh khí','灵力':'linh lực','靈力':'linh lực',
    '灵根':'linh căn','靈根':'linh căn','炼气':'Luyện Khí','練氣':'Luyện Khí','筑基':'Trúc Cơ','築基':'Trúc Cơ',
    '金丹':'Kim Đan','元婴':'Nguyên Anh','元嬰':'Nguyên Anh','渡劫':'Độ Kiếp','宗门':'tông môn','宗門':'tông môn',
    '师尊':'sư tôn','師尊':'sư tôn','师兄':'sư huynh','師兄':'sư huynh','师姐':'sư tỷ','師姐':'sư tỷ',
    '师弟':'sư đệ','師弟':'sư đệ','师妹':'sư muội','師妹':'sư muội','掌门':'chưởng môn','掌門':'chưởng môn',
    '长老':'trưởng lão','長老':'trưởng lão','道友':'đạo hữu','法宝':'pháp bảo','法寶':'pháp bảo',
    '丹药':'đan dược','丹藥':'đan dược','功法':'công pháp','秘境':'bí cảnh','洞府':'động phủ',
    '魔修':'ma tu','正道':'chính đạo','天道':'thiên đạo','飞升':'phi thăng','飛升':'phi thăng','境界':'cảnh giới',
}

# Style profile extracted from the user's reference clip: short dialogue bursts,
# clear pauses, moderate pace, slightly bright tone. This is a style match,
# not speaker-identity cloning.
STYLE={
    'max_tempo':1.22,
    'min_tempo':0.90,
    'pitch_ratio':1.025,
    'max_extra_gap':0.24,
    'words_per_second':3.25,
}

def post(path:str,payload:dict,*,token:bool=True)->dict:
    data=json.dumps(payload,ensure_ascii=False).encode()
    headers={'content-type':'application/json'}
    if token:
        headers['x-job-token']=JOB_TOKEN
    req=Request(API+path,data=data,headers=headers,method='POST')
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def heartbeat(stage:str,message:str):
    _stage['name']=stage; _stage['message']=message
    try:
        post('/heartbeat',{'job_id':JOB_ID,'stage':stage,'message':message})
        print('HB_HEARTBEAT',stage,message,flush=True)
    except Exception as exc:
        print('HB_HEARTBEAT_ERROR',stage,repr(exc),flush=True)

def heartbeat_loop():
    while not _stop.wait(60):
        heartbeat(_stage['name'],_stage['message'])

def download(url:str,path:Path):
    req=Request(url,headers={'User-Agent':'Hidden-Beyond-AI/3.0'})
    with urlopen(req,timeout=180) as src,path.open('wb') as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk: break
            dst.write(chunk)

def upload(url:str,path:Path,mime:str):
    with path.open('rb') as f:
        data=f.read()
    req=Request(url,data=data,headers={'content-type':mime,'x-upsert':'true'},method='PUT')
    with urlopen(req,timeout=300) as r:
        if r.status not in (200,201):
            raise RuntimeError(f'upload failed {r.status}')

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def clean(s:str)->str:
    value=re.sub(r'\s+',' ',str(s)).strip().strip('“”"')
    previous=None
    while previous!=value:
        previous=value
        value=REPEAT_RE.sub(r'\1',value)
    value=re.sub(r'\s+([,.;:!?])',r'\1',value)
    return value.strip()

def wav_duration(path:Path)->float:
    with wave.open(str(path),'rb') as w:
        return w.getnframes()/w.getframerate()

def ts(sec:float)->str:
    ms=max(0,int(round(sec*1000)))
    h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def json_object(raw:str)->dict:
    text=raw.strip()
    text=re.sub(r'^\`\`\`(?:json)?\s*','',text,flags=re.I)
    text=re.sub(r'\s*\`\`\`$','',text)
    start=text.find('{')
    if start<0: raise ValueError('no JSON object in model response')
    # Qwen occasionally emits literal newlines/control chars inside JSON strings.
    # strict=False accepts those without discarding the whole episode.
    obj,_=json.JSONDecoder(strict=False).raw_decode(text[start:])
    if not isinstance(obj,dict): raise ValueError('model response is not an object')
    return obj

def glossary_hint(text:str)->str:
    terms=[]
    for zh,vi in FANTASY_GLOSSARY.items():
        if zh in text and vi not in terms:
            terms.append(vi)
    return ', '.join(terms[:10])

def validate_vi(text:str,source:str,*,field:str)->str:
    value=clean(text)
    if not value: raise ValueError(f'{field} empty')
    if CJK_RE.search(value): raise ValueError(f'{field} still contains CJK: {value}')
    if REPEAT_RE.search(value): raise ValueError(f'{field} has repeated words: {value}')
    for number in re.findall(r'\d+',source):
        if number not in value:
            raise ValueError(f'{field} lost number {number}: {value}')
    return value

def main():
    heartbeat('installing','Installing local AI runtime')
    run([sys.executable,'-m','pip','install','--quiet',
         'faster-whisper>=1.1,<2','transformers<5','accelerate<2',
         'sentencepiece','sacremoses'])

    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    heartbeat('fetching_input','Fetching compressed source audio')
    cfg=post('/worker-config',{'job_id':JOB_ID})
    download(cfg['input_audio_url'],AUDIO)
    if AUDIO.stat().st_size<100000:
        raise RuntimeError('input audio is unexpectedly small')

    device='cuda' if torch.cuda.is_available() else 'cpu'
    compute='float16' if device=='cuda' else 'int8'
    whisper_name='large-v3-turbo' if device=='cuda' else 'small'
    beam_size=3 if device=='cuda' else 3
    heartbeat('transcribing',f'Speech recognition on {device} with {whisper_name}')
    whisper=WhisperModel(whisper_name,device=device,compute_type=compute)
    seg_iter,info=whisper.transcribe(
        str(AUDIO),language='zh',vad_filter=True,beam_size=beam_size,
        condition_on_previous_text=True,word_timestamps=True,
        vad_parameters={'min_silence_duration_ms':220,'speech_pad_ms':80},
    )
    raw=[{'start':float(s.start),'end':float(s.end),'text':clean(s.text)} for s in seg_iter if clean(s.text)]
    if len(raw)<10: raise RuntimeError(f'too few transcript segments: {len(raw)}')

    # Preserve scene/dialogue pauses. Merge only fragments that are almost contiguous.
    merged=[]
    for s in raw:
        if not merged:
            merged.append(dict(s)); continue
        cur=merged[-1]
        candidate=(cur['text']+' '+s['text']).strip()
        if s['start']-cur['end']<=0.12 and s['end']-cur['start']<=4.5 and len(candidate)<=52:
            cur['end']=s['end']; cur['text']=candidate
        else:
            merged.append(dict(s))
    segments=merged
    for i,s in enumerate(segments):
        s['index']=i+1
        s['slot']=max(0.30,float(s['end'])-float(s['start']))
        s['max_words']=max(2,int(math.floor(s['slot']*STYLE['words_per_second']+0.5)))
    heartbeat('transcribed',f'Transcript ready: {len(segments)} timed dialogue segments')
    del whisper; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    heartbeat('translating',f'Contextual cultivation translation for {len(segments)} segments')
    translated_title=''
    translation_model=''

    if device=='cuda':
        translation_model='Qwen/Qwen2.5-3B-Instruct'
        tok=AutoTokenizer.from_pretrained(translation_model)
        model=AutoModelForCausalLM.from_pretrained(
            translation_model,torch_dtype=torch.float16,low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        system=(
            'Bạn là biên tập viên lồng tiếng Việt cho phim hoạt hình tiên hiệp Trung Quốc. '
            'Dịch đúng nghĩa theo ngữ cảnh, dùng tiếng Việt tự nhiên để đọc thành lời, không dịch máy từng chữ. '
            'Ưu tiên thuật ngữ Hán-Việt quen thuộc của thể loại tu tiên/tiên hiệp; giữ nhất quán tên người, môn phái, cảnh giới và xưng hô. '
            'Không tự thêm nội dung, không lặp từ, không để lại chữ Hán. Câu phải ngắn đủ để đọc trong thời lượng được cấp. '
            'Chỉ trả về JSON đúng schema được yêu cầu, không markdown, không giải thích.'
        )

        def qwen(prompt:str,max_new:int=900)->str:
            messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=tok([text],return_tensors='pt').to(device)
            with torch.inference_mode():
                out=model.generate(
                    **inputs,max_new_tokens=max_new,do_sample=False,
                    repetition_penalty=1.08,no_repeat_ngram_size=3,
                    eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,
                )
            generated=out[0][inputs.input_ids.shape[1]:]
            return tok.decode(generated,skip_special_tokens=True)

        def parse_tagged_lines(raw:str)->dict[int,str]:
            parsed={}
            for line in str(raw).splitlines():
                m=re.match(r'^\\s*(?:[-*]\\s*)?\\[(\\d+)\\]\\s*(.+?)\\s*$',line)
                if not m:
                    continue
                parsed[int(m.group(1))]=clean(m.group(2))
            return parsed

        def qwen_tagged(prompt:str,max_new:int)->dict[int,str]:
            # Tagged plain-text output is substantially more robust than forcing JSON
            # from a small local model; malformed JSON caused the expensive retry storm.
            return parse_tagged_lines(qwen(prompt,max_new=max_new))

        translated={}
        fallback_mt={'tok':None,'model':None}
        translation_started=time.monotonic()
        qwen_attempted=0
        qwen_fallbacks=0
        qwen_budget_seconds=10*60

        def deterministic_mt_fallback(s:dict)->str:
            if fallback_mt['tok'] is None:
                heartbeat('translating_retry','Loading deterministic Chinese→Vietnamese MT fallback')
                fallback_mt['tok']=AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-vi')
                fallback_mt['model']=AutoModelForSeq2SeqLM.from_pretrained(
                    'Helsinki-NLP/opus-mt-zh-vi',low_cpu_mem_usage=True
                ).to('cpu')
                fallback_mt['model'].eval()
            ftok=fallback_mt['tok']; fmodel=fallback_mt['model']
            inp=ftok([s['text']],return_tensors='pt',padding=True,truncation=True,max_length=256)
            with torch.inference_mode():
                out=fmodel.generate(
                    **inp,max_new_tokens=128,num_beams=4,
                    repetition_penalty=1.08,no_repeat_ngram_size=3
                )
            vi=clean(ftok.batch_decode(out,skip_special_tokens=True)[0]).strip('"“” ')
            return validate_vi(vi,s['text'],field=f'deterministic fallback segment {s["index"]}')

        def recover_segment(s:dict)->str:
            nonlocal qwen_fallbacks
            qwen_fallbacks+=1
            plain_prompt=(
                'Dịch đúng một câu tiếng Trung sau sang tiếng Việt tự nhiên để lồng tiếng phim tiên hiệp. '
                'Giữ nguyên nghĩa, phủ định, số liệu, tên riêng và xưng hô; không để chữ Hán. '
                f'Tối đa {s["max_words"]+2} từ. Chỉ trả về câu tiếng Việt, không JSON, không giải thích.\n'
                +s['text']
            )
            try:
                vi=clean(qwen(
                    plain_prompt+'\nBẮT BUỘC chỉ dùng tiếng Việt Latin; tuyệt đối không chép lại tiếng Trung.',
                    max_new=100
                )).strip('"“” ')
                return validate_vi(vi,s['text'],field=f'plain fallback segment {s["index"]}')
            except Exception:
                heartbeat('translating_retry',
                          f'Segment {s["index"]} Qwen fallback failed; using deterministic MT')
                return deterministic_mt_fallback(s)

        def translate_block(block:list[dict]):
            nonlocal qwen_attempted,qwen_fallbacks
            first=block[0]['index']; last=block[-1]['index']
            pos=first-1
            context_before=segments[pos-1]['text'] if pos>0 else ''
            after_pos=pos+len(block)
            context_after=segments[after_pos]['text'] if after_pos<len(segments) else ''
            input_lines=[]
            for s in block:
                gloss=glossary_hint(s['text'])
                hint=f' | glossary={gloss}' if gloss else ''
                input_lines.append(
                    f"[{s['index']}] seconds={s['slot']:.2f} max_words={s['max_words']} | {s['text']}{hint}"
                )
            prompt=(
                'Dịch từng dòng tiếng Trung sang tiếng Việt tự nhiên để lồng tiếng phim tiên hiệp. '
                'BEFORE/AFTER chỉ để hiểu ngữ cảnh, không đưa vào kết quả. Giữ đúng nghĩa, số liệu, tên riêng, '
                'xưng hô và thuật ngữ; không để chữ Hán. Tuân thủ max_words càng sát càng tốt. '
                'Trả về đúng MỘT dòng cho mỗi id theo dạng [id] bản dịch tiếng Việt. '
                'Không JSON, không markdown, không giải thích, không bỏ id.\\n'
                f'BEFORE: {context_before}\\nAFTER: {context_after}\\nINPUT:\\n'
                +'\\n'.join(input_lines)
            )
            qwen_attempted+=len(block)
            try:
                by_id=qwen_tagged(prompt,max_new=max(420,len(block)*48))
            except Exception as exc:
                heartbeat('translating_retry',
                          f'Block {first}-{last} model call failed ({type(exc).__name__}); safe MT fallback')
                for s in block:
                    translated[s['index']]=deterministic_mt_fallback(s)
                    qwen_fallbacks+=1
                return

            recovered=0
            for s in block:
                idx=s['index']
                raw_vi=by_id.get(idx,'')
                try:
                    translated[idx]=validate_vi(raw_vi,s['text'],field=f'segment {idx}')
                except Exception:
                    translated[idx]=recover_segment(s)
                    recovered+=1
            pct=last*100.0/max(1,len(segments))
            heartbeat('translating',
                      f'Translated {last}/{len(segments)} ({pct:.1f}%); recovered={recovered}')

        chunk_size=20
        next_start=0
        for start in range(0,len(segments),chunk_size):
            elapsed=time.monotonic()-translation_started
            fallback_ratio=qwen_fallbacks/max(1,qwen_attempted)
            if elapsed>=qwen_budget_seconds or (qwen_attempted>=60 and fallback_ratio>0.20):
                reason='time_budget' if elapsed>=qwen_budget_seconds else 'high_retry_ratio'
                heartbeat('translating_fast_path',
                          f'Switch remaining to fast MT: {reason}; elapsed={elapsed/60:.1f}m fallback_ratio={fallback_ratio:.1%}')
                next_start=start
                break
            translate_block(segments[start:start+chunk_size])
            next_start=start+chunk_size
        else:
            next_start=len(segments)

        if next_start<len(segments):
            # Release Qwen before the smaller batched translation model.
            del model,tok
            gc.collect()
            torch.cuda.empty_cache()
            translation_model='hybrid-qwen-opus-gpu-budget-v1'
            ftok=AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-vi')
            fmodel=AutoModelForSeq2SeqLM.from_pretrained(
                'Helsinki-NLP/opus-mt-zh-vi',low_cpu_mem_usage=True
            ).to(device)
            fmodel.eval()
            pending=segments[next_start:]
            with torch.inference_mode():
                for off in range(0,len(pending),16):
                    batch=pending[off:off+16]
                    inp=ftok([s['text'] for s in batch],return_tensors='pt',padding=True,truncation=True,max_length=256).to(device)
                    out=fmodel.generate(**inp,max_new_tokens=128,num_beams=2,
                                        repetition_penalty=1.06,no_repeat_ngram_size=3)
                    vis=ftok.batch_decode(out,skip_special_tokens=True)
                    for s,vi in zip(batch,vis,strict=True):
                        translated[s['index']]=validate_vi(vi,s['text'],field=f'fast segment {s["index"]}')
                    done=min(next_start+off+len(batch),len(segments))
                    pct=done*100.0/max(1,len(segments))
                    heartbeat('translating_fast_path',
                              f'Fast MT {done}/{len(segments)} ({pct:.1f}%)')
            del fmodel,ftok
            gc.collect()
            torch.cuda.empty_cache()
            # Reload Qwen only for title and a bounded set of severe timing rewrites.
            tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
            model=AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen2.5-3B-Instruct',torch_dtype=torch.float16,low_cpu_mem_usage=True
            ).to(device)
            model.eval()

        # Keep expensive rewrite calls bounded; first-pass prompts already include max_words.
        severe=[]
        for s in segments:
            vi=translated[s['index']]
            words=len(re.findall(r'[A-Za-zÀ-ỹ0-9]+',vi))
            limit=max(s['max_words']+3,int(s['max_words']*1.45))
            if words>limit:
                severe.append((words/max(1,s['max_words']),s))
        severe.sort(key=lambda x:x[0],reverse=True)
        for _,s in severe[:8]:
            vi=translated[s['index']]
            prompt=(
                'Rút gọn câu tiếng Việt sau để lồng tiếng nhưng phải giữ nguyên ý chính, phủ định, số liệu, tên riêng và xưng hô. '
                f'Tối đa {s["max_words"]} từ. Chỉ trả về câu tiếng Việt, không JSON, không giải thích.\\n'
                f'Nguyên văn Trung: {s["text"]}\\nBản hiện tại: {vi}'
            )
            try:
                shorter=clean(qwen(prompt,max_new=80))
                translated[s['index']]=validate_vi(shorter,s['text'],field=f'short segment {s["index"]}')
            except Exception as exc:
                print('HB_SHORTEN_FALLBACK',s['index'],repr(exc),flush=True)

        title_prompt=(
            'Dịch tiêu đề phim sau sang tiếng Việt tự nhiên theo phong cách tiên hiệp. '
            'Giữ tên riêng, không thêm quảng cáo. Chỉ trả về tiêu đề tiếng Việt, không JSON, không giải thích.\\n'
            +str(cfg.get('title') or '')
        )
        try:
            candidate=clean(qwen(title_prompt,max_new=80))
            if not candidate or CJK_RE.search(candidate):
                raise ValueError('title output invalid')
            translated_title=candidate
        except Exception as exc:
            print('HB_TITLE_FALLBACK',repr(exc),flush=True)
            translated_title=clean(str(cfg.get('series_title') or cfg.get('title') or 'Hidden Beyond'))
        for s in segments: s['vi']=translated[s['index']]
        del model,tok; gc.collect(); torch.cuda.empty_cache()
    else:
        # CPU fallback keeps the worker usable, but GPU/Qwen is the quality path.
        translation_model='Helsinki-NLP/opus-mt-zh-vi'
        tok=AutoTokenizer.from_pretrained(translation_model)
        model=AutoModelForSeq2SeqLM.from_pretrained(translation_model).to('cpu')
        model.eval()
        texts=[s['text'] for s in segments]+[str(cfg.get('title') or '')]
        out_text=[]
        with torch.inference_mode():
            for i in range(0,len(texts),4):
                inp=tok(texts[i:i+4],return_tensors='pt',padding=True,truncation=True,max_length=256)
                out=model.generate(**inp,max_new_tokens=128,num_beams=4,repetition_penalty=1.08,no_repeat_ngram_size=3)
                out_text.extend(tok.batch_decode(out,skip_special_tokens=True))
        for s,vi in zip(segments,out_text[:-1],strict=True): s['vi']=validate_vi(vi,s['text'],field=f'segment {s["index"]}')
        translated_title=clean(out_text[-1])
        del model,tok; gc.collect()

    heartbeat('packaging_translation','Packaging translated timing data for CPU voice render')
    voice_mode='piper-vais1000-sample-style-v2'
    voice_name='vi_VN-vais1000-medium'

    lines=[]
    for i,s in enumerate(segments,1):
        lines += [str(i),f"{ts(s['start'])} --> {ts(s['end'])}",s['vi'],'']
    SRT.write_text('\n'.join(lines),encoding='utf-8')

    translation_seconds=round(time.monotonic()-translation_started,2) if device=='cuda' else None
    meta={
        'ok':True,'job_id':JOB_ID,'source_video_id':cfg['source_video_id'],
        'series_id':cfg['series_id'],'episode_number':cfg['episode_number'],
        'translated_title':translated_title,'segments':len(segments),
        'translation_model':translation_model,'whisper_model':whisper_name,'tts_voice':voice_mode,
        'translation_mode':'contextual_cultivation_dubbing','timing_mode':'speech-segment-sync-no-hard-speedup',
        'style_profile':'reference-inspired-short-bursts-pauses-moderate-bright',
        'style':STYLE,'voice_name':voice_name,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'gpu_efficiency_mode':'qwen-partial-salvage-12min-budget-cpu-tts-v2',
        'gpu_translation_seconds':translation_seconds,
        'qwen_attempted_segments':qwen_attempted if device=='cuda' else 0,
        'qwen_fallback_segments':qwen_fallbacks if device=='cuda' else 0,
        'timed_segments':[{
            'index':s['index'],'start':round(float(s['start']),3),'end':round(float(s['end']),3),
            'vi':s['vi'],'max_words':s['max_words']
        } for s in segments]
    }
    META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    heartbeat('uploading_translation','Uploading subtitles and timing metadata; GPU work is done')
    upload(cfg['subtitle_upload_url'],SRT,'text/plain')
    upload(cfg['metadata_upload_url'],META,'application/json')
    result=post('/ai-complete',{
        'job_id':JOB_ID,'translated_title':translated_title,
        'output_sha256':'','output_bytes':META.stat().st_size,'voice_generated':False
    })
    print('HB_AI_COMPLETE_GPU_RELEASE',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    thread=threading.Thread(target=heartbeat_loop,daemon=True)
    thread.start()
    try:
        main()
    except Exception as exc:
        print('HB_AI_FAIL',repr(exc),flush=True)
        try: post('/fail',{'job_id':JOB_ID,'error':repr(exc)})
        except Exception as fail_exc: print('HB_FAIL_REPORT_ERROR',repr(fail_exc),flush=True)
        raise
    finally:
        _stop.set()

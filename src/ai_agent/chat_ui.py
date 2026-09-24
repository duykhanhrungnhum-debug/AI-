"""Self-contained mobile-first web chat for AI-."""
CHAT_HTML = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>AI- Chat</title>
<style>
:root{--bg:#111214;--panel:#191b1f;--muted:#9ca3af;--line:#2a2d33;--text:#f3f4f6;--bubble:#23262c;--accent:#f3f4f6;--danger:#ef4444}
*{box-sizing:border-box}html,body{height:100%;margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
body{overflow:hidden}.app{height:100dvh;display:grid;grid-template-rows:56px 1fr auto;max-width:980px;margin:0 auto}
header{display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:rgba(17,18,20,.96);backdrop-filter:blur(10px)}
.brand{display:flex;align-items:center;gap:10px;font-weight:700}.dot{width:10px;height:10px;border-radius:50%;background:#22c55e;box-shadow:0 0 12px #22c55e80}
.actions{display:flex;gap:8px}button{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 11px;cursor:pointer}
button:hover{background:#22252a}button:disabled{opacity:.5;cursor:not-allowed}
#messages{overflow-y:auto;padding:22px 14px 34px;scroll-behavior:smooth}.empty{max-width:640px;margin:10vh auto 0;text-align:center;color:var(--muted)}
.empty h1{font-size:28px;color:var(--text);margin-bottom:8px}.msg{max-width:760px;margin:0 auto 18px;display:flex;gap:10px}.msg.user{justify-content:flex-end}
.bubble{max-width:min(86%,720px);white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55;padding:12px 14px;border-radius:16px;background:var(--bubble);border:1px solid var(--line)}
.user .bubble{background:#e5e7eb;color:#111827;border-color:#e5e7eb}.meta{font-size:12px;color:var(--muted);margin-top:7px}.error{color:#fecaca;border-color:#7f1d1d;background:#2b1517}
.composer{padding:10px 12px calc(10px + env(safe-area-inset-bottom));background:linear-gradient(180deg,transparent,var(--bg) 20%)}.composebox{max-width:780px;margin:0 auto;display:flex;align-items:flex-end;gap:8px;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:8px}
textarea{flex:1;resize:none;max-height:180px;min-height:44px;background:transparent;border:0;outline:0;color:var(--text);font:inherit;line-height:1.45;padding:11px 8px}
.send{border-radius:12px;min-width:46px;height:44px;font-weight:700;background:var(--accent);color:#111;border-color:var(--accent)}
.status{max-width:780px;margin:5px auto 0;padding:0 8px;font-size:12px;color:var(--muted);min-height:16px}
dialog{width:min(92vw,420px);border:1px solid var(--line);border-radius:16px;background:var(--panel);color:var(--text);padding:18px}dialog::backdrop{background:#0009}
label{display:block;font-size:13px;color:var(--muted);margin:12px 0 6px}input{width:100%;background:#101114;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:11px;font:inherit}
.row{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}.hint{font-size:13px;color:var(--muted);line-height:1.45}.typing{display:inline-flex;gap:5px;padding:4px 0}.typing i{width:6px;height:6px;border-radius:50%;background:#9ca3af;animation:pulse 1s infinite}.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}@keyframes pulse{0%,80%,100%{opacity:.3}40%{opacity:1}}
.drawerWrap{position:fixed;inset:0;z-index:30;display:none}.drawerWrap.open{display:block}.drawerBackdrop{position:absolute;inset:0;background:#0009}.drawer{position:absolute;left:0;top:0;bottom:0;width:min(86vw,360px);background:#15171a;border-right:1px solid var(--line);padding:14px;overflow-y:auto}.drawerHead{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}.drawerHead h3{margin:0;font-size:18px}.threadList{display:flex;flex-direction:column;gap:7px}.threadItem{display:block;width:100%;text-align:left;padding:11px 12px;border-radius:11px;background:#1b1e22;border:1px solid transparent;color:var(--text)}.threadItem.active{border-color:#4b5563;background:#23262c}.threadTitle{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.threadTime{font-size:11px;color:var(--muted);margin-top:4px}.drawerEmpty{color:var(--muted);padding:12px 4px}
@media(max-width:600px){header{padding:0 10px}.label-hide{display:none}#messages{padding-top:16px}.bubble{max-width:91%}}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="brand"><span class="dot"></span><span>AI- Chat</span></div>
  <div class="actions">
    <button id="historyBtn">☰ <span class="label-hide">Lịch sử</span></button>
    <button id="newChat">＋ <span class="label-hide">Chat mới</span></button>
    <button id="settings">⚙ <span class="label-hide">Kết nối</span></button>
  </div>
</header>
<main id="messages"></main>
<footer class="composer">
  <div class="composebox">
    <textarea id="input" rows="1" placeholder="Nhắn tin cho AI-..." aria-label="Tin nhắn"></textarea>
    <button class="send" id="send" aria-label="Gửi">↑</button>
  </div>
  <div class="status" id="status"></div>
</footer>
</div>

<div class="drawerWrap" id="historyPanel">
  <div class="drawerBackdrop" id="historyBackdrop"></div>
  <aside class="drawer">
    <div class="drawerHead"><h3>Cuộc trò chuyện</h3><button id="closeHistory">✕</button></div>
    <div class="threadList" id="threadList"></div>
  </aside>
</div>

<dialog id="connectDialog">
  <h3 style="margin:0 0 6px">Kết nối AI-</h3>
  <div class="hint">Nhập khóa truy cập của AI-. Khóa chỉ được giữ trong phiên trình duyệt này và không lưu vào lịch sử chat.</div>
  <label for="token">AI_AGENT_API_TOKEN</label>
  <input id="token" type="password" autocomplete="off" placeholder="Nhập khóa truy cập">
  <div class="row">
    <button id="cancelConnect">Đóng</button>
    <button id="saveConnect">Kết nối</button>
  </div>
</dialog>

<script>
const $ = s => document.querySelector(s);
const messagesEl = $('#messages'), input = $('#input'), send = $('#send'), statusEl = $('#status');
const dialog = $('#connectDialog'), tokenInput = $('#token');
const historyPanel = $('#historyPanel'), threadList = $('#threadList');
const LEGACY_STORE = 'ai_chat_messages_v1', THREADS_STORE = 'ai_chat_threads_v1', CURRENT_STORE = 'ai_chat_current_thread_v1', TOKEN = 'ai_chat_token_v1';
let threads = [];
let currentId = '';
let messages = [];
let busy = false;

function newId(){ return Date.now().toString(36)+Math.random().toString(36).slice(2,9); }
function makeThread(msgs=[], title='Chat mới'){
  const now=Date.now();
  return {id:newId(),title,createdAt:now,updatedAt:now,messages:Array.isArray(msgs)?msgs.slice(-80):[]};
}
function currentThread(){ return threads.find(t=>t.id===currentId) || null; }
function syncMessages(){ const t=currentThread(); messages=t?t.messages:[]; }
function deriveTitle(t){
  const first=(t.messages||[]).find(m=>m.role==='user' && String(m.text||'').trim());
  if(!first) return t.title || 'Chat mới';
  const s=String(first.text).trim().replace(/\s+/g,' ');
  return s.length>42?s.slice(0,42)+'…':s;
}
function fmtTime(ts){
  try{return new Date(ts).toLocaleString('vi-VN',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});}catch{return '';}
}

function bootstrapAccess(){
  const params = new URLSearchParams(location.hash.slice(1));
  const access = params.get('access');
  if(access){
    sessionStorage.setItem(TOKEN, access);
  }
}
function load(){
  bootstrapAccess();
  try{ threads=JSON.parse(localStorage.getItem(THREADS_STORE)||'[]'); if(!Array.isArray(threads)) threads=[]; }catch{ threads=[]; }
  if(!threads.length){
    let legacy=[];
    try{ legacy=JSON.parse(localStorage.getItem(LEGACY_STORE)||'[]'); if(!Array.isArray(legacy)) legacy=[]; }catch{ legacy=[]; }
    threads=[makeThread(legacy, legacy.length?'Cuộc trò chuyện cũ':'Chat mới')];
  }
  currentId=localStorage.getItem(CURRENT_STORE)||'';
  if(!threads.some(t=>t.id===currentId)) currentId=threads[0].id;
  syncMessages();
  persist();
  render();
  renderHistory();
  if(!sessionStorage.getItem(TOKEN)) dialog.showModal();
}
function persist(){
  const t=currentThread();
  if(t){
    t.messages=messages.slice(-80);
    t.updatedAt=Date.now();
    if(!t.title || t.title==='Chat mới' || t.title==='Cuộc trò chuyện cũ') t.title=deriveTitle(t);
  }
  const ordered=[...threads].sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0)).slice(0,30);
  threads=ordered;
  localStorage.setItem(THREADS_STORE,JSON.stringify(threads));
  localStorage.setItem(CURRENT_STORE,currentId);
}
function renderHistory(){
  const ordered=[...threads].sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  threadList.innerHTML=ordered.length?ordered.map(t=>
    '<button class="threadItem '+(t.id===currentId?'active':'')+'" data-thread="'+esc(t.id)+'">'+
    '<div class="threadTitle">'+esc(t.title||deriveTitle(t))+'</div>'+
    '<div class="threadTime">'+esc(fmtTime(t.updatedAt||t.createdAt))+'</div></button>'
  ).join(''):'<div class="drawerEmpty">Chưa có cuộc trò chuyện.</div>';
}
function openHistory(){ renderHistory(); historyPanel.classList.add('open'); }
function closeHistory(){ historyPanel.classList.remove('open'); }
function createNewChat(){
  if(busy) return;
  persist();
  const t=makeThread();
  threads.unshift(t);
  currentId=t.id;
  messages=t.messages;
  persist();
  render();
  renderHistory();
  closeHistory();
  input.focus();
}
function switchThread(id){
  if(busy || id===currentId) { closeHistory(); return; }
  persist();
  if(!threads.some(t=>t.id===id)) return;
  currentId=id;
  syncMessages();
  localStorage.setItem(CURRENT_STORE,currentId);
  render();
  renderHistory();
  closeHistory();
  input.focus();
}
function esc(s){ return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
function render(){
  if(!messages.length){
    messagesEl.innerHTML='<div class="empty"><h1>AI-</h1><div>Trò chuyện trực tiếp với AI của dự án.</div></div>'; return;
  }
  messagesEl.innerHTML = messages.map(m =>
    '<div class="msg '+m.role+'"><div class="bubble '+(m.error?'error':'')+'">'+esc(m.text)+
    (m.meta?'<div class="meta">'+esc(m.meta)+'</div>':'')+'</div></div>'
  ).join('');
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function setBusy(v, label='AI- đang trả lời...'){
  busy=v; send.disabled=v; input.disabled=v;
  statusEl.innerHTML=v?'<span class="typing"><i></i><i></i><i></i></span> '+esc(label):'';
}
function sleep(ms){ return new Promise(resolve=>setTimeout(resolve,ms)); }
function statusLabel(data){
  if(data.worker_state==='starting') return 'Đang khởi động Qwen trên Kaggle...';
  if(data.status==='pending') return 'Đang chờ AI- sẵn sàng...';
  if(data.status==='processing') return 'AI- đang xử lý...';
  return 'AI- đang trả lời...';
}
function autoSize(){ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,180)+'px'; }
function buildPrompt(){
  const recent=messages.filter(m=>!m.error).slice(-18);
  return 'Đây là cuộc trò chuyện trực tiếp với người dùng. Hãy trả lời tự nhiên bằng tiếng Việt trừ khi người dùng yêu cầu ngôn ngữ khác. Duy trì ngữ cảnh hội thoại.\n\n' +
    recent.map(m => (m.role==='user'?'Người dùng':'AI')+': '+m.text).join('\n\n') + '\n\nAI:';
}
async function submit(){
  const text=input.value.trim(); if(!text || busy) return;
  const token=sessionStorage.getItem(TOKEN);
  if(!token){ dialog.showModal(); return; }
  messages.push({role:'user',text}); persist(); render(); input.value=''; autoSize();
  setBusy(true,'Đang khởi động Qwen trên Kaggle...');
  try{
    const start=await fetch('/v1/chat',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+token},body:JSON.stringify({prompt:buildPrompt()})});
    const started=await start.json().catch(()=>({}));
    if(!start.ok) throw new Error(start.status===401?'Khóa truy cập không đúng.':(started.detail||started.error||'Không thể khởi động phiên chat.'));
    const jobId=started.job_id;
    if(!jobId) throw new Error('AI- không tạo được phiên chat.');
    let data=null;
    for(let i=0;i<180;i++){
      await sleep(2000);
      const res=await fetch('/v1/chat/status?job_id='+encodeURIComponent(jobId),{headers:{'authorization':'Bearer '+token}});
      data=await res.json().catch(()=>({}));
      if(!res.ok) throw new Error(res.status===401?'Khóa truy cập không đúng.':(data.detail||data.error||'Không đọc được trạng thái AI-.'));
      setBusy(true,statusLabel(data));
      if(data.status==='done') break;
      if(data.status==='error') throw new Error(data.error||data.worker_error||'Phiên AI- bị lỗi.');
    }
    if(!data || data.status!=='done') throw new Error('Qwen khởi động quá lâu. Phiên chat đã dừng để tránh chờ vô hạn.');
    messages.push({role:'assistant',text:data.text||'',meta:[data.provider,data.model].filter(Boolean).join(' · ')});
    persist(); render();
  }catch(e){
    messages.push({role:'assistant',text:'Lỗi: '+e.message,error:true}); persist(); render();
    if(String(e.message).includes('Khóa truy cập')) dialog.showModal();
  }finally{ setBusy(false); input.focus(); }
}
input.addEventListener('input',autoSize);
input.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit();}});
send.addEventListener('click',submit);
$('#settings').addEventListener('click',()=>{tokenInput.value='';dialog.showModal();tokenInput.focus();});
$('#saveConnect').addEventListener('click',()=>{const v=tokenInput.value.trim();if(v){sessionStorage.setItem(TOKEN,v);dialog.close();input.focus();}});
$('#cancelConnect').addEventListener('click',()=>dialog.close());
$('#historyBtn').addEventListener('click',()=>{if(!busy)openHistory();});
$('#closeHistory').addEventListener('click',closeHistory);
$('#historyBackdrop').addEventListener('click',closeHistory);
threadList.addEventListener('click',e=>{const b=e.target.closest('[data-thread]');if(b)switchThread(b.dataset.thread);});
$('#newChat').addEventListener('click',createNewChat);
load(); autoSize();
</script>
</body>
</html>'''

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
@media(max-width:600px){header{padding:0 10px}.label-hide{display:none}#messages{padding-top:16px}.bubble{max-width:91%}}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="brand"><span class="dot"></span><span>AI- Chat</span></div>
  <div class="actions">
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
const STORE = 'ai_chat_messages_v1', TOKEN = 'ai_chat_token_v1';
let messages = [];
let busy = false;

function load(){
  try{ messages = JSON.parse(localStorage.getItem(STORE) || '[]'); if(!Array.isArray(messages)) messages=[]; }catch{ messages=[]; }
  render();
  if(!sessionStorage.getItem(TOKEN)) dialog.showModal();
}
function persist(){ localStorage.setItem(STORE, JSON.stringify(messages.slice(-80))); }
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
function setBusy(v){
  busy=v; send.disabled=v; input.disabled=v;
  statusEl.innerHTML=v?'<span class="typing"><i></i><i></i><i></i></span> AI- đang trả lời...':'';
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
  messages.push({role:'user',text}); persist(); render(); input.value=''; autoSize(); setBusy(true);
  try{
    const res=await fetch('/v1/generate',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+token},body:JSON.stringify({prompt:buildPrompt()})});
    const data=await res.json().catch(()=>({}));
    if(!res.ok) throw new Error(res.status===401?'Khóa truy cập không đúng.':(data.detail||data.error||'Không thể kết nối AI-.'));
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
$('#newChat').addEventListener('click',()=>{messages=[];persist();render();input.focus();});
load(); autoSize();
</script>
</body>
</html>'''

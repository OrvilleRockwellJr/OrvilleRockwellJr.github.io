/* Chats: auto-saved in this browser (IndexedDB, localStorage fallback). No account, nothing leaves the device.
   app.js dispatches "ptp:turn" after each finished run; this file stores it, lists chats, and exports them. */
(function(){
"use strict";
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const DB="ptp-studio", ST="chats", LS="ptp-chats-v1";
let cur=null, list=[], idb=null;

/* ---------- storage ---------- */
function openDB(){ return new Promise(res=>{ try{ const q=indexedDB.open(DB,1); q.onupgradeneeded=()=>q.result.createObjectStore(ST,{keyPath:"id"}); q.onsuccess=()=>res(q.result); q.onerror=()=>res(null); }catch(e){ res(null); } }); }
const lsAll=()=>{ try{ return JSON.parse(localStorage.getItem(LS)||"[]"); }catch(e){ return []; } };
const lsSave=a=>{ try{ localStorage.setItem(LS,JSON.stringify(a)); }catch(e){} };
function dbAll(){ return new Promise(res=>{ if(!idb) return res(lsAll()); const r=idb.transaction(ST).objectStore(ST).getAll(); r.onsuccess=()=>res(r.result||[]); r.onerror=()=>res([]); }); }
function dbPut(c){ return new Promise(res=>{ if(!idb){ const a=lsAll().filter(x=>x.id!==c.id); a.push(c); lsSave(a); return res(); } const t=idb.transaction(ST,"readwrite"); t.objectStore(ST).put(c); t.oncomplete=()=>res(); t.onerror=()=>res(); }); }
function dbDel(id){ return new Promise(res=>{ if(!idb){ lsSave(lsAll().filter(x=>x.id!==id)); return res(); } const t=idb.transaction(ST,"readwrite"); t.objectStore(ST).delete(id); t.oncomplete=()=>res(); t.onerror=()=>res(); }); }

/* ---------- helpers ---------- */
const uid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,6);
const fdate=t=>new Date(t).toLocaleString([], {month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
const num=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):"";
const title=p=>{ const t=String(p||"").replace(/\s+/g," ").trim(); return t?(t.length>48?t.slice(0,46)+"…":t):"Untitled chat"; };
function slim(d){
  const mol=(d.mol||[]).filter(o=>o&&o.valid).map(o=>({smiles:(o.a&&o.a.canon)||o.smiles||"",pass:!!o.pass,critic:o.cp||null,judge:o.jp||null,mw:o.a&&o.a.mw,logp:o.a&&o.a.logp,tpsa:o.a&&o.a.tpsa,ro5:o.a&&o.a.ro5}));
  const dna=(d.dna||[]).map(o=>({guide:o.seq,pass:!!o.pass,onTarget:o.hit?(typeof o.hit==="string"?o.hit:true):false,critic:o.cp,judge:o.jp,usable:o.ck?!!o.ck.usable:null}));
  return {mol,dna,pairs:d.pairs||[],seed:d.seed||null,photo:d.photo?{name:d.photo.name,kind:d.photo.kind,found:d.photo.found}:null};
}
const textOf=h=>{ const t=document.createElement("div"); t.innerHTML=h; t.querySelectorAll("svg,button,script,style").forEach(n=>n.remove()); return t.textContent.replace(/\s+\n/g,"\n").replace(/[ \t]+/g," ").trim(); };
function dl(name,mime,text){ const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([text],{type:mime})); a.download=name; document.body.appendChild(a); a.click(); setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); },500); }
const safe=s=>String(s).toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,"").slice(0,40)||"chat";
const prob=v=>Number.isFinite(v)?num(v,2):"";
function toMarkdown(c){
  let m="# "+c.title+"\n\n_PTPσ Studio · exported "+new Date().toLocaleString()+". All scores are in-silico predictions from small public-data models; none is lab-validated._\n";
  c.turns.forEach((t,i)=>{
    m+="\n## Turn "+(i+1)+" · "+fdate(t.t)+"\n\n**Request:** "+(t.prompt||"(none)")+"\n";
    if(t.kind==="gen"){
      m+="\n**Settings:** "+[t.meta&&t.meta.make,t.meta&&t.meta.n&&(t.meta.n+" each"),t.meta&&t.meta.push&&(t.meta.push+" push")].filter(Boolean).join(", ")+"\n";
      if(t.seed) m+="\n**Seed molecule (analog design):** `"+t.seed.smiles+"` at distance "+t.seed.strength+"\n";
      if(t.photo) m+="\n**Photo:** "+t.photo.name+" ("+t.photo.kind+"). "+(t.photo.found||"")+"\n";
      if(t.pairs&&t.pairs.length){ m+="\n### Two-key pairs (matched experiment: vehicle, molecule, guide knockout, molecule in knockout cells)\n\n| Pair | Molecule (SMILES) | Guide context (30 nt) | PTPσ site | Pair score |\n|---|---|---|---|---|\n"; t.pairs.forEach((q,k)=>{ m+="| "+(k+1)+" | `"+q.smiles+"` | `"+q.guide+"` | "+(q.site||"")+" | "+q.score+" |\n"; }); }
      if(t.mol&&t.mol.length){ m+="\n### Small molecules\n\n| # | SMILES | Passes audit | Critic P(brain / active / tox) | Judge P(brain / active / tox) |\n|---|---|---|---|---|\n"; t.mol.forEach((o,k)=>{ const c3=o.critic?[o.critic.bbb,o.critic.act,o.critic.tox].map(prob).join(" / "):"", j3=o.judge?[o.judge.bbb,o.judge.act,o.judge.tox].map(prob).join(" / "):""; m+="| "+(k+1)+" | `"+o.smiles+"` | "+(o.pass?"yes":"no")+" | "+c3+" | "+j3+" |\n"; }); }
      if(t.dna&&t.dna.length){ m+="\n### Gene-editing guides\n\n| # | Guide context (30 nt) | Passes judge and checks | On target | Critic P | Judge P |\n|---|---|---|---|---|---|\n"; t.dna.forEach((o,k)=>{ m+="| "+(k+1)+" | `"+o.guide+"` | "+(o.pass?"yes":"no")+" | "+(o.onTarget?"yes":"")+" | "+prob(o.critic)+" | "+prob(o.judge)+" |\n"; }); }
    } else { m+="\n"+(t.text||textOf(t.html||""))+"\n"; }
  });
  return m;
}
const toJSON=c=>JSON.stringify({app:"PTPσ Studio",exported:new Date().toISOString(),note:"In-silico predictions only; not lab-validated.",chat:{id:c.id,title:c.title,created:c.created,updated:c.updated,turns:c.turns.map(t=>({time:t.t,kind:t.kind,request:t.prompt,settings:t.meta||null,molecules:t.mol||undefined,guides:t.dna||undefined,twoKeyPairs:t.pairs&&t.pairs.length?t.pairs:undefined,seedMolecule:t.seed||undefined,photo:t.photo||undefined,text:t.kind==="gen"?undefined:(t.text||textOf(t.html||""))}))}},null,2);

/* ---------- UI ---------- */
const p=$("#p-go");
const stopB=$("#g-stop"); if(stopB) new MutationObserver(()=>p.classList.toggle("busy",!stopB.hidden)).observe(stopB,{attributes:true,attributeFilter:["hidden"]});
function setHome(on){ p.classList.toggle("home",on); $("#home").hidden=!on; $("#chathead").hidden=on||!cur; }
function renderList(){
  const ul=$("#chatlist"); list.sort((a,b)=>b.updated-a.updated);
  $("#nochats").hidden=list.length>0;
  ul.innerHTML=list.map(c=>'<li class="'+(cur&&cur.id===c.id?"on":"")+'" data-id="'+c.id+'"><button type="button" class="ci" title="'+esc(c.title)+'">'+esc(c.title)+'<small>'+fdate(c.updated)+' · '+c.turns.length+' run'+(c.turns.length>1?"s":"")+'</small></button><span class="ca"><button type="button" data-a="ren" aria-label="Rename chat" title="Rename">✎</button><button type="button" data-a="md" aria-label="Download chat as Markdown" title="Download .md">↓</button><button type="button" data-a="del" aria-label="Delete chat" title="Delete">✕</button></span></li>').join("");
  $("#ct").textContent=cur?cur.title:"";
}
function turnHTML(t){ return '<article class="turn"><div class="you"><span>You</span>'+esc(t.prompt||"(no text)")+'</div><div class="res">'+t.html+'</div></article>'; }
function drawThread(){ $("#thread").innerHTML=cur?cur.turns.map(turnHTML).join(""):""; }
function newChat(){ cur=null; drawThread(); ["#g-out","#sc-out"].forEach(s=>$(s).innerHTML=""); ["#g-status","#sc-status"].forEach(s=>$(s).textContent=""); const b=$("#g-bar"); if(b) b.style.width="0"; $("#g-prompt").value=""; $("#sc-in").value=""; $("#g-prompt").dispatchEvent(new Event("input")); $("#sc-in").dispatchEvent(new Event("input")); setHome(true); renderList(); document.body.classList.remove("sideopen"); }
function openChat(id){ const c=list.find(x=>x.id===id); if(!c) return; cur=c; ["#g-out","#sc-out"].forEach(s=>$(s).innerHTML=""); drawThread(); setHome(false); renderList(); document.body.classList.remove("sideopen"); const t=$("#thread"); t.scrollIntoView({block:"start"}); }
async function onTurn(e){
  const d=e.detail; if(!d||!d.html) return;
  if(!cur){ cur={id:uid(),title:title(d.prompt),created:Date.now(),updated:Date.now(),turns:[]}; list.push(cur); }
  const s=d.kind==="gen"?slim(d):{};
  const t={t:Date.now(),kind:d.kind,prompt:d.prompt||"",html:d.html,meta:d.kind==="gen"?{make:d.make,n:d.n,push:d.push,use:d.use}:null,mol:s.mol,dna:s.dna,pairs:s.pairs,seed:s.seed,photo:s.photo,text:d.kind==="gen"?undefined:textOf(d.html)};
  cur.turns.push(t); cur.updated=Date.now(); await dbPut(cur);
  $("#thread").insertAdjacentHTML("beforeend",turnHTML(t)); ["#g-out","#sc-out"].forEach(s2=>$(s2).innerHTML="");
  setHome(false); renderList(); const last=$("#thread").lastElementChild; if(last) last.scrollIntoView({block:"start",behavior:"smooth"});
}
document.addEventListener("ptp:turn",onTurn);
["#g-make-go","#sc-go"].forEach(s=>$(s).addEventListener("click",()=>{ if(!$(s).disabled) setHome(false); },true));
$("#newchat").addEventListener("click",newChat);
$("#chatlist").addEventListener("click",async e=>{
  const li=e.target.closest("li"); if(!li) return; const id=li.dataset.id, a=e.target.closest("[data-a]");
  if(!a){ if(e.target.closest(".ci")) openChat(id); return; }
  const c=list.find(x=>x.id===id); if(!c) return;
  if(a.dataset.a==="md") dl(safe(c.title)+".md","text/markdown",toMarkdown(c));
  else if(a.dataset.a==="ren"){ const n=prompt("Rename chat",c.title); if(n&&n.trim()){ c.title=n.trim().slice(0,80); await dbPut(c); renderList(); } }
  else if(a.dataset.a==="del"){ if(confirm("Delete this chat from this browser? This cannot be undone.")){ await dbDel(id); list=list.filter(x=>x.id!==id); if(cur&&cur.id===id) newChat(); else renderList(); } }
});
$("#ct-md").addEventListener("click",()=>cur&&dl(safe(cur.title)+".md","text/markdown",toMarkdown(cur)));
$("#ct-json").addEventListener("click",()=>cur&&dl(safe(cur.title)+".json","application/json",toJSON(cur)));
$("#dl-all").addEventListener("click",()=>{ if(!list.length) return; dl("ptp-studio-chats.json","application/json",JSON.stringify(list.map(c=>JSON.parse(toJSON(c)).chat),null,2)); });
const sb=$("#sidebtn"); if(sb) sb.addEventListener("click",()=>{ const o=document.body.classList.toggle("sideopen"); sb.setAttribute("aria-expanded",String(o)); });
document.addEventListener("keydown",e=>{ if(e.key==="Escape") document.body.classList.remove("sideopen"); });

(async function init(){ idb=await openDB(); list=await dbAll(); setHome(true); renderList(); })();
})();

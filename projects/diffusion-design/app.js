/* Platform UI. Generation runs in worker.js; chemistry (validity, drawing, descriptors, fingerprints) runs here in RDKit.js. */
(function(){
"use strict";
const E=window.GDEngine, $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const sig=E.sigmoid, fmt=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):"–", pct=v=>Number.isFinite(v)?Math.round(v*100)+"%":"–";
const BASE=new URL("./",(document.currentScript&&document.currentScript.src)||(document.querySelector('script[src$="app.js"]')||{}).src||location.href).href;
async function getJSON(rel){ let last=null; for(let k=0;k<3;k++){ const u=BASE+rel+(k?"?r="+Date.now():""); try{ const r=await fetch(u,k?{cache:"reload"}:{}); const t=await r.text(); if(!r.ok||/^\s*</.test(t)) throw new Error("HTTP "+r.status+(/^\s*</.test(t)?" (got a web page instead of data)":"")); return JSON.parse(t); }catch(e){ last=e; await new Promise(r=>setTimeout(r,400*(k+1))); } } throw new Error(rel+": "+(last&&last.message||last)); }
const S={rd:null,worker:null,ready:false,judges:null,ref:null,ptprs:null,job:0,mol:{items:[],cmp:false},dna:{items:[]}};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* ---------- tabs and sliders ---------- */
const tabs=$$(".tabs button");
function showTab(id){ tabs.forEach(b=>{ const on=b.id===id; b.setAttribute("aria-selected",on); $("#"+b.getAttribute("aria-controls")).hidden=!on; }); if(history.replaceState) history.replaceState(null,"","#"+id.replace("t-","")); }
tabs.forEach(b=>b.addEventListener("click",()=>showTab(b.id)));
const hashTab={go:"t-go",mol:"t-mol",dna:"t-dna",scan:"t-scan",evidence:"t-ev",ev:"t-ev",method:"t-me",me:"t-me"}[location.hash.slice(1)]; if(hashTab) showTab(hashTab);
[["m-n"],["m-g"],["m-wb"],["m-wa"],["m-wt"],["m-tau"],["d-n"],["d-g"],["d-tau"],["d-wl"],["s-k"]].forEach(([id])=>{ const el=$("#"+id), o=$("#"+id+"-o"); const f=()=>o.textContent=(+el.value).toString().replace(/^(\d)$/,(m,a)=>a); el.addEventListener("input",f); f(); });
$("#m-tau-o").textContent=(+$("#m-tau").value).toFixed(1); $("#m-tau").addEventListener("input",e=>$("#m-tau-o").textContent=(+e.target.value).toFixed(1));
$("#d-tau-o").textContent=(+$("#d-tau").value).toFixed(1); $("#d-tau").addEventListener("input",e=>$("#d-tau-o").textContent=(+e.target.value).toFixed(1));
$("#d-lib").addEventListener("change",e=>$("#d-paste").style.display=e.target.value==="paste"?"block":"none");
$("#s-src").addEventListener("change",e=>$("#s-paste").style.display=e.target.value==="paste"?"block":"none");

/* ---------- boot ---------- */
async function boot(){
  try{
    const [rd,judges,ref,ptprs]=await Promise.all([
      initRDKitModule({locateFile:f=>BASE+"vendor/"+f}),
      getJSON("models/judges.json"), getJSON("models/ref.json"), getJSON("models/ptprs.json")]);
    S.rd=rd; S.judges=judges; S.ref=ref.map(([s,bits])=>({s,bits:new Set(bits)})); S.ptprs=ptprs;
    S.worker=new Worker(BASE+"worker.js"); S.worker.onmessage=onWorker; S.worker.onerror=e=>fail("Worker error: "+e.message);
    S.worker.postMessage({cmd:"init"});
  }catch(err){ fail("Could not load the models. "+(err&&err.message||err)+" Reload the page; if it persists, open orvillerockwelljr.com/projects/diffusion-design/ directly."); }
}
function fail(msg){ $("#loadmsg").innerHTML='<div class="err">'+esc(msg)+'</div>'; }
function onWorker(ev){
  const m=ev.data;
  if(m.evt==="ready"){ S.ready=true; $("#loadmsg").textContent="Models loaded ("+m.info.molHeads+"-head critic ensembles; runs on your device)."; ["m-go","d-go","s-go","g-read"].forEach(i=>$("#"+i).disabled=false); scInit(); if($("#g-prompt").value.trim()) doRead(); setTimeout(()=>{ if(S.ready) $("#loadmsg").style.display="none"; },2500); loadEvidence(); }
  else if(m.evt==="trace") onTrace(m);
  else if(m.evt==="item") onItem(m);
  else if(m.evt==="done") onDone(m);
  else if(m.evt==="scores"){ if(S.sc&&S.sc.id===m.id) onScScores(m); else onScores(m); }
  else if(m.evt==="report") onReport(m);
  else if(m.evt==="scanprog") $("#s-status").textContent="Scoring "+m.i+" / "+m.n+" sites…";
  else if(m.evt==="error") fail(m.message);
}

/* ---------- chemistry helpers ---------- */
const DESC_KEYS={mw:["exactmw","amw"],logp:["CrippenClogP","crippenClogP"],tpsa:["tpsa","TPSA"],hbd:["NumHBD","lipinskiHBD"],hba:["NumHBA","lipinskiHBA"],rot:["NumRotatableBonds"],qed:["QED","qed"],rings:["NumRings"]};
function pick(d,keys){ for(const k of keys) if(k in d) return d[k]; return NaN; }
function analyse(smi){
  if(!smi) return null; let m=null; try{ m=S.rd.get_mol(smi); }catch(e){ m=null; } if(!m||!m.is_valid()){ if(m) m.delete(); return null; }
  const out={}; try{ const d=JSON.parse(m.get_descriptors()); for(const k in DESC_KEYS) out[k]=pick(d,DESC_KEYS[k]); }catch(e){}
  out.ro5=(out.mw>500?1:0)+(out.logp>5?1:0)+(out.hbd>5?1:0)+(out.hba>10?1:0); out.canon=m.get_smiles(); out.svg=m.get_svg(300,200);
  const fp=m.get_morgan_fp(JSON.stringify({radius:2,nBits:1024})); const on=[]; for(let i=0;i<fp.length;i++) if(fp.charCodeAt(i)===49) on.push(i); out.fp=on; out.fpset=new Set(on);
  m.delete(); return out;
}
function drawSvg(smi,w,h){ try{ const m=S.rd.get_mol(smi); if(!m||!m.is_valid()){ if(m) m.delete(); return null; } const s=m.get_svg(w||132,h||100); m.delete(); return s; }catch(e){ return null; } }
function judgeMol(a){ const o={}; for(const p of ["bbb","act","tox"]){ const J=S.judges.mol[p]; let z=J.b; for(const i of a.fp) z+=J.coef[i]; o[p]=z; } return o; }
function nearest(a){ let best=0,bs=null; for(const r of S.ref){ let inter=0; for(const b of a.fp) if(r.bits.has(b)) inter++; const t=inter/(a.fp.length+r.bits.size-inter); if(t>best){best=t;bs=r.s;} } return {t:best,s:bs}; }

/* ---------- molecule generation ---------- */
function resetBar(p){ $(p+"-bar").style.width="0"; }
$("#m-go").addEventListener("click",()=>{
  const job={cmd:"genMol",id:++S.job,n:+$("#m-n").value,seed:+$("#m-seed").value||0,steps:+$("#m-steps").value,gamma:+$("#m-g").value,w:[+$("#m-wb").value,+$("#m-wa").value,+$("#m-wt").value],gate:$("#m-gate").checked,tau:+$("#m-tau").value,compare:$("#m-cmp").checked&&+$("#m-g").value>0};
  S.cur={kind:"mol",job,t0:performance.now(),got:0}; S.mol={items:[],cmp:job.compare,job};
  $("#m-cards").innerHTML=""; $("#m-trace").innerHTML=""; $("#m-strip").innerHTML=""; $("#m-leg").hidden=true; $("#m-sort").hidden=true; $("#m-traceh").hidden=false; resetBar("#m"); $("#m-go").disabled=true; $("#m-stop").hidden=false;
  $("#m-status").textContent="Denoising "+job.n+" candidates…"; S.worker.postMessage(job);
});
$("#m-stop").addEventListener("click",()=>{ S.worker.postMessage({cmd:"cancel",id:S.cur.job.id}); });
function onTrace(m){
  if(m.kind==="mol"){ const box=$("#m-trace"); const fig=document.createElement("figure"); const svg=m.smiles?drawSvg(m.smiles):null; const pc=Math.round((m.si+1)/m.n*100);
    fig.innerHTML='<div class="tb'+(svg?"":" inv")+'">'+(svg||"not a valid molecule yet")+'</div>step '+(m.si+1)+'/'+m.n+(m.gate<0.999?' · gate '+fmt(m.gate):''); box.appendChild(fig); box.scrollLeft=box.scrollWidth; }
  else if(m.kind==="dna"){ renderTraceDna(m); }
}
function onItem(m){
  if(!S.cur||S.cur.job.id!==m.id) return; S.cur.got++; if(S.cur.chain) chainItem(m);
  const bar=m.kind==="mol"?"#m-bar":"#d-bar"; $(bar).style.width=(S.cur.got/S.cur.job.n*100)+"%";
  if(m.kind==="mol"){ S.mol.items.push({i:m.i,g:procMol(m.guided),u:m.unguided?procMol(m.unguided):null}); if(!S.cur.chain) renderMol(); if(!S.cur.chain) $("#m-status").textContent="Generated "+S.cur.got+" / "+S.cur.job.n+"…"; }
  else{ S.dna.items.push({i:m.i,g:procDna(m.guided),u:m.unguided?procDna(m.unguided):null}); if(!S.cur.chain){ renderDna(); $("#d-status").textContent="Generated "+S.cur.got+" / "+S.cur.job.n+"…"; } }
}
function procMol(r){
  const a=analyse(r.smiles); const o={smiles:r.smiles,valid:!!a,crit:r.crit};
  o.cp={bbb:sig(r.crit.bbb.mean),act:sig(r.crit.act.mean),tox:sig(r.crit.tox.mean)};   // P(brain-penetrant), P(active), P(toxic)
  if(a){ Object.assign(o,{a}); const j=judgeMol(a); o.jz=j; o.jp={bbb:sig(j.bbb),act:sig(j.act),tox:sig(j.tox)}; o.jscore=j.bbb+j.act-j.tox; o.pass=j.bbb>0&&j.act>0&&j.tox<0; o.nn=nearest(a); }
  o.cscore=r.crit.bbb.mean+r.crit.act.mean-r.crit.tox.mean; return o;
}
function pair(label,cp,jp,good){ // good: true if high is good, false if low is good (toxicity)
  const cw=Math.round((good?cp:1-cp)*100), jw=jp==null?null:Math.round((good?jp:1-jp)*100);
  return '<div class="pair c"><span>'+label+'</span><div class="tr"><i style="width:'+cw+'%"></i></div><b>'+cw+'</b></div>'+(jw==null?'':'<div class="pair j"><span></span><div class="tr"><i style="width:'+jw+'%"></i></div><b>'+jw+'</b></div>');
}
function auditDna(o){ if(o.cp>=0.8&&o.jp<0.5) return "hack"; return o.pass&&Math.abs(o.cp-o.jp)<=0.25?"ok":"mid"; }
function auditMol(o){ if(!o.valid) return null; const g=[[o.cp.bbb,o.jp.bbb],[o.cp.act,o.jp.act],[1-o.cp.tox,1-o.jp.tox]]; if(g.some(([c,j])=>c>=0.8&&j<0.5)) return "hack"; return o.pass&&g.every(([c,j])=>Math.abs(c-j)<=0.3)?"ok":"mid"; }
function auditTag(k){ return k==="hack"?'<span class="tag no" title="Rule: the steering critic scores it 0.8 or higher on an objective while the held-out judge scores it below 0.5">audit: likely exploited</span>':k==="ok"?'<span class="tag ok" title="Judge passes and critic and judge agree">audit: trusted</span>':'<span class="tag" title="Critic and judge disagree, or the judge fails it">audit: disputed</span>'; }
function molCard(o,idx,un){
  if(!o.valid) return '<div class="card'+(un?" un":"")+'"><div class="sv small">invalid structure</div><div class="smi">'+esc(o.smiles||"(empty)")+'</div><span class="tag no">not a valid molecule</span></div>';
  const a=o.a, cns=a.tpsa<90&&a.mw<450&&a.hbd<=3;
  return '<div class="card'+(un?" un":"")+'"><div class="sv">'+a.svg+'</div><div class="smi" title="Click to copy" data-smi="'+esc(a.canon)+'">'+esc(a.canon)+'</div><div>'+
   (o.pass?'<span class="tag ok">judge: passes all three</span>':'<span class="tag no">judge: fails ≥1</span>')+auditTag(auditMol(o))+(cns?'<span class="tag ok">CNS-like physchem</span>':'<span class="tag">outside CNS physchem</span>')+'</div>'+
   pair("brain",o.cp.bbb,o.jp.bbb,true)+pair("activity",o.cp.act,o.jp.act,true)+pair("low tox",o.cp.tox,o.jp.tox,false)+
   '<div class="small" style="font-size:11.5px">MW '+fmt(a.mw,0)+' · cLogP '+fmt(a.logp,1)+' · TPSA '+fmt(a.tpsa,0)+' · HBD '+fmt(a.hbd,0)+' · Ro5 violations '+a.ro5+'</div>'+
   '<div class="small" style="font-size:11.5px">Nearest known compound: Tanimoto '+fmt(o.nn.t)+(o.nn.t>=0.99?' (known)':o.nn.t<0.4?' (new scaffold)':'')+'</div></div>';
}
function stripHtml(items,key,labels){
  const v=items.map(x=>x[key]).filter(Boolean), n=v.length; if(!n) return ""; const valid=v.filter(x=>x.valid);
  const m=f=>valid.length?valid.reduce((s,x)=>s+f(x),0)/valid.length:NaN; const pass=valid.filter(x=>x.pass).length;
  return '<div><span>'+labels+' · valid</span><b>'+pct(valid.length/n)+'</b><small>'+valid.length+' of '+n+' decode to a molecule</small></div>'+
   '<div><span>Judge passes all three</span><b>'+pct(pass/n)+'</b><small>'+pass+' of '+n+' (brain, activity, low toxicity)</small></div>'+
   '<div><span>Critic P(brain) / judge</span><b>'+fmt(m(x=>x.cp.bbb),2)+' / '+fmt(m(x=>x.jp.bbb),2)+'</b><small>steering vs held-out</small></div>'+
   '<div><span>Critic P(active) / judge</span><b>'+fmt(m(x=>x.cp.act),2)+' / '+fmt(m(x=>x.jp.act),2)+'</b><small>steering vs held-out</small></div>'+
   '<div><span>P(toxic) critic / judge</span><b>'+fmt(m(x=>x.cp.tox),2)+' / '+fmt(m(x=>x.jp.tox),2)+'</b><small>lower is better</small></div>';
}
function renderMol(){
  const it=S.mol.items.slice(); const by=$("#m-sortby").value;
  const key={judge:o=>o.valid?o.jscore:-1e9,critic:o=>o.valid?o.cscore:-1e9,qed:o=>o.valid?-o.a.ro5*100+o.jscore:-1e9}[by];
  const g=it.map(x=>x.g).sort((a,b)=>key(b)-key(a));
  let html=""; if(S.mol.cmp){ html+='<div style="grid-column:1/-1;font-family:var(--sans);font-size:14px;font-weight:600;color:var(--navy)">Guided (&gamma; = '+S.mol.job.gamma+')</div>'; }
  html+=g.map((o,i)=>molCard(o,i,false)).join("");
  if(S.mol.cmp){ const u=it.map(x=>x.u).sort((a,b)=>key(b)-key(a)); html+='<div style="grid-column:1/-1;font-family:var(--sans);font-size:14px;font-weight:600;color:var(--navy);margin-top:10px">Unguided (&gamma; = 0), same starting noise</div>'+u.map((o,i)=>molCard(o,i,true)).join(""); }
  $("#m-cards").innerHTML=html;
  let strip='<div style="grid-column:1/-1;display:none"></div>'+stripHtml(it,"g",S.mol.cmp?"Guided":"Candidates"); 
  if(S.mol.cmp) strip+=stripHtml(it,"u","Unguided");
  const sEl=$("#m-strip"); sEl.innerHTML=strip; $("#m-leg").hidden=false; $("#m-sort").hidden=false;
}
$("#m-sortby").addEventListener("change",()=>{ if(S.mol.items.length) renderMol(); });
$("#m-cards").addEventListener("click",e=>{ const t=e.target.closest(".smi"); if(t&&navigator.clipboard){ navigator.clipboard.writeText(t.dataset.smi).then(()=>{ const o=t.textContent; t.textContent="copied"; setTimeout(()=>t.textContent=o,700); }); } });
$("#m-csv").addEventListener("click",()=>{
  const rows=[["set","smiles","valid","critic_p_brain","critic_p_active","critic_p_toxic","judge_p_brain","judge_p_active","judge_p_toxic","judge_pass_all3","mw","clogp","tpsa","hbd","ro5_violations","nearest_tanimoto"]];
  const add=(set,o)=>rows.push([set,o.valid?o.a.canon:o.smiles,o.valid,fmt(o.cp.bbb,3),fmt(o.cp.act,3),fmt(o.cp.tox,3),o.valid?fmt(o.jp.bbb,3):"",o.valid?fmt(o.jp.act,3):"",o.valid?fmt(o.jp.tox,3):"",o.valid?o.pass:"",o.valid?fmt(o.a.mw,1):"",o.valid?fmt(o.a.logp,2):"",o.valid?fmt(o.a.tpsa,1):"",o.valid?o.a.hbd:"",o.valid?o.a.ro5:"",o.valid?fmt(o.nn.t,3):""]);
  S.mol.items.forEach(x=>{ add("guided",x.g); if(x.u) add("unguided",x.u); }); download("ptpsigma_molecules.csv",rows);
});
function download(name,rows){ const csv=rows.map(r=>r.map(v=>/[",\n]/.test(String(v))?'"'+String(v).replace(/"/g,'""')+'"':v).join(",")).join("\n"); const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"})); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000); }

/* ---------- locus library (target sites the guide generator is steered toward) ---------- */
function buildLib(kind,pasted){
  const rows=[]; let label="";
  if(kind==="ptprs"){ for(const ex of S.ptprs.exons) E.findSites(ex.seq,"PTPRS exon "+ex.exon,30,30+ex.len,rows); label="PTPRS exons"; }
  else if(kind==="paste"){ const raw=(pasted||"").toUpperCase().replace(/[^ACGT]/g,"").slice(0,20000); if(raw.length<40) return null; E.findSites(raw,"your sequence",0,raw.length,rows); label="your sequence"; }
  else return null;
  const map=new Map(); for(const r of rows){ if(!r.cutIn&&kind==="ptprs") continue; if(!map.has(r.win)) map.set(r.win,{label:r.label,strand:r.strand}); }
  if(!map.size) return null; return {kind,label,wins:[...map.keys()],map,key:kind+":"+map.size+":"+[...map.keys()][0]+[...map.keys()].slice(-1)};
}
S.lib=null;
function dnaJob(o,lib){ return Object.assign({cmd:"genDna",id:++S.job,ew:1},o,lib?{lib:lib.wins,libKey:lib.key+":"+o.tilt}:{lib:null,wl:0}); }
/* ---------- DNA ---------- */
const NT="ACGT";
function dnaChecks(s){ const gc=[...s.slice(4,24)].filter(c=>c==="G"||c==="C").length/20; const pam=s.slice(25,27)==="GG"; const t4=s.slice(4,24).includes("TTTT"); return {gc,pam,t4,usable:pam&&gc>=0.4&&gc<=0.7&&!t4}; }
function procDna(r){ const seq=r.seq, ck=dnaChecks(seq), jz=E.dnaJudge(seq,S.judges.dna); const hit=S.lib&&S.lib.map.get(seq); return {seq,ck,cp:sig(r.crit.mean),csd:r.crit.sd,jp:sig(jz),jz,pass:ck.usable&&jz>0,hit:hit||null,onpass:!!hit&&ck.usable&&jz>0}; }
function seqHtml(s,conf){ let h=""; for(let i=0;i<s.length;i++){ const cls=(i<4||i>=27?"ctx ":"")+(i>=24&&i<27?"pam ":"")+s[i]; h+='<span class="'+cls+'"'+(conf?' style="opacity:'+(i<4||i>=27?Math.min(.45,.12+conf[i]*.4):(.18+conf[i]*.82))+'"':"")+'>'+s[i]+'</span>'; } return '<div class="seq">'+h+'</div>'; }
function renderTraceDna(m){ $("#d-tracek").hidden=false; $("#d-trace").innerHTML='<div style="font-family:var(--sans);font-size:12px;color:var(--muted);width:100%">step '+(m.si+1)+' / '+m.n+'</div>'+seqHtml(m.seq,m.conf); }
function dnaRow(o,label,un){
  const ck=o.ck; return '<div class="row" style="'+(un?"background:#fafbfc":"")+'"><div>'+(label?'<div style="font-weight:600;color:var(--navy);margin-bottom:4px">'+label+'</div>':'')+seqHtml(o.seq)+'<div class="seqkey">'+o.seq.slice(4,24)+' + '+o.seq.slice(24,27)+' (protospacer + PAM)</div></div><div>'+
   (o.pass?'<span class="tag ok">judge: efficient + passes checks</span>':'<span class="tag no">fails judge or checks</span>')+auditTag(auditDna(o))+(ck.pam?'<span class="tag ok">NGG PAM</span>':'<span class="tag no">no NGG PAM</span>')+'<span class="tag '+(ck.gc>=.4&&ck.gc<=.7?'ok':'no')+'">GC '+Math.round(ck.gc*100)+'%</span>'+(ck.t4?'<span class="tag no">TTTT run</span>':'')+(S.lib?(o.hit?'<span class="tag ok">exact site: '+esc(o.hit.label)+', '+(o.hit.strand>0?'+':'−')+' strand</span>':'<span class="tag no">not a site in '+esc(S.lib.label)+'</span>'):'')+
   pair("critic",o.cp,null,true).replace('pair c','pair c')+pair("judge",o.jp,null,true).replace('class="pair c"','class="pair j"')+'</div><div style="grid-column:1/-1"><button type="button" class="repbtn" data-seq="'+o.seq+'">Per-base report</button><div class="rep-out"></div></div></div>';
}
$("#d-go").addEventListener("click",()=>{
  const lib=buildLib($("#d-lib").value,$("#d-paste").value); if($("#d-lib").value==="paste"&&!lib){ $("#d-status").textContent="Paste at least 40 letters of A, C, G, T."; return; } S.lib=lib;
  const job=dnaJob({n:+$("#d-n").value,seed:+$("#d-seed").value||0,steps:+$("#d-steps").value,gamma:+$("#d-g").value,gate:$("#d-gate").checked,tau:+$("#d-tau").value,compare:$("#d-cmp").checked&&+$("#d-g").value>0,wl:+$("#d-wl").value,tilt:$("#d-tilt").checked?2:0},lib);
  S.cur={kind:"dna",job,got:0}; S.dna={items:[],job,cmp:job.compare}; $("#d-rows").innerHTML=""; $("#d-strip").innerHTML=""; $("#d-trace").innerHTML=""; resetBar("#d"); $("#d-go").disabled=true; $("#d-status").textContent="Denoising "+job.n+" candidates…"; S.worker.postMessage(job);
});
function renderDna(){
  const it=S.dna.items; const g=it.map(x=>x.g).sort((a,b)=>b.jz-a.jz); const u=it.map(x=>x.u).filter(Boolean);
  const st=(arr,l)=>{ const n=arr.length; const m=f=>arr.reduce((s,x)=>s+f(x),0)/n; return '<div><span>'+l+' · valid PAM</span><b>'+pct(m(x=>x.ck.pam?1:0))+'</b><small>NGG at the PAM position</small></div><div><span>Critic / judge P(efficient)</span><b>'+fmt(m(x=>x.cp))+' / '+fmt(m(x=>x.jp))+'</b><small>steering vs held-out</small></div><div><span>Pass judge + checks</span><b>'+pct(m(x=>x.pass?1:0))+'</b><small>'+arr.filter(x=>x.pass).length+' of '+n+'</small></div>'; };
  $("#d-strip").innerHTML=st(g,S.dna.cmp?"Guided":"Candidates")+(u.length?st(u,"Unguided"):"");
  $("#d-rows").innerHTML=(S.dna.cmp?'<div style="font-family:var(--sans);font-weight:600;color:var(--navy);margin:4px 0 8px">Guided (&gamma; = '+S.dna.job.gamma+')</div>':"")+g.map(o=>dnaRow(o,"",false)).join("")+(u.length?'<div style="font-family:var(--sans);font-weight:600;color:var(--navy);margin:14px 0 8px">Unguided, same starting noise</div>'+u.sort((a,b)=>b.jz-a.jz).map(o=>dnaRow(o,"",true)).join(""):"");
}
function onDone(m){ if(!S.cur||S.cur.job.id!==m.id) return; if(S.cur.chain){ chainDone(m); return; } const k=S.cur.kind; const sec=((performance.now()-(S.cur.t0||performance.now()))/1000);
  if(k==="mol"){ $("#m-go").disabled=false; $("#m-stop").hidden=true; $("#m-status").textContent=(m.cancelled?"Stopped. ":"Done. ")+S.mol.items.length+" candidates"+(S.cur.t0?" in "+sec.toFixed(1)+" s":"")+" on your device."; $("#m-bar").style.width="100%"; }
  else if(k==="dna"){ $("#d-go").disabled=false; $("#d-status").textContent="Done. "+S.dna.items.length+" candidates on your device."; $("#d-bar").style.width="100%"; } }

/* ---------- scan ---------- */
const {findSites,rc}=E;
$("#s-go").addEventListener("click",()=>{
  const rows=[]; let src="PTPRS";
  if($("#s-src").value==="ptprs"){ for(const ex of S.ptprs.exons){ findSites(ex.seq,"exon "+ex.exon,30,30+ex.len,rows); } }
  else{ const raw=$("#s-paste").value.toUpperCase().replace(/[^ACGT]/g,"").slice(0,20000); src="your sequence"; if(raw.length<40){ $("#s-status").textContent="Paste at least 40 letters of A, C, G, T."; return; } findSites(raw,"pasted",0,raw.length,rows); }
  const seen=new Set(); S.scan={rows:rows.filter(r=>{ const k=r.win+r.label; if(seen.has(k)) return false; seen.add(k); return true; }),src};
  $("#s-go").disabled=true; $("#s-status").textContent="Scoring "+S.scan.rows.length+" sites…"; S.worker.postMessage({cmd:"scoreDna",id:++S.job,seqs:S.scan.rows.map(r=>r.win)});
});
function onScores(m){
  const R=S.scan.rows; R.forEach((r,i)=>{ r.cp=sig(m.scores[i][0]); r.jz=E.dnaJudge(r.win,S.judges.dna); r.jp=sig(r.jz); r.ck=dnaChecks(r.win); });
  $("#s-go").disabled=false; const k=+$("#s-k").value, needCut=$("#s-req").checked&&S.scan.src==="PTPRS", needUse=$("#s-use").checked;
  let pool=R.filter(r=>(!needCut||r.cutIn)&&(!needUse||r.ck.usable)); pool.sort((a,b)=>(b.cp+b.jp)-(a.cp+a.jp)); const top=pool.slice(0,k);
  const agree=pool.filter(r=>r.cp>0.5&&r.jp>0.5).length;
  $("#s-status").textContent="Scored "+R.length+" candidate sites in "+S.scan.src+".";
  $("#s-strip").innerHTML='<div><span>Sites found (both strands)</span><b>'+R.length+'</b><small>NGG with full 30-nt context</small></div><div><span>After your filters</span><b>'+pool.length+'</b><small>'+agree+' scored &gt; 0.5 by both critic and judge</small></div><div><span>Best site (avg of both)</span><b>'+(top[0]?fmt((top[0].cp+top[0].jp)/2):"–")+'</b><small>'+(top[0]?top[0].label:"")+'</small></div>';
  const T=$("#s-tab"); T.hidden=false;
  T.innerHTML='<table class="t"><thead><tr><th>#</th><th>Where</th><th>Strand</th><th>Guide (20 nt) + PAM</th><th class="n">Critic P</th><th class="n">Judge P</th><th class="n">GC</th><th>Cut in exon</th></tr></thead><tbody>'+top.map((r,i)=>'<tr><td>'+(i+1)+'</td><td>'+esc(r.label)+'</td><td>'+(r.strand===1?"sense":"antisense")+'</td><td style="font-family:ui-monospace,Menlo,monospace">'+r.win.slice(4,24)+' <b style="color:var(--bio)">'+r.win.slice(24,27)+'</b></td><td class="n">'+fmt(r.cp)+'</td><td class="n">'+fmt(r.jp)+'</td><td class="n">'+Math.round(r.ck.gc*100)+'%</td><td>'+(r.cutIn?"yes":"no")+'</td></tr>').join("")+'</tbody></table>';
}

/* ---------- evidence + method (built from saved result files) ---------- */
async function loadEvidence(){
  try{
    const [mt,dt,me,de,le]=await Promise.all(["mol_train_metrics","dna_train_metrics","mol_experiments","dna_experiments","locus_experiments"].map(n=>getJSON("results/"+n+".json")));
    S.locusRes=le; renderEvidence(mt,dt,me,de);
  }catch(err){ $("#ev-body").innerHTML='<div class="err">Results files not found: '+esc(err)+'</div>'; }
  renderMethod();
}
const f2=v=>Number.isFinite(v)?v.toFixed(2):"–", f3=v=>Number.isFinite(v)?v.toFixed(3):"–", ms=a=>a[0].toFixed(2)+" ± "+a[1].toFixed(2);
function lineChart(title,series,yl,xl,opts){
  const W=420,H=240,P={l:42,r:12,t:12,b:36}; const xs=opts.xs; const xmax=Math.max(...xs),xmin=Math.min(...xs); const ymax=opts.ymax, ymin=opts.ymin||0;
  const X=x=>P.l+(x-xmin)/(xmax-xmin||1)*(W-P.l-P.r), Y=y=>H-P.b-(y-ymin)/(ymax-ymin)*(H-P.t-P.b);
  let g=""; for(let i=0;i<=4;i++){ const v=ymin+(ymax-ymin)*i/4; g+='<line x1="'+P.l+'" x2="'+(W-P.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'" class="gl"/><text x="'+(P.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end" fill="currentColor" font-size="10">'+v.toFixed(opts.d||2)+'</text>'; }
  xs.forEach(x=>g+='<text x="'+X(x)+'" y="'+(H-P.b+14)+'" text-anchor="middle" fill="currentColor" font-size="10">'+x+'</text>');
  g+='<text x="'+(W/2)+'" y="'+(H-4)+'" text-anchor="middle" fill="currentColor" font-size="11">'+xl+'</text><text transform="translate(11,'+(H/2)+') rotate(-90)" text-anchor="middle" fill="currentColor" font-size="11">'+yl+'</text>';
  series.forEach(s=>{ if(s.hline!=null){ g+='<line x1="'+P.l+'" x2="'+(W-P.r)+'" y1="'+Y(s.hline)+'" y2="'+Y(s.hline)+'" stroke="'+s.c+'" stroke-dasharray="5 4"/><text x="'+(W-P.r)+'" y="'+(Y(s.hline)-4)+'" text-anchor="end" fill="'+s.c+'" font-size="10">'+s.n+'</text>'; return; }
    const pts=s.pts.map(p=>[X(p[0]),Y(p[1]),p[2]]); g+='<polyline fill="none" stroke="'+s.c+'" stroke-width="2" points="'+pts.map(p=>p[0]+","+p[1]).join(" ")+'"/>';
    pts.forEach((p,i)=>{ if(s.pts[i][2]!=null){ g+='<line x1="'+p[0]+'" x2="'+p[0]+'" y1="'+Y(s.pts[i][1]-s.pts[i][2])+'" y2="'+Y(s.pts[i][1]+s.pts[i][2])+'" stroke="'+s.c+'" opacity=".5"/>'; } g+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="3.5" fill="'+s.c+'"/>'; }); });
  const leg=series.filter(s=>!s.hline).map((s,i)=>'<span style="margin-right:14px"><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+s.c+';vertical-align:middle;margin-right:4px"></i>'+s.n+'</span>').join("");
  return '<div class="chart"><h4>'+title+'</h4><svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(title)+'">'+g+'</svg><div style="margin-top:4px">'+leg+'</div></div>';
}
function renderEvidence(mt,dt,me,de){
  const find=(R,g,gate)=>R.results.find(r=>r.gamma===g&&r.gate===gate), base=find(me,0,"none").metrics, baseD=find(de,0,"none").metrics;
  const gs=[1,2,4,8], gD=[1,2,4,8,16];
  const ser=(R,key,gl,gate,c,n)=>({n,c,pts:[[0,find(R,0,"none").metrics[key][0],find(R,0,"none").metrics[key][1]]].concat(gl.map(g=>[g,find(R,g,gate).metrics[key][0],find(R,g,gate).metrics[key][1]]))});
  const cls=(k)=>mt[k];
  let h='<h3>1. Do the classifiers work on chemistry they have not seen?</h3><p class="small">Scaffold split: test molecules share no Bemis–Murcko scaffold with training molecules. The steering critic is a small network on the latent code (ensemble mean); the held-out judge is logistic regression on ECFP4 fingerprints trained on <b>different data</b>, so each model is also scored on the other&rsquo;s test set. A model that only works on its own data would drop there.</p>'+
   '<div class="tw"><table class="t"><thead><tr><th>Property</th><th>Critic data / judge data</th><th class="n">Critic train / test</th><th class="n">Critic AUC (clean)</th><th class="n">Critic AUC (noised, t=100)</th><th class="n">Judge AUC (own test)</th><th class="n">Critic AUC on judge&rsquo;s test</th><th class="n">Judge AUC on critic&rsquo;s test</th></tr></thead><tbody>'+
   [["bbb","Brain penetration","B3DB (minus BBBP) / BBBP"],["act","PTP-family activity","ChEMBL / ChEMBL (same source, not independent)"],["tox","Toxicity","Tox21 / ToxCast non-Tox21 assays"]].map(([k,n,src])=>'<tr><td>'+n+'</td><td class="small">'+src+'</td><td class="n">'+cls(k).n_train+' / '+cls(k).n_test+'</td><td class="n">'+f3(cls(k).critic_auc_t0)+'</td><td class="n">'+f3(cls(k).critic_auc_t100)+'</td><td class="n">'+f3(cls(k).judge_auc_test)+'</td><td class="n">'+(cls(k).judge_independent?f3(cls(k).critic_auc_on_judge_test):"–")+'</td><td class="n">'+(cls(k).judge_independent?f3(cls(k).judge_auc_on_critic_test):"–")+'</td></tr>').join("")+'</tbody></table></div>';
  const px=mt.ptpsigma_external; h+='<p class="small" style="margin-top:8px"><b>PTP&sigma; check (n = '+px.n+' molecules, '+px.n_active+' active; too few to rank on).</b> Critic AUC '+f2(px.critic_auc)+' (95% bootstrap CI '+f2(px.critic_ci[0])+'–'+f2(px.critic_ci[1])+'), judge AUC '+f2(px.judge_auc)+' ('+f2(px.judge_ci[0])+'–'+f2(px.judge_ci[1])+'). ChEMBL holds 30 distinct PTP&sigma; molecules with exact potencies; the activity classifier learns from the wider phosphatase family and is only a weak proxy for PTP&sigma; itself.</p>';
  h+='<h3 style="margin-top:26px">2. Does guidance move generated molecules, and does the held-out judge agree?</h3><p class="small">'+me.N+' molecules per setting × '+me.seeds.length+' seeds ('+(me.N*me.seeds.length)+' per row), '+me.steps+' denoising steps, identical starting noise across rows. &ldquo;All three&rdquo; = valid, and the judge calls brain-penetrant, active and non-toxic. Top-10% = the best tenth ranked by the critics, the fair baseline for &ldquo;generate then filter&rdquo;.</p>';
  h+='<div class="tw"><table class="t"><thead><tr><th>&gamma;</th><th>Trust gate</th><th class="n">Valid</th><th class="n">Novel</th><th class="n">Critic P(brain / active / toxic)</th><th class="n">Judge P(brain / active / toxic)</th><th class="n">Judge: all three</th><th class="n">Top-10%: all three</th></tr></thead><tbody>'+
   me.results.map(r=>{ const m=r.metrics; return '<tr><td>'+r.gamma+'</td><td>'+(r.gamma===0?"–":r.gate==="ens"?"on":"off")+'</td><td class="n">'+f2(m.valid[0])+'</td><td class="n">'+f2(m.novel_of_unique[0])+'</td><td class="n">'+f2(m.critic_p_bbb[0])+' / '+f2(m.critic_p_act[0])+' / '+f2(m.critic_p_tox[0])+'</td><td class="n">'+f2(m.judge_p_bbb[0])+' / '+f2(m.judge_p_act[0])+' / '+f2(m.judge_p_tox[0])+'</td><td class="n">'+ms(m.judge_success_all3)+'</td><td class="n">'+ms(m.topk_judge_success_all3)+'</td></tr>'; }).join("")+
   '<tr><td colspan="6"><i>Random ZINC molecules (reference)</i></td><td class="n">'+f2(me.reference.zinc_random.judge_success_all3)+'</td><td></td></tr></tbody></table></div>';
  h+='<div class="charts" style="margin-top:16px">'+
   lineChart("Held-out judge: all three pass",[ser(me,"judge_success_all3",gs,"none","#8BA6C9","gate off"),ser(me,"judge_success_all3",gs,"ens","#B23A64","gate on")],"fraction of samples","guidance strength γ",{xs:[0,1,2,4,8],ymax:Math.max(0.05,...me.results.map(r=>r.metrics.judge_success_all3[0]))*1.25,d:2})+
   lineChart("Critic–judge gap, P(brain)",[ser(me,"gap_bbb",gs,"none","#8BA6C9","gate off"),ser(me,"gap_bbb",gs,"ens","#B23A64","gate on")],"critic − judge","guidance strength γ",{xs:[0,1,2,4,8],ymin:0,ymax:Math.max(0.1,...me.results.map(r=>r.metrics.gap_bbb[0]))*1.25,d:2})+'</div>';
  h+='<h3 style="margin-top:26px">3. Gene-editing guide sequences</h3><p class="small">The critic learned from one lab&rsquo;s SpCas9 library (Kim 2019, '+dt.n_train+' guides) and the judge from a different one (CRISPRon 2021, '+dt.judge_n_train+' guides), with no guide in both. Each model is scored on three held-out sets: the critic&rsquo;s own lab, the judge&rsquo;s lab, and the Doench screens (a third assay; genes CD13, CD45, NF2, THY1, CD28). The prior is a diffusion model trained on '+(de.prior_n||"~110,000")+' NGG-bearing sites from the coding sequence of 419 random human genes.</p>'+
   '<div class="tw"><table class="t"><thead><tr><th>Held-out set</th><th class="n">Guides</th><th class="n">Critic AUC</th><th class="n">Judge AUC</th></tr></thead><tbody>'+
   [["kim_test","Kim 2019 test (critic&rsquo;s lab)"],["luo_test","CRISPRon 2021 test (judge&rsquo;s lab)"],["azimuth_test","Doench genes (third assay)"]].map(([k,n])=>'<tr><td>'+n+'</td><td class="n">'+dt.eval["critic@"+k].n+'</td><td class="n">'+f3(dt.eval["critic@"+k].auc)+'</td><td class="n">'+f3(dt.eval["judge@"+k].auc)+'</td></tr>').join("")+'</tbody></table></div>';
  h+='<div class="tw" style="margin-top:12px"><table class="t"><thead><tr><th>&gamma;</th><th>Gate</th><th class="n">Valid PAM</th><th class="n">Novel</th><th class="n">Critic P</th><th class="n">Judge P</th><th class="n">Usable &amp; judge-efficient</th><th class="n">Top-10%</th></tr></thead><tbody>'+
   de.results.map(r=>{ const m=r.metrics; return '<tr><td>'+r.gamma+'</td><td>'+(r.gamma===0?"–":r.gate==="ens"?"on":"off")+'</td><td class="n">'+f2(m.pam_valid[0])+'</td><td class="n">'+f2(m.novel_of_unique[0])+'</td><td class="n">'+f2(m.critic_p[0])+'</td><td class="n">'+f2(m.judge_p[0])+'</td><td class="n">'+ms(m.judge_success)+'</td><td class="n">'+ms(m.topk_judge_success)+'</td></tr>'; }).join("")+
   '<tr><td colspan="6"><i>Real held-out Doench guides (reference) / natural genomic sites</i></td><td class="n">'+f2(de.reference.doench_test_guides.judge_success)+' / '+f2(de.reference.genomic_prior_sites.judge_success)+'</td><td></td></tr></tbody></table></div>';
  h+='<div class="charts" style="margin-top:16px">'+lineChart("Guides: judge-efficient and usable",[ser(de,"judge_success",gD,"none","#8BA6C9","gate off"),ser(de,"judge_success",gD,"ens","#B23A64","gate on")],"fraction of samples","guidance strength γ",{xs:[0,1,2,4,8,16],ymax:Math.max(0.05,...de.results.map(r=>r.metrics.judge_success[0]))*1.25})+lineChart("Guides: critic–judge gap",[ser(de,"gap",gD,"none","#8BA6C9","gate off"),ser(de,"gap",gD,"ens","#B23A64","gate on")],"critic − judge","guidance strength γ",{xs:[0,1,2,4,8,16],ymin:Math.min(0,...de.results.map(r=>r.metrics.gap[0])),ymax:Math.max(0.1,...de.results.map(r=>r.metrics.gap[0]))*1.25})+'</div>';
  const note=(me.notes||"")+" "; h+='<div class="callout" id="ev-read"></div>';
  if(S.locusRes){ const L=S.locusRes, f=(v,d)=>v?(v[0]*100).toFixed(0)+"% \u00b1 "+(v[1]*100).toFixed(0):"\u2013"; h+='<h3 style="margin-top:26px">4. Aiming guides at a locus (PTPRS exons)</h3><p class="small" style="margin:6px 0 10px">Library: '+L.library_size+' unique NGG sites whose cut falls inside a PTPRS exon. '+L.N+' guides per row \u00d7 '+L.seeds.length+' seeds, '+L.steps+' steps, guidance \u03b3 = '+L.gamma+'. "Success" = exact PTPRS site, judge P &gt; 0.5, passes the usability checks.</p><div class="tw"><table class="t"><thead><tr><th>Setting</th><th class="n">Exact PTPRS site</th><th class="n">Distinct sites</th><th class="n">Success</th></tr></thead><tbody>'+L.results.map(r=>"<tr><td>"+r.setting+"</td><td class=\"n\">"+f(r.exact)+"</td><td class=\"n\">"+f(r.distinct)+"</td><td class=\"n\">"+f(r.success)+"</td></tr>").join("")+"<tr><td><i>Baseline: random draw from the library</i></td><td class=\"n\">100%</td><td class=\"n\">\u2013</td><td class=\"n\">"+f(L.baselines.random_library_draw.success)+"</td></tr><tr><td><i>Baseline: scan and rank (top by critic)</i></td><td class=\"n\">100%</td><td class=\"n\">100%</td><td class=\"n\">"+(L.baselines.scan_and_rank_top_by_critic.success*100).toFixed(0)+"% (deterministic)</td></tr></tbody></table></div>"; }
  $("#ev-body").innerHTML=h; fetch(BASE+"results/READING.html").then(r=>r.ok?r.text():"").then(t=>{ if(t) $("#ev-read").innerHTML=t; else $("#ev-read").remove(); }).catch(()=>$("#ev-read").remove());
}
function renderMethod(){
  fetch(BASE+"results/METHOD.html").then(r=>r.text()).then(t=>$("#me-body").innerHTML=t).catch(()=>{});
}
/* ---------- describe-what-you-want: rule-based request reader + combined run ---------- */
const NUMW={one:1,two:2,three:3,four:4,five:5,six:6,eight:8,ten:10,twelve:12,sixteen:16,twenty:20,"twenty-four":24};
function readPrompt(raw){
  const t=raw.toLowerCase(); const notes=[]; const out={make:"both",n:12,push:"std",lib:"ptprs",w:[1,1,1],pasted:"",use:""};
  const compact=raw.replace(/\s+/g,""); const m=compact.match(/[ACGTacgt]{40,}/); if(m){ out.pasted=m[0].toUpperCase(); out.lib="paste"; notes.push("Found "+out.pasted.length+" letters of DNA in your text; guides will be steered toward cut sites inside it."); }
  const rest=m?raw.replace(/[ACGTacgt\s]{40,}/,"[sequence]"):raw; out.use=rest.trim().replace(/\s+/g," ").slice(0,220);
  const wantDna=/\b(guide|guides|grna|sgrna|crispr|cas9|cas-9|knock[- ]?out|gene[- ]edit\w*|edit(ing)?|genom\w*|dna|exon|sequence)\b/.test(t), wantMol=/\b(molecule|molecules|small[- ]molecule\w*|compound\w*|drug\w*|inhibitor\w*|ligand\w*|chemical\w*|smiles|oral|pill)\b/.test(t);
  const onlyMol=/\b(molecules?|compounds?) only\b|\bonly (molecules?|compounds?)\b|\bno guides\b/.test(t), onlyDna=/\b(guides?|crispr) only\b|\bonly (guides?|crispr)\b|\bno molecules\b/.test(t);
  if(onlyMol||(wantMol&&!wantDna)) out.make="mol"; else if(onlyDna||(wantDna&&!wantMol)) out.make="dna"; else out.make="both";
  notes.push(out.make==="both"?(wantMol||wantDna?"Making both molecules and guides.":"No clear modality in your text, so I will make both molecules and guides."):out.make==="mol"?"Making small molecules only.":"Making gene-editing guides only.");
  const bb=/\b(brain|bbb|blood[- ]brain|cns|central nervous|cross(es)? the|penetrat\w*|neuro\w*)\b/.test(t), ac=/\b(potent|potency|active|activity|inhibit\w*|bind\w*|affinity|phosphatase|ptp\w*|target\w*)\b/.test(t), tx=/\b(safe|safety|non[- ]?toxic|low[- ]tox\w*|toxic\w*|side effects?|clean)\b/.test(t);
  const off=k=>new RegExp("(ignore|skip|don'?t care( about)?|not (worried|concerned) (about|with)|no need for|without)\\s+(the\\s+)?("+k+")").test(t);
  const hi=(k,flag)=>flag?(new RegExp("(must|need|especially|priority|above all|most important|mainly)[^.]{0,40}("+k+")|("+k+")[^.]{0,20}(is|are) (the )?(priority|key|crucial|essential)").test(t)?2:1.5):1;
  const w=[hi("brain|bbb|cns|penetrat\\w*",bb),hi("potent\\w*|activ\\w*|inhibit\\w*|affinity",ac),hi("safe\\w*|toxic\\w*|non-?toxic",tx)];
  if(off("brain|bbb|cns|blood[- ]brain")) w[0]=0; if(off("potency|activity|inhibition|affinity")) w[1]=0; if(off("toxic\\w*|safety|tox")) w[2]=0;
  out.w=w.map(v=>v>=2?2:v>0?1:0); [[bb,"brain penetration"],[ac,"target activity"],[tx,"low toxicity"]].forEach(([f,n],i)=>{ if(w[i]===0) notes.push("Switching off "+n+" because you said to ignore it."); else if(f) notes.push("Weighting "+n+(out.w[i]===2?" high.":" normally (you mentioned it).")); });
  if(out.make!=="dna"&&!bb&&!ac&&!tx) notes.push("No molecule objectives named, so brain penetration, activity and low toxicity are all on at normal weight.");
  if(/\b(aggressive\w*|strong(ly)?|maxim\w+|push (it )?hard|hard|as (good|strong) as possible)\b/.test(t)) out.push="hard"; else if(/\b(gentle\w*|diverse|diversity|varied|explor\w*|novel|broad)\b/.test(t)) out.push="gentle";
  if(out.push!=="std") notes.push(out.push==="hard"?"Guidance pushed hard (less diverse, more optimized).":"Guidance gentle (more diverse, less optimized).");
  let n=null; const nm=t.match(/\b(\d{1,2})\s*(?:of each|each|candidates?|molecules?|compounds?|guides?|designs?|options|ideas)\b/)||t.match(/\b(?:give|make|generate|design|want|need|show|produce)\s+(?:me\s+)?(?:about\s+)?(\d{1,2}|one|two|three|four|five|six|eight|ten|twelve|sixteen|twenty(?:-four)?)\b/);
  if(nm){ const v=NUMW[nm[1]]||+nm[1]; if(v>0) n=v; } if(n!=null){ const opts=[6,8,12,16,24]; out.n=opts.reduce((b,c)=>Math.abs(c-n)<Math.abs(b-n)?c:b); notes.push("Making "+out.n+(out.make==="both"?" of each":" candidates")+(out.n!==n?" (nearest option to "+n+"; the limit is 24)":"")+"."); }
  const geneM=raw.match(/\b(?:gene|knock(?:ing)?[- ]?out|edit|target|disrupt)\s+(?:the\s+)?([A-Z][A-Z0-9]{2,8})\b/); const ptprs=/\bptprs\b|ptp[-\s]?(σ|sigma)|ptpσ/.test(t);
  if(wantDna||out.make!=="mol"){ if(out.lib==="paste"){} else if(geneM&&geneM[1]!=="PTPRS"&&!ptprs){ out.lib="none"; notes.push("You named "+geneM[1]+", but only PTPRS sequence is built in. Paste "+geneM[1]+" DNA into the request, or I will make generic efficient guides with no locus."); } else if(ptprs||!geneM){ out.lib="ptprs"; notes.push("Guides steered toward PTPRS exon cut sites (PTPRS is the built-in target)."); } }
  notes.push("Anything else in your text is kept as the use-case label in the report but does not change the model.");
  return {cfg:out,notes};
}
$("#g-chips").addEventListener("click",e=>{ const b=e.target.closest("button"); if(!b) return; $("#g-prompt").value=b.dataset.p.replace(/&sigma;/g,"σ"); doRead(); });
$("#g-read").addEventListener("click",doRead);
function doRead(){
  const raw=$("#g-prompt").value.trim(); if(!raw){ $("#g-live").textContent="Waiting for your request."; $("#g-make-go").disabled=true; S.go=null; return; } $("#g-readmsg").textContent="";
  const {cfg,notes}=readPrompt(raw); S.go={cfg,raw}; $("#g-make").value=cfg.make; $("#g-n").value=String(cfg.n); $("#g-push").value=cfg.push; $("#g-lib").value=cfg.lib; $("#g-wb").value=String(cfg.w[0]); $("#g-wa").value=String(cfg.w[1]); $("#g-wt").value=String(cfg.w[2]);
  $("#g-notes").innerHTML=notes.map(n=>"<li>"+esc(n)+"</li>").join(""); $("#g-make-go").disabled=!S.ready; liveLine(cfg);
}
$("#g-make-go").addEventListener("click",()=>{
  if(!S.go||S.go.raw!==$("#g-prompt").value.trim()) doRead(); const g=S.go; if(!g) return; const make=$("#g-make").value, n=+$("#g-n").value, push=$("#g-push").value, libk=$("#g-lib").value;
  const w=[+$("#g-wb").value,+$("#g-wa").value,+$("#g-wt").value]; const seed=+$("#g-seed").value||0;
  const lib=make!=="mol"?buildLib(libk,g.cfg.pasted):null; if(make!=="mol"&&libk==="paste"&&!lib){ $("#g-status").textContent="No pasted DNA found in your request (need 40+ letters)."; return; }
  if(make!=="dna"&&w.every(v=>v===0)){ $("#g-status").textContent="All three molecule objectives are off; switch one on."; return; }
  S.lib=lib; const gam={mol:{gentle:2,std:4,hard:8},dna:{gentle:2,std:4,hard:12}};
  const stages=[]; if(make!=="dna") stages.push({kind:"mol",job:{cmd:"genMol",id:0,n:Math.min(48,n*2),seed,steps:60,gamma:gam.mol[push],w,gate:$("#g-gate").value==="on",tau:1,compare:false}});
  if(make!=="mol") stages.push({kind:"dna",job:dnaJob({n,seed:seed+1,steps:50,gamma:gam.dna[push],gate:false,tau:1,compare:false,wl:1,tilt:2},lib)});
  S.go.run={stages,i:0,make,n,mol:[],dna:[],lib,libk,use:g.cfg.use,push,w,t0:performance.now(),stopped:false}; $("#g-out").innerHTML=""; $("#g-make-go").disabled=true; $("#g-stop").hidden=false; $("#g-bar").style.width="0"; startStage();
});
$("#g-stop").addEventListener("click",()=>{ const r=S.go&&S.go.run; if(!r) return; r.stopped=true; if(S.cur) S.worker.postMessage({cmd:"cancel",id:S.cur.job.id}); });
function startStage(){
  const r=S.go.run, st=r.stages[r.i]; st.job.id=++S.job; S.cur={kind:st.kind,job:st.job,got:0,t0:performance.now(),chain:true};
  if(st.kind==="mol") S.mol={items:[],cmp:false,job:st.job}; else S.dna={items:[],job:st.job,cmp:false};
  $("#g-status").textContent="Stage "+(r.i+1)+" of "+r.stages.length+": "+(st.kind==="mol"?"molecules":"guides")+"…"; S.worker.postMessage(st.job);
}
function chainItem(m){ const r=S.go&&S.go.run; if(!r) return; let done=S.cur.got, tot=0; r.stages.forEach((s,k)=>{ tot+=s.job.n; if(k<r.i) done+=s.job.n; }); $("#g-bar").style.width=(done/tot*100)+"%"; }
function chainDone(m){
  const r=S.go.run, st=r.stages[r.i]; if(st.kind==="mol") r.mol=S.mol.items.map(x=>x.g); else r.dna=S.dna.items.map(x=>x.g);
  r.i++; if(r.i<r.stages.length&&!m.cancelled&&!r.stopped){ startStage(); return; }
  $("#g-make-go").disabled=false; $("#g-stop").hidden=true; $("#g-bar").style.width="100%"; const sec=(performance.now()-r.t0)/1000; $("#g-status").textContent=(r.stopped?"Stopped. ":"Done. ")+"Took "+sec.toFixed(1)+" s on your device."; renderShortlist();
}
function renderShortlist(){
  const r=S.go.run, K=6; let html='<h3 class="h3s">Shortlist'+(r.use?' for: <i>'+esc(r.use.length>150?r.use.slice(0,150)+"…":r.use)+'</i>':'')+'</h3>';
  const parts=[];
  if(r.make!=="dna"){ const v0=r.mol.filter(o=>o.valid); const v=v0.filter(o=>o.a.ro5<=1&&o.a.mw<=600&&o.a.logp<=6&&o.a.mw>=120); const dropped=v0.length-v.length; const pass=v.filter(o=>o.pass); const ranked=v.slice().sort((a,b)=>(b.jz.bbb*r.w[0]+b.jz.act*r.w[1]-b.jz.tox*r.w[2])-(a.jz.bbb*r.w[0]+a.jz.act*r.w[1]-a.jz.tox*r.w[2])).slice(0,K);
    parts.push('<div><h3>Small molecules</h3><p class="small" style="margin-bottom:8px">'+v0.length+' of '+r.mol.length+' decoded to a valid molecule'+(dropped?'; '+dropped+' removed by a basic drug-likeness filter (MW 120 to 600, cLogP &le; 6, at most one rule-of-five violation), because a steered generator can find structures the judge likes but no chemist would make':'')+'. '+pass.length+' of the rest pass the held-out judge on all three objectives. Generated '+r.mol.length+' (fewer than half decode, so the run draws twice the number you asked for). Showing the top '+ranked.length+' by your weights.</p><div class="cards">'+(ranked.length?ranked.map(o=>molCard(o,0,false)).join(""):'<p class="small">No valid molecules this run. Try a different seed or a gentler push.</p>')+'</div></div>'); }
  if(r.make!=="mol"){ const g=r.dna; const on=g.filter(o=>o.hit), onp=g.filter(o=>o.onpass), pass=g.filter(o=>o.pass);
    const ranked=g.slice().sort((a,b)=>(+!!b.hit-+!!a.hit)||(b.jz-a.jz)).filter((o,i,arr)=>arr.findIndex(x=>x.seq===o.seq)===i).slice(0,K);
    const head=r.lib?(on.length+' of '+g.length+' are exact cut sites in '+r.lib.label+'; '+onp.length+' of those also pass the judge and the checks ('+new Set(on.map(o=>o.seq)).size+' distinct sites).'):(pass.length+' of '+g.length+' pass the judge and the checks (no locus was set).');
    parts.push('<div><h3>Gene-editing guides</h3><p class="small" style="margin-bottom:8px">'+head+' Showing the top '+ranked.length+' distinct.</p><div class="dna">'+ranked.map(o=>dnaRow(o,"",false)).join("")+'</div></div>'); }
  const gdna=r.make!=="mol"?r.dna:[], gmol=r.make!=="dna"?r.mol.filter(o=>o.valid):[]; const gaps=[...gdna.map(o=>Math.abs(o.cp-o.jp)),...gmol.flatMap(o=>[Math.abs(o.cp.bbb-o.jp.bbb),Math.abs(o.cp.act-o.jp.act),Math.abs(o.cp.tox-o.jp.tox)])]; const gapm=gaps.length?gaps.reduce((s,v)=>s+v,0)/gaps.length:NaN; const nflag=gmol.filter(o=>auditMol(o)==="hack").length+gdna.filter(o=>auditDna(o)==="hack").length; const nreq=r.n*(r.make==="both"?2:1), ngen=(r.make!=="dna"?r.mol.length:0)+gdna.length, passN=gmol.filter(o=>o.pass).length+gdna.filter(o=>o.pass).length, onT=gdna.filter(o=>o.hit).length;
  html+='<div class="metrics"><div><span>Requested</span><b>'+nreq+'</b><small>'+(r.make==="both"?r.n+" molecules + "+r.n+" guides":r.n+(r.make==="mol"?" molecules":" guides"))+'</small></div><div><span>Generated</span><b>'+ngen+'</b><small>'+(r.make!=="dna"?r.mol.length+" molecule draws, "+gmol.length+" valid":"")+(r.make==="both"?", ":"")+(r.make!=="mol"?gdna.length+" guides":"")+'</small></div>'+(r.make!=="mol"&&r.lib?'<div><span>On target</span><b>'+onT+'</b><small>of '+gdna.length+' guides are exact cut sites in '+esc(r.lib.label)+'</small></div>':'')+'<div><span>Pass held-out judge</span><b>'+passN+'</b><small>of '+(gmol.length+gdna.length)+' valid</small></div><div><span>Audit flags</span><b>'+nflag+'</b><small>likely exploited: critic confident, judge not</small></div><div><span>Critic&ndash;judge gap</span><b>'+fmt(gapm)+'</b><small>mean |critic P &minus; judge P|</small></div></div>';
  const stg=r.stages.map(s=>s.job), pm=stg.find(j=>j.cmd==="genMol"), pd=stg.find(j=>j.cmd==="genDna");
  html+='<p class="small" style="margin:2px 0 14px"><b>How these were made.</b> Each candidate was denoised from pure noise by a trained diffusion model (cosine schedule, T = 200, DDIM sampling), with the gradient of the critic ensemble added at every step. '+(pm?'Molecules: 64-d latent, '+pm.steps+' steps, guidance &gamma; = '+pm.gamma+', uncertainty gate '+(pm.gate?'on':'off')+'. ':'')+(pd?'Guides: 30-nt one-hot, '+pd.steps+' steps, guidance &gamma; = '+pd.gamma+(pd.lib?', pulled toward PTPRS cut sites':'')+'. ':'')+'Seed '+(pm?pm.seed:pd.seed)+'. Not autoregressive: nothing is generated token by token.</p>';
  html+='<div class="twocol">'+parts.join("")+'</div><p class="small" style="margin-top:14px">Scores are predictions from small public-data models; the judge is a separate model, not an experiment. For a known locus, ranking every site on the Scan tab is the strongest baseline for guides; see the Evidence tab for how this generator compares.</p>';
  $("#g-out").innerHTML=html;
}

/* ---------- Evo-Designer-style dock: modes, live reader, gear ---------- */
function liveLine(cfg){ const mk={both:"molecules + guides",mol:"molecules",dna:"guides"}[cfg.make], pu={gentle:"gentle",std:"standard",hard:"hard"}[cfg.push];
  const tg=cfg.make==="mol"?"":(cfg.lib==="ptprs"?" · target PTPRS exons":cfg.lib==="paste"?" · target your DNA ("+cfg.pasted.length+" bases)":" · no locus");
  $("#g-live").textContent="Reads as: "+mk+" · "+cfg.n+(cfg.make==="both"?" each":"")+" · "+pu+" push"+tg; }
let liveT=null; $("#g-prompt").addEventListener("input",()=>{ clearTimeout(liveT); liveT=setTimeout(()=>{ if(S.ready) doRead(); else $("#g-live").textContent=$("#g-prompt").value.trim()?"Models still loading…":"Waiting for your request."; },250); });
$("#g-lib").addEventListener("change",()=>{ if(S.go){ S.go.cfg.lib=$("#g-lib").value; } });
$("#g-gear").addEventListener("click",()=>{ const p=$("#g-plan"); p.hidden=!p.hidden; $("#g-gear").setAttribute("aria-expanded",String(!p.hidden)); });
document.querySelectorAll("#go-modes .mode").forEach(b=>b.addEventListener("click",()=>{ const m=b.dataset.m; document.querySelectorAll("#go-modes .mode").forEach(x=>x.classList.toggle("on",x===b)); $("#go-gen").hidden=m!=="gen"; $("#go-score").hidden=m!=="score"; }));
/* ---------- per-base report ---------- */
S.rep=new Map(); let _ptLib=null; function ptLib(){ return _ptLib||(_ptLib=buildLib("ptprs")); }
function onReport(m){ const cb=S.rep.get(m.id); if(cb){ S.rep.delete(m.id); cb(m.reports[0]); } }
function reqReport(seq,cb){ const id=++S.job; const lib=S.lib&&S.lib.kind==="ptprs"?S.lib:ptLib(); S.rep.set(id,cb); S.worker.postMessage({cmd:"report",id,seqs:[seq],lib:lib?lib.wins:null,libKey:lib?lib.key:null}); }
function repHtml(r){
  const NT="ACGT", s=r.seq, cw=18, x0=34, H=14, gap=5;
  const jz=E.dnaJudge(s,S.judges.dna); const jism=[]; for(let i=0;i<30;i++){ const row=[]; for(let b=0;b<4;b++) row.push(NT[b]===s[i]?jz:E.dnaJudge(s.slice(0,i)+NT[b]+s.slice(i+1),S.judges.dna)); jism.push(row); }
  const am=row=>row.indexOf(Math.max(...row)); const c0=sig(r.crit.mean), j0=sig(jz); let dis=0, mat=0; const dpos=[]; for(let i=0;i<30;i++){ const gc=sig(Math.max(...r.ism[i]))-c0, gj=sig(Math.max(...jism[i]))-j0; if(gc>=0.03||gj>=0.03){ mat++; if(am(r.ism[i])!==am(jism[i])){ dis++; dpos.push(i); } } }
  const cur=[sig(r.crit.mean),sig(jz)];
  const heat=(rows,base,y0,name)=>{ let g='<text x="0" y="'+(y0+H*2+3)+'" font-size="9" fill="currentColor" transform="rotate(-90 8 '+(y0+H*2)+')" text-anchor="middle">'+name+'</text>';
    for(let b=0;b<4;b++){ g+='<text x="'+(x0-6)+'" y="'+(y0+b*H+H-3)+'" font-size="9" text-anchor="end" fill="currentColor">'+NT[b]+'</text>';
      for(let i=0;i<30;i++){ const d=sig(rows[i][b])-base, isCur=NT[b]===s[i], al=Math.min(1,Math.abs(d)/0.3);
        const fill=isCur?"var(--cellcur)":(d>=0?"rgba(31,111,165,"+(0.08+0.85*al).toFixed(2)+")":"rgba(201,92,30,"+(0.08+0.85*al).toFixed(2)+")");
        g+='<rect x="'+(x0+i*cw)+'" y="'+(y0+b*H)+'" width="'+(cw-1)+'" height="'+(H-1)+'" rx="2" style="fill:'+fill+'"><title>'+name+': position '+(i+1)+' '+s[i]+'→'+NT[b]+' changes P by '+(isCur?"0 (current)":(d>=0?"+":"")+d.toFixed(2))+'</title></rect>'; } }
    return g; };
  let svg='<svg viewBox="0 0 '+(x0+30*cw)+' 196" role="img" aria-label="Per-base report">';
  for(let i=0;i<30;i++){ svg+='<text x="'+(x0+i*cw+cw/2-.5)+'" y="9" font-size="10" text-anchor="middle" font-family="ui-monospace,Menlo,monospace" fill="'+(i>=24&&i<27?"#1F6FA5":"#14305A")+'" font-weight="'+(i>=24&&i<27?700:500)+'">'+s[i]+'</text>'; }
  svg+=heat(r.ism,cur[0],14,"critic")+heat(jism,cur[1],14+4*H+gap+4,"judge");
  const ty=14+8*H+gap*2+14, mx=Math.max(1.6,...r.surp.pos); svg+='<text x="0" y="'+(ty+22)+'" font-size="9" fill="currentColor" transform="rotate(-90 8 '+(ty+22)+')" text-anchor="middle">surprise</text>';
  r.surp.pos.forEach((v,i)=>{ const h=Math.max(1,v/mx*40); svg+='<rect x="'+(x0+i*cw)+'" y="'+(ty+44-h)+'" width="'+(cw-1)+'" height="'+h+'" rx="2" fill="#7A8CA8"><title>prior surprise at position '+(i+1)+': '+v.toFixed(2)+'</title></rect>'; });
  for(const i of dpos) svg+='<rect x="'+(x0+i*cw)+'" y="'+(ty+47)+'" width="'+(cw-1)+'" height="3" fill="#C95C1E"/>';
  svg+='</svg>';
  const ann=(S.lib&&S.lib.map.get(s))||(ptLib()&&ptLib().map.get(s)); const sp=r.pct==null?"":" Lower surprise than "+Math.round((1-r.pct)*100)+"% of PTPRS sites.";
  return '<div class="rep"><h5>Per-base report</h5>'+svg+'<div class="rk"><span><i style="background:rgba(31,111,165,.8)"></i>mutation raises P(efficient)</span><span><i style="background:rgba(201,92,30,.8)"></i>lowers it</span><span><i class="cb" style="background:var(--cellcur)"></i>current base</span><span><i style="background:#C95C1E"></i>critic and judge pick different best bases</span></div>'+
   '<p style="margin:8px 0 0"><b>Critic P '+cur[0].toFixed(2)+' · judge P '+cur[1].toFixed(2)+'.</b> Each cell is the change in predicted efficiency if that one base were swapped (in-silico mutagenesis). At the <b>'+mat+'</b> positions where either model sees a gain of 0.03 or more, they disagree on the best base at <b>'+dis+'</b>.</p>'+
   '<p style="margin:4px 0 0">Prior surprise (denoising error under the coding-sequence prior, our analogue of Evo\'s per-base entropy): mean '+r.surp.mean.toFixed(2)+'; a random sequence is about 1.2.'+sp+'</p>'+
   '<p style="margin:4px 0 0">'+(ann?"Annotation: "+esc(ann.label)+", "+(ann.strand>0?"sense":"antisense")+" strand, PAM at positions 25 to 27, cut between positions 21 and 22 of the 30-mer.":"Annotation: not an exact NGG cut site inside a PTPRS exon.")+'</p></div>'; }
document.addEventListener("click",e=>{ const b=e.target.closest(".repbtn"); if(!b) return; const out=b.parentNode.querySelector(".rep-out"); if(out.dataset.on==="1"){ out.innerHTML=""; out.dataset.on="0"; b.textContent="Per-base report"; return; } b.textContent="Computing…"; reqReport(b.dataset.seq,r=>{ out.innerHTML=repHtml(r); out.dataset.on="1"; b.textContent="Hide per-base report"; }); });
/* ---------- Score a candidate ---------- */
function scKind(raw){ const t=raw.trim(), d=t.replace(/\s+/g,"").toUpperCase(); if(d.length>=20&&/^[ACGT]+$/.test(d)&&new Set(d).size>=3) return {kind:"dna",dna:d}; if(t&&!/\s/.test(t)) return {kind:"smiles",smi:t}; return {kind:"none"}; }
function scUpdate(){ $("#sc-status").textContent=""; const raw=$("#sc-in").value, k=scKind(raw); const n=raw.replace(/\s+/g,"").length;
  $("#sc-live").textContent=!n?"0 letters":k.kind==="dna"?(k.dna.length+" bases"+(k.dna.length===30?" · one guide context":k.dna.length>500?" · only the first 500 are used":" · every NGG site is scored")):k.kind==="smiles"?n+" characters · read as a SMILES":n+" characters · paste DNA (20+ bases, A/C/G/T) or one SMILES with no spaces";
  $("#sc-go").disabled=!S.ready||k.kind==="none"; }
$("#sc-in").addEventListener("input",scUpdate);
function scInit(){ const ex=[]; const L=ptLib(); if(L){ const w=L.wins[Math.floor(L.wins.length/3)]; ex.push(["A PTPRS cut site (30-mer)",w]); } if(S.ptprs&&S.ptprs.exons[4]) ex.push(["PTPRS exon 5 stretch (DNA)",S.ptprs.exons[4].seq.slice(0,160)]); ex.push(["Melatonin (SMILES)","COc1ccc2[nH]cc(CCNC(C)=O)c2c1"]); ex.push(["Aspirin (SMILES)","CC(=O)Oc1ccccc1C(=O)O"]);
  $("#sc-chips").innerHTML=ex.map(([l,v])=>'<button type="button" data-p="'+esc(v)+'">'+esc(l)+'</button>').join(""); scUpdate(); }
$("#sc-chips").addEventListener("click",e=>{ const b=e.target.closest("button"); if(!b) return; $("#sc-in").value=b.dataset.p; scUpdate(); $("#sc-go").click(); });
$("#sc-go").addEventListener("click",()=>{ const k=scKind($("#sc-in").value); if(k.kind==="none") return; $("#sc-out").innerHTML="";
  if(k.kind==="smiles"){ const a=analyse(k.smi); if(!a){ $("#sc-status").textContent="That is not a valid SMILES string."; return; } const j=judgeMol(a), jp={bbb:sig(j.bbb),act:sig(j.act),tox:sig(j.tox)}, nn=nearest(a), pass=j.bbb>0&&j.act>0&&j.tox<0, cns=a.tpsa<90&&a.mw<450&&a.hbd<=3;
    $("#sc-status").textContent="Scored on your device."; $("#sc-out").innerHTML='<div class="card" style="max-width:420px"><div class="sv">'+a.svg+'</div><div class="smi" data-smi="'+esc(a.canon)+'">'+esc(a.canon)+'</div><div>'+(pass?'<span class="tag ok">judge: passes all three</span>':'<span class="tag no">judge: fails at least one</span>')+(cns?'<span class="tag ok">CNS-like physchem</span>':'<span class="tag">outside CNS physchem</span>')+(a.ro5>1||a.mw>600||a.mw<120||a.logp>6?'<span class="tag no">not drug-like</span>':'')+'</div>'+pair("brain",jp.bbb,null,true)+pair("activity",jp.act,null,true)+pair("low tox",jp.tox,null,false)+'<div class="small" style="font-size:11.5px">MW '+fmt(a.mw,0)+' · cLogP '+fmt(a.logp,1)+' · TPSA '+fmt(a.tpsa,0)+' · HBD '+fmt(a.hbd,0)+' · Ro5 violations '+a.ro5+'</div><div class="small" style="font-size:11.5px">Nearest known compound: Tanimoto '+fmt(nn.t)+(nn.t>=0.99?' (known)':nn.t<0.4?' (new scaffold)':'')+'</div></div><p class="small" style="margin-top:10px">Judge scores only. The steering critics read the generator\'s latent space and this page has no SMILES encoder, so critic scores and per-atom effects are not available for a pasted molecule. Guides get the full per-base report.</p>'; return; }
  let seq=k.dna.slice(0,500), rows=[]; if(seq.length===30) rows=[{win:seq,label:"your 30-mer",strand:1,pos:0}]; else if(seq.length<30){ $("#sc-status").textContent="Need at least 30 bases for a guide context."; return; } else{ const raw=[]; E.findSites(seq,"your DNA",0,seq.length,raw); const seen=new Set(); rows=raw.filter(r=>{ const kk=r.win; if(seen.has(kk)) return false; seen.add(kk); return true; }); }
  if(!rows.length){ $("#sc-status").textContent="No NGG site with 30 bases of context found in that DNA."; return; }
  S.lib=ptLib(); S.sc={id:++S.job,rows,len:seq.length}; $("#sc-go").disabled=true; $("#sc-status").textContent="Scoring "+rows.length+" site"+(rows.length>1?"s":"")+"…"; S.worker.postMessage({cmd:"scoreDna",id:S.sc.id,seqs:rows.map(r=>r.win)}); });
function onScScores(m){ const R=S.sc.rows; $("#sc-go").disabled=false; const O=R.map((r,i)=>{ const ck=dnaChecks(r.win), jz=E.dnaJudge(r.win,S.judges.dna); return {seq:r.win,ck,cp:sig(m.scores[i][0]),csd:m.scores[i][1],jp:sig(jz),jz,pass:ck.usable&&jz>0,hit:S.lib.map.get(r.win)||null,label:r.label,strand:r.strand}; }); O.sort((a,b)=>(b.cp+b.jp)-(a.cp+a.jp));
  const top=O.slice(0,R.length===1?1:8), gap=O.reduce((s,o)=>s+Math.abs(o.cp-o.jp),0)/O.length;
  $("#sc-status").textContent="Done. Scored "+O.length+" site"+(O.length>1?"s":"")+" on your device.";
  $("#sc-out").innerHTML='<div class="metrics"><div><span>Sites scored</span><b>'+O.length+'</b><small>NGG with 30-nt context, both strands</small></div><div><span>Pass judge and checks</span><b>'+O.filter(o=>o.pass).length+'</b><small>of '+O.length+'</small></div><div><span>Best critic / judge</span><b>'+fmt(top[0].cp)+' / '+fmt(top[0].jp)+'</b><small>top-ranked site</small></div><div><span>Mean critic&ndash;judge gap</span><b>'+fmt(gap)+'</b><small>how far the two models differ</small></div></div>'+(R.length>1?'<p class="small">Top '+top.length+' by average of critic and judge. Open any row for its per-base report.</p>':'')+'<div class="dna">'+top.map((o,i)=>dnaRow(o,R.length>1?"Rank "+(i+1)+" · "+(o.strand>0?"sense":"antisense")+" strand":"",false)).join("")+'</div>'; if(O.length===1||top.length===1){ const b=$("#sc-out .repbtn"); if(b) b.click(); } }

/* ---------- header that recedes on scroll, theme toggle ---------- */
(function(){ const hd=document.querySelector(".top"), root=document.documentElement; let last=window.scrollY, tick=false;
  function upd(){ const y=window.scrollY, d=y-last; if(y<80){ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); } else if(d>6){ hd.classList.add("hide"); root.style.setProperty("--hh","8px"); } else if(d<-6){ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); } if(Math.abs(d)>6||y<80) last=y; tick=false; }
  window.addEventListener("scroll",()=>{ if(!tick){ tick=true; requestAnimationFrame(upd); } },{passive:true});
  hd.addEventListener("focusin",()=>{ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); });
  const btn=document.getElementById("theme"); function paint(){ const dark=root.dataset.theme==="dark"; btn.setAttribute("aria-pressed",String(dark)); btn.title=dark?"Switch to light theme":"Switch to dark theme"; const m=document.querySelector('meta[name=theme-color]'); if(m) m.content=dark?"#0B1020":"#14305A"; }
  btn.addEventListener("click",()=>{ const dark=root.dataset.theme!=="dark"; root.dataset.theme=dark?"dark":"light"; try{ localStorage.setItem("ptp-theme",root.dataset.theme); }catch(e){} paint(); }); paint(); })();
boot();
})();

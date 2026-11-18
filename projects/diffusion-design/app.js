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
    S.rd=rd; S.judges=judges; S.ref=ref.map(([s,bits])=>({s,bits:new Set(bits)})); S.ptprs=ptprs; S.pref=null; getJSON("models/ptprs_ref.json").then(P=>{ S.pref={meta:P,rows:P.smiles.map((s,i)=>{ const m=rd.get_mol(s); if(!m) return null; const fp=m.get_morgan_fp(JSON.stringify({radius:2,nBits:1024})); m.delete(); const on=new Set(); for(let k=0;k<fp.length;k++) if(fp.charCodeAt(k)===49) on.add(k); return {y:P.y[i],bits:on}; }).filter(Boolean)}; }).catch(()=>{});
    S.worker=new Worker(BASE+"worker.js"); S.worker.onmessage=onWorker; S.worker.onerror=e=>fail("Worker error: "+e.message);
    S.worker.postMessage({cmd:"init"});
  }catch(err){ fail("Could not load the models. "+(err&&err.message||err)+" Reload the page; if it persists, open orvillerockwelljr.com/projects/diffusion-design/ directly."); }
}
function fail(msg){ $("#loadmsg").innerHTML='<div class="err">'+esc(msg)+'</div>'; }
function onWorker(ev){
  const m=ev.data;
  if(m.evt==="prog"){ $("#loadmsg").innerHTML='<span class="ld"><i style="width:'+m.pct+'%"></i></span> Loading models on your device '+m.pct+'%'; return; }
  if(m.evt==="ready"){ S.ready=true; $("#loadmsg").textContent="Models loaded ("+m.info.molHeads+"-head critic ensembles; runs on your device)."; ["m-go","d-go","s-go","g-read"].forEach(i=>$("#"+i).disabled=false); scInit(); if($("#g-prompt").value.trim()) doRead(); setTimeout(()=>{ if(S.ready) $("#loadmsg").style.display="none"; },2500); }
  else if(m.evt==="trace") onTrace(m);
  else if(m.evt==="item") onItem(m);
  else if(m.evt==="done") onDone(m);
  else if(m.evt==="scores"){ if(S.sc&&S.sc.id===m.id) onScScores(m); else onScores(m); }
  else if(m.evt==="report") onReport(m);
  else if(m.evt==="molScores") onMolScores(m);
  else if(m.evt==="seedCheck"){ const r=S.go&&S.go.run, si=r&&r.seedInfo; if(si&&S.cur&&S.cur.job.id===m.id){ const ra=analyse(m.recon); si.recon=m.recon; si.fid=ra?tani(si.a,ra):0; si.exact=ra&&ra.canon===si.smiles; } }
  else if(m.evt==="seedError"){ $("#g-status").textContent="Cannot use that seed molecule: "+m.why+"."; $("#g-make-go").disabled=false; $("#g-stop").hidden=true; if(S.go&&S.go.run) S.go.run.stopped=true; }
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
function ptpRef(a){ if(!S.pref) return null; let ma=0,mi=0; for(const r of S.pref.rows){ let inter=0; for(const b of a.fp) if(r.bits.has(b)) inter++; const t=inter/(a.fp.length+r.bits.size-inter); if(r.y){ if(t>ma) ma=t; } else if(t>mi) mi=t; } return {act:ma,inact:mi,d:ma-mi}; }
function ptpRefHtml(a){ const r=ptpRef(a); if(!r) return ""; const lean=r.d>0.05?"leans active":r.d<-0.05?"leans inactive":"no clear lean"; return '<div class="small" style="font-size:11.5px" title="Closest PTPσ-tested molecule in ChEMBL (60 molecules). Held out by chemical series this rule scores AUC 0.73 (95% CI 0.59-0.86): weak evidence, not a validated prediction.">PTPσ reference: closest active '+fmt(r.act)+', closest inactive '+fmt(r.inact)+' · '+lean+'</div>'; }
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
function sane(a){ return a.tpsa>=10&&/[0-9]/.test(a.canon)&&!/C{8,}/.test(a.canon); }
function procMol(r){
  let a=analyse(r.smiles), used=r.smiles, repaired=false; if(!a&&r.alts){ for(const alt of r.alts){ const aa=analyse(alt); if(aa){ a=aa; used=alt; repaired=true; break; } } } const o={smiles:used,valid:!!a,crit:r.crit,repaired};
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
   (o.pass?'<span class="tag ok">judge: passes all three</span>':'<span class="tag no">judge: fails ≥1</span>')+auditTag(auditMol(o))+(o.repaired?'<span class="tag" title="The first decode was not a valid molecule; a tiny perturbation of the same latent code decoded to this one.">repaired decode</span>':'')+(cns?'<span class="tag ok">CNS-like physchem</span>':'<span class="tag">outside CNS physchem</span>')+'</div>'+
   pair("brain",o.cp.bbb,o.jp.bbb,true)+pair("activity",o.cp.act,o.jp.act,true)+pair("low tox",o.cp.tox,o.jp.tox,false)+
   '<div class="small" style="font-size:11.5px">MW '+fmt(a.mw,0)+' · cLogP '+fmt(a.logp,1)+' · TPSA '+fmt(a.tpsa,0)+' · HBD '+fmt(a.hbd,0)+' · Ro5 violations '+a.ro5+'</div>'+
   '<div class="cact"><button type="button" class="chip" data-seedsmi="'+esc(a.canon)+'">Design analogs</button><button type="button" class="chip" data-scoresmi="'+esc(a.canon)+'">Score with critics</button></div>'+'<div class="small" style="font-size:11.5px">Nearest known compound: Tanimoto '+fmt(o.nn.t)+(o.nn.t>=0.99?' (known)':o.nn.t<0.4?' (new scaffold)':'')+'</div>'+ptpRefHtml(o.a)+(o.simSeed!=null?'<div class="small" style="font-size:11.5px">Similarity to your seed: Tanimoto '+fmt(o.simSeed)+(o.simSeed>=0.99?' (same molecule)':'')+'</div>':'')+'</div>';
}
function tani(a,b){ let i=0; for(const x of a.fp) if(b.fpset.has(x)) i++; return i/(a.fp.length+b.fp.length-i||1); }
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
function siteTag(o){ const F=window.PTPFusion; if(!F||!F.ok||!o.hit||!/PTPRS/.test(o.hit.label||"")) return ""; const m=/exon (\d+)/.exec(o.hit.label), si=F.siteInfo(o.seq,m?+m[1]:0); if(!si||!si.coding) return si?'<span class="tag">'+esc(si.region)+'</span>':""; return '<span class="tag '+(si.catalytic||si.aa<1393?'ok':'')+'" title="'+esc(si.text)+'">aa '+si.aa+' · '+esc(si.domain)+'</span>'; }
function dnaRow(o,label,un){
  const ck=o.ck; return '<div class="row" style="'+(un?"background:#fafbfc":"")+'"><div>'+(label?'<div style="font-weight:600;color:var(--navy);margin-bottom:4px">'+label+'</div>':'')+seqHtml(o.seq)+'<div class="seqkey">'+o.seq.slice(4,24)+' + '+o.seq.slice(24,27)+' (protospacer + PAM)</div></div><div>'+
   (o.pass?'<span class="tag ok">judge: efficient + passes checks</span>':'<span class="tag no">fails judge or checks</span>')+auditTag(auditDna(o))+(ck.pam?'<span class="tag ok">NGG PAM</span>':'<span class="tag no">no NGG PAM</span>')+'<span class="tag '+(ck.gc>=.4&&ck.gc<=.7?'ok':'no')+'">GC '+Math.round(ck.gc*100)+'%</span>'+(ck.t4?'<span class="tag no">TTTT run</span>':'')+(S.lib?(o.hit?'<span class="tag ok">exact site: '+esc(o.hit.label)+', '+(o.hit.strand>0?'+':'−')+' strand</span>'+siteTag(o):'<span class="tag no">not a site in '+esc(S.lib.label)+'</span>'):'')+
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
  const raw=$("#g-prompt").value.trim(); if(!raw){ $("#g-live").textContent=S.go&&S.go.run&&!S.go.run.finished?"Working…":"Waiting for your request."; $("#g-make-go").disabled=true; if(!(S.go&&S.go.run&&!S.go.run.finished)) S.go=null; return; } $("#g-readmsg").textContent="";
  const hd=attachDna(); const {cfg,notes}=readPrompt(raw+(hd?" "+hd:"")); if(S.files.length) notes.push("Using "+S.files.length+" attached file"+(S.files.length>1?"s":"")+" as context"+(attachSeed()?"; first molecule found is the seed for analogs":"")+(hd?"; DNA found is the guide target":"")+"."); const prevRun=S.go&&S.go.run; S.go={cfg,raw}; if(prevRun&&!prevRun.finished) S.go.run=prevRun; $("#g-make").value=cfg.make; $("#g-n").value=String(cfg.n); $("#g-push").value=cfg.push; $("#g-lib").value=cfg.lib; $("#g-wb").value=String(cfg.w[0]); $("#g-wa").value=String(cfg.w[1]); $("#g-wt").value=String(cfg.w[2]);
  $("#g-notes").innerHTML=notes.map(n=>"<li>"+esc(n)+"</li>").join(""); $("#g-make-go").disabled=!S.ready; liveLine(cfg);
}
$("#g-make-go").addEventListener("click",()=>{
  if(!S.go||S.go.raw!==$("#g-prompt").value.trim()) doRead(); const g=S.go; if(!g) return; const make=$("#g-make").value, n=+$("#g-n").value, push=$("#g-push").value, libk=$("#g-lib").value;
  const w=[+$("#g-wb").value,+$("#g-wa").value,+$("#g-wt").value]; const seed=+$("#g-seed").value||0;
  const lib=make!=="mol"?buildLib(libk,g.cfg.pasted):null; if(make!=="mol"&&libk==="paste"&&!lib){ $("#g-status").textContent="No pasted DNA found in your request (need 40+ letters)."; return; }
  if(make!=="dna"&&w.every(v=>v===0)){ $("#g-status").textContent="All three molecule objectives are off; switch one on."; return; }
  S.lib=lib; const gam={mol:{gentle:2,std:4,hard:8},dna:{gentle:2,std:4,hard:12}};
  const seedSmi=$("#g-smi").value.trim()||(make!=="dna"?attachSeed():""); let seedInfo=null;
  if(seedSmi&&make!=="dna"){ const sa=analyse(seedSmi); if(!sa){ $("#g-status").textContent="The seed molecule is not a valid SMILES string."; return; } seedInfo={smiles:sa.canon,a:sa,strength:+$("#g-str").value}; }
  const stages=[]; if(make!=="dna") stages.push({kind:"mol",job:seedInfo?{cmd:"genAnalog",id:0,n:Math.min(48,n*2),seed,steps:60,gamma:gam.mol[push],w,gate:$("#g-gate").value==="on",tau:1,smiles:seedInfo.smiles,strength:seedInfo.strength}:{cmd:"genMol",id:0,n:Math.min(48,n*2),seed,steps:60,gamma:gam.mol[push],w,gate:$("#g-gate").value==="on",tau:1,compare:false}});
  if(make!=="mol") stages.push({kind:"dna",job:dnaJob({n,seed:seed+1,steps:50,gamma:gam.dna[push],gate:false,tau:1,compare:false,wl:1,tilt:2},lib)});
  S.go.run={seedInfo,photo:S.photo||null,stages,i:0,make,n,mol:[],dna:[],lib,libk,use:g.cfg.use,push,w,t0:performance.now(),stopped:false}; $("#g-out").innerHTML=""; $("#g-make-go").disabled=true; $("#g-stop").hidden=false; $("#g-bar").style.width="0"; startStage(); clearComposer();
});
function clearComposer(){ const P=$("#g-prompt"); P.value=""; $("#g-smi").value=""; S.photo=null; S.files=[]; const att=$("#g-att"); att.hidden=true; att.innerHTML=""; $("#g-img").innerHTML=""; $("#g-notes").innerHTML=""; const pl=$("#g-plan"); if(pl&&!pl.hidden){ $("#g-gear").click(); } $("#g-live").textContent="Ask a follow-up, or describe something new."; $("#g-make-go").disabled=true; }
$("#g-stop").addEventListener("click",()=>{ const r=S.go&&S.go.run; if(!r) return; r.stopped=true; if(S.cur) S.worker.postMessage({cmd:"cancel",id:S.cur.job.id}); });
function startStage(){
  const r=S.go.run, st=r.stages[r.i]; st.job.id=++S.job; S.cur={kind:st.kind,job:st.job,got:0,t0:performance.now(),chain:true};
  if(st.kind==="mol") S.mol={items:[],cmp:false,job:st.job}; else S.dna={items:[],job:st.job,cmp:false};
  $("#g-status").textContent="Stage "+(r.i+1)+" of "+r.stages.length+": "+(st.kind==="mol"?"molecules":"guides")+"…"; S.worker.postMessage(st.job);
}
function chainItem(m){ const r=S.go&&S.go.run; if(!r) return; let done=S.cur.got, tot=0; r.stages.forEach((s,k)=>{ tot+=s.job.n; if(k<r.i) done+=s.job.n; }); $("#g-bar").style.width=(done/tot*100)+"%"; }
function chainDone(m){
  const r=S.go.run, st=r.stages[r.i]; if(st.kind==="mol"){ r.mol=S.mol.items.map(x=>x.g); if(r.seedInfo) r.mol.forEach(o=>{ if(o.valid) o.simSeed=tani(o.a,r.seedInfo.a); }); } else r.dna=S.dna.items.map(x=>x.g);
  r.i++; if(r.i<r.stages.length&&!m.cancelled&&!r.stopped){ startStage(); return; }
  $("#g-make-go").disabled=!$("#g-prompt").value.trim(); $("#g-stop").hidden=true; $("#g-bar").style.width="100%"; const sec=(performance.now()-r.t0)/1000; $("#g-status").textContent=(r.stopped?"Stopped. ":"Done. ")+"Took "+sec.toFixed(1)+" s on your device."; r.finished=true; renderShortlist();
}
/* ---------- Two-key pairs: a reversible molecule and a permanent guide designed as one matched experiment ---------- */
function trustW(k){ return k==="ok"?1:k==="mid"?0.7:0.3; }
function molScore(o){ const m=(o.jp.bbb+o.jp.act+(1-o.jp.tox))/3; return m*(o.pass?1:0.5)*trustW(auditMol(o)); }
function guideScore(o){ const F=window.PTPFusion; let site=1, si=null; if(o.hit&&F&&F.ok&&/PTPRS/.test(o.hit.label||"")){ const m=/exon (\d+)/.exec(o.hit.label); si=F.siteInfo(o.seq,m?+m[1]:0); if(si&&!si.coding) site=0.2; }
  return {s:o.jp*(o.ck.usable?1:0.5)*trustW(auditDna(o))*site*(o.hit?1:0.6), si}; }
function buildPairs(mols,dnas,K){
  const ms=mols.filter(o=>o.valid).map(o=>({o,s:molScore(o)})).sort((a,b)=>b.s-a.s).slice(0,8);
  const seen=new Set(), gs=dnas.filter(o=>{ if(seen.has(o.seq)) return false; seen.add(o.seq); return true; }).map(o=>{ const g=guideScore(o); return {o,s:g.s,si:g.si}; }).sort((a,b)=>b.s-a.s).slice(0,8);
  const all=[]; ms.forEach(m=>gs.forEach(g=>all.push({m,g,score:Math.sqrt(m.s*g.s)}))); all.sort((a,b)=>b.score-a.score);
  const out=[], um=new Set(), ug=new Set(); for(const p of all){ if(um.has(p.m.o.a.canon)||ug.has(p.g.o.seq)) continue; um.add(p.m.o.a.canon); ug.add(p.g.o.seq); out.push(p); if(out.length>=K) break; }
  return out;
}
function pairsHtml(pairs){
  if(!pairs.length) return "";
  const card=(p,i)=>{ const o=p.m.o, g=p.g.o, si=p.g.si;
    return '<article class="pr"><header><b>Pair '+(i+1)+'</b><span class="tag '+(p.score>=0.6?'ok':'')+'" title="Geometric mean of the molecule score and the guide score">pair score '+p.score.toFixed(2)+'</span></header>'+
     '<div class="pr2"><div class="k"><small>KEY 1 · reversible molecule</small><div class="sv">'+o.a.svg+'</div><div class="smi" data-smi="'+esc(o.a.canon)+'">'+esc(o.a.canon)+'</div><div>'+(o.pass?'<span class="tag ok">judge passes</span>':'<span class="tag no">judge fails ≥1</span>')+auditTag(auditMol(o))+'</div></div>'+
     '<div class="k"><small>KEY 2 · permanent guide (PTPRS knockout)</small>'+seqHtml(g.seq)+'<div class="seqkey">'+g.seq.slice(4,24)+' + '+g.seq.slice(24,27)+'</div><div>'+(g.pass?'<span class="tag ok">judge: efficient + passes checks</span>':'<span class="tag no">fails judge or checks</span>')+auditTag(auditDna(g))+'</div>'+
     '<p class="small" style="margin-top:6px">'+(si?esc(si.text)+'.':(g.hit?'Exact PTPRS site.':'Not mapped to PTPRS.'))+'</p></div></div>'+
     '<details class="armsd"><summary>Run it as 4 arms</summary><ol class="arms"><li><b>Vehicle + non-targeting guide</b> baseline</li><li><b>Molecule alone</b> does it do what you want?</li><li><b>Guide alone</b> the genetic answer for PTPσ loss</li><li><b>Molecule in knockout cells</b> if it still acts here, the effect is off-target</li></ol></details></article>'; };
  return '<section class="twokey"><h3>Two-key pairs</h3><p class="small">Each pair is a matched experiment: a reversible molecule plus a permanent PTP&sigma; knockout. If the molecule works through PTP&sigma;, its effect should match the knockout and vanish in knockout cells. A molecule that still acts there is off-target.</p><div class="pairs">'+pairs.map(card).join("")+'</div>'+
   '<details class="how"><summary>How pairs are ranked</summary><p class="small">Pair score = square root of (molecule score &times; guide score). Molecule score = mean of the judge&rsquo;s brain, activity and low-toxicity probabilities, halved if the judge fails it, times an audit-trust weight (trusted 1, disputed 0.7, likely exploited 0.3). Guide score = the judge&rsquo;s efficiency probability, halved if it fails the PAM, GC or run checks, times the same trust weight, times 0.2 if the cut is outside coding sequence and 0.6 if it is not an exact PTPRS site. This is a ranking rule I defined for choosing which pairs to test first. It is not a validated predictor of therapeutic benefit.</p></details></section>';
}
function renderShortlist(){
  const r=S.go.run, K=6; let html='<h3 class="h3s">Shortlist'+(r.use?' for: <i>'+esc(r.use.length>150?r.use.slice(0,150)+"…":r.use)+'</i>':'')+'</h3>';
  const parts=[];
  if(r.make!=="dna"){ const v0=r.mol.filter(o=>o.valid); const v=v0.filter(o=>o.a.ro5<=1&&o.a.mw<=600&&o.a.logp<=6&&o.a.mw>=120&&sane(o.a)&&!(r.seedInfo&&o.simSeed>=0.99)); const dropped=v0.length-v.length; const pass=v.filter(o=>o.pass); const ranked=v.slice().sort((a,b)=>(b.jz.bbb*r.w[0]+b.jz.act*r.w[1]-b.jz.tox*r.w[2])-(a.jz.bbb*r.w[0]+a.jz.act*r.w[1]-a.jz.tox*r.w[2])).slice(0,K);
    parts.push('<div><h3>Small molecules</h3><details class="how"><summary>'+v0.length+' of '+r.mol.length+' decoded to valid molecules, '+pass.length+' of '+v.length+' pass the held-out judge</summary><p class="small" style="margin-bottom:8px">'+v0.length+' of '+r.mol.length+' decoded to a valid molecule'+(dropped?'; '+dropped+' removed by a basic drug-likeness filter (MW 120 to 600, cLogP &le; 6, at most one rule-of-five violation, at least one ring and one polar atom, no unbranched chain of 8 or more carbons), because a steered generator can find structures the judge likes but no chemist would make':'')+'. '+pass.length+' of the rest pass the held-out judge on all three objectives. Generated '+r.mol.length+' (fewer than half decode, so the run draws twice the number you asked for). Showing the top '+ranked.length+' by your weights.</p></details><p class="small legend"><i class="lg c"></i>critic (steers the model) <i class="lg j"></i>held-out judge (audit)</p><div class="cards">'+(ranked.length?ranked.map(o=>molCard(o,0,false)).join(""):'<p class="small">No valid molecules this run. Try a different seed or a gentler push.</p>')+'</div></div>'); }
  if(r.make!=="mol"){ const g=r.dna; const on=g.filter(o=>o.hit), onp=g.filter(o=>o.onpass), pass=g.filter(o=>o.pass);
    const ranked=g.slice().sort((a,b)=>(+!!b.hit-+!!a.hit)||(b.jz-a.jz)).filter((o,i,arr)=>arr.findIndex(x=>x.seq===o.seq)===i).slice(0,K);
    const head=r.lib?(on.length+' of '+g.length+' are exact cut sites in '+r.lib.label+'; '+onp.length+' of those also pass the judge and the checks ('+new Set(on.map(o=>o.seq)).size+' distinct sites).'):(pass.length+' of '+g.length+' pass the judge and the checks (no locus was set).');
    parts.push('<div><h3>Gene-editing guides</h3><p class="small" style="margin-bottom:8px">'+head+' Showing the top '+ranked.length+' distinct.</p><div class="dna">'+ranked.map(o=>dnaRow(o,"",false)).join("")+'</div></div>'); }
  const gdna=r.make!=="mol"?r.dna:[], gmol=r.make!=="dna"?r.mol.filter(o=>o.valid):[]; const gaps=[...gdna.map(o=>Math.abs(o.cp-o.jp)),...gmol.flatMap(o=>[Math.abs(o.cp.bbb-o.jp.bbb),Math.abs(o.cp.act-o.jp.act),Math.abs(o.cp.tox-o.jp.tox)])]; const gapm=gaps.length?gaps.reduce((s,v)=>s+v,0)/gaps.length:NaN; const nflag=gmol.filter(o=>auditMol(o)==="hack").length+gdna.filter(o=>auditDna(o)==="hack").length; const nreq=r.n*(r.make==="both"?2:1), ngen=(r.make!=="dna"?r.mol.length:0)+gdna.length, passN=gmol.filter(o=>o.pass).length+gdna.filter(o=>o.pass).length, onT=gdna.filter(o=>o.hit).length;
  if(r.seedInfo) html+='<p class="small">Analog design from your seed <span class="smi" data-smi="'+esc(r.seedInfo.smiles)+'">'+esc(r.seedInfo.smiles)+'</span>: the seed was encoded into the model\'s latent space, noised to '+Math.round(r.seedInfo.strength*100)+'% of the schedule, then denoised with the same critic guidance and decoded. Lower distance keeps more of the seed.</p>'+(r.seedInfo.exact?'<p class="small">Round-trip check: the model encodes and decodes your seed back to the same molecule, so analogs are anchored to it.</p>':'<p class="small"><b>Round-trip check failed:</b> the model decodes your seed as '+(r.seedInfo.recon?'<span class="smi" data-smi="'+esc(r.seedInfo.recon)+'">'+esc(r.seedInfo.recon)+'</span> (similarity '+fmt(r.seedInfo.fid||0)+')':'nothing valid')+'. It is outside what this small model represents well, so treat the results as loosely inspired by it, not as close analogs.</p>');
  if(r.photo) html+='<p class="small">'+(r.photo.thumb?'<img class="thumb" alt="" src="'+r.photo.thumb+'"> ':'')+'Used as context: <b>'+esc(r.photo.name)+'</b>. '+esc(r.photo.foundText||'')+'</p>';
  html+='<div class="metrics"><div><span>Requested</span><b>'+nreq+'</b><small>'+(r.make==="both"?r.n+" molecules + "+r.n+" guides":r.n+(r.make==="mol"?" molecules":" guides"))+'</small></div><div><span>Generated</span><b>'+ngen+'</b><small>'+(r.make!=="dna"?r.mol.length+" molecule draws, "+gmol.length+" valid":"")+(r.make==="both"?", ":"")+(r.make!=="mol"?gdna.length+" guides":"")+'</small></div>'+(r.make!=="mol"&&r.lib?'<div><span>On target</span><b>'+onT+'</b><small>of '+gdna.length+' guides are exact cut sites in '+esc(r.lib.label)+'</small></div>':'')+'<div><span>Pass held-out judge</span><b>'+passN+'</b><small>of '+(gmol.length+gdna.length)+' valid</small></div><div><span>Audit flags</span><b>'+nflag+'</b><small>likely exploited: critic confident, judge not</small></div><div><span>Critic&ndash;judge gap</span><b>'+fmt(gapm)+'</b><small>mean |critic P &minus; judge P|</small></div></div>';
  const stg=r.stages.map(s=>s.job), pm=stg.find(j=>j.cmd==="genMol"||j.cmd==="genAnalog"), pd=stg.find(j=>j.cmd==="genDna");
  html+='<details class="how"><summary>How these were made</summary><p class="small"> Each candidate was denoised from pure noise by a trained diffusion model (cosine schedule, T = 200, DDIM sampling), with the gradient of the critic ensemble added at every step. '+(pm?'Molecules: 64-d latent, '+pm.steps+' steps, guidance &gamma; = '+pm.gamma+', uncertainty gate '+(pm.gate?'on':'off')+'. ':'')+(pd?'Guides: 30-nt one-hot, '+pd.steps+' steps, guidance &gamma; = '+pd.gamma+(pd.lib?', pulled toward PTPRS cut sites':'')+'. ':'')+'Seed '+(pm?pm.seed:pd.seed)+'. Not autoregressive: nothing is generated token by token.</p></details>';
  const pairs=r.make==="both"?buildPairs(r.mol,r.dna,3):[]; html+=pairsHtml(pairs); html+='<div class="twocol">'+parts.join("")+'</div><p class="small" style="margin-top:14px">In-silico predictions; the judge is a model, not an experiment. For a known locus, ranking every site on the Scan tab is the strongest baseline for guides. See the <a href="evidence.html">Evidence page</a>.</p>';
  $("#g-out").innerHTML=html;
  document.dispatchEvent(new CustomEvent("ptp:turn",{detail:{kind:"gen",prompt:S.go.raw||"",use:r.use||"",make:r.make,n:r.n,push:r.push,libk:r.libk,mol:r.make!=="dna"?r.mol:[],dna:r.make!=="mol"?r.dna:[],seed:r.seedInfo?{smiles:r.seedInfo.smiles,strength:r.seedInfo.strength}:null,photo:r.photo?{name:r.photo.name,thumb:r.photo.thumb,kind:r.photo.kindLabel,found:r.photo.foundText}:null,pairs:pairs.map(p=>({smiles:p.m.o.a.canon,guide:p.g.o.seq,score:+p.score.toFixed(3),site:p.g.si?p.g.si.text:""})),html}}));
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
    $("#sc-status").textContent="Scored on your device."; $("#sc-out").innerHTML='<div class="card" style="max-width:420px"><div class="sv">'+a.svg+'</div><div class="smi" data-smi="'+esc(a.canon)+'">'+esc(a.canon)+'</div><div>'+(pass?'<span class="tag ok">judge: passes all three</span>':'<span class="tag no">judge: fails at least one</span>')+(cns?'<span class="tag ok">CNS-like physchem</span>':'<span class="tag">outside CNS physchem</span>')+(a.ro5>1||a.mw>600||a.mw<120||a.logp>6?'<span class="tag no">not drug-like</span>':'')+'</div>'+pair("brain",jp.bbb,null,true)+pair("activity",jp.act,null,true)+pair("low tox",jp.tox,null,false)+'<div class="small" style="font-size:11.5px">MW '+fmt(a.mw,0)+' · cLogP '+fmt(a.logp,1)+' · TPSA '+fmt(a.tpsa,0)+' · HBD '+fmt(a.hbd,0)+' · Ro5 violations '+a.ro5+'</div><div class="small" style="font-size:11.5px">Nearest known compound: Tanimoto '+fmt(nn.t)+(nn.t>=0.99?' (known)':nn.t<0.4?' (new scaffold)':'')+'</div>'+ptpRefHtml(a)+'</div><div id="sc-crit" class="small" style="margin-top:10px">Reading it with the steering critics…</div>'; S.msc={id:++S.job,a,j,jp}; S.worker.postMessage({cmd:"scoreMol",id:S.msc.id,smiles:[a.canon]}); return; }
  let seq=k.dna.slice(0,500), rows=[]; if(seq.length===30) rows=[{win:seq,label:"your 30-mer",strand:1,pos:0}]; else if(seq.length<30){ $("#sc-status").textContent="Need at least 30 bases for a guide context."; return; } else{ const raw=[]; E.findSites(seq,"your DNA",0,seq.length,raw); const seen=new Set(); rows=raw.filter(r=>{ const kk=r.win; if(seen.has(kk)) return false; seen.add(kk); return true; }); }
  if(!rows.length){ $("#sc-status").textContent="No NGG site with 30 bases of context found in that DNA."; return; }
  S.lib=ptLib(); S.sc={id:++S.job,rows,len:seq.length}; $("#sc-go").disabled=true; $("#sc-status").textContent="Scoring "+rows.length+" site"+(rows.length>1?"s":"")+"…"; S.worker.postMessage({cmd:"scoreDna",id:S.sc.id,seqs:rows.map(r=>r.win)}); });
function onScScores(m){ const R=S.sc.rows; $("#sc-go").disabled=false; const O=R.map((r,i)=>{ const ck=dnaChecks(r.win), jz=E.dnaJudge(r.win,S.judges.dna); return {seq:r.win,ck,cp:sig(m.scores[i][0]),csd:m.scores[i][1],jp:sig(jz),jz,pass:ck.usable&&jz>0,hit:S.lib.map.get(r.win)||null,label:r.label,strand:r.strand}; }); O.sort((a,b)=>(b.cp+b.jp)-(a.cp+a.jp));
  const top=O.slice(0,R.length===1?1:8), gap=O.reduce((s,o)=>s+Math.abs(o.cp-o.jp),0)/O.length;
  $("#sc-status").textContent="Done. Scored "+O.length+" site"+(O.length>1?"s":"")+" on your device.";
  $("#sc-out").innerHTML='<div class="metrics"><div><span>Sites scored</span><b>'+O.length+'</b><small>NGG with 30-nt context, both strands</small></div><div><span>Pass judge and checks</span><b>'+O.filter(o=>o.pass).length+'</b><small>of '+O.length+'</small></div><div><span>Best critic / judge</span><b>'+fmt(top[0].cp)+' / '+fmt(top[0].jp)+'</b><small>top-ranked site</small></div><div><span>Mean critic&ndash;judge gap</span><b>'+fmt(gap)+'</b><small>how far the two models differ</small></div></div>'+(R.length>1?'<p class="small">Top '+top.length+' by average of critic and judge. Open any row for its per-base report.</p>':'')+'<div class="dna">'+top.map((o,i)=>dnaRow(o,R.length>1?"Rank "+(i+1)+" · "+(o.strand>0?"sense":"antisense")+" strand":"",false)).join("")+'</div>'; if(O.length===1||top.length===1){ const b=$("#sc-out .repbtn"); if(b) b.click(); } turn("score",$("#sc-in").value,"#sc-out"); }

document.addEventListener("click",e=>{ const b=e.target.closest("button[data-seedsmi],button[data-scoresmi]"); if(!b) return;
  if(b.dataset.seedsmi){ const modeBtn=document.querySelector('#go-modes .mode[data-m="gen"]'); if(modeBtn) modeBtn.click(); $("#g-smi").value=b.dataset.seedsmi; const P=$("#g-prompt"); if(!P.value.trim()) P.value="Analogs of this molecule that are brain-penetrant and non-toxic. 12 candidates."; P.dispatchEvent(new Event("input")); doRead(); P.focus(); $("#g-live").textContent="Seed set: "+b.dataset.seedsmi.slice(0,28)+(b.dataset.seedsmi.length>28?"…":"")+". Press Generate."; }
  else { const m=document.querySelector('#go-modes .mode[data-m="score"]'); if(m) m.click(); $("#sc-in").value=b.dataset.scoresmi; scUpdate(); $("#sc-go").click(); } });
function onMolScores(m){ const c=S.msc; if(!c||c.id!==m.id) return; const r=m.scores[0], box=$("#sc-crit");
  if(box){ if(r&&r.ok){ const cp={bbb:sig(r.crit.bbb.mean),act:sig(r.crit.act.mean),tox:sig(r.crit.tox.mean)}; const sdv=Math.max(r.crit.bbb.sd,r.crit.act.sd,r.crit.tox.sd);
      box.innerHTML='<div class="card" style="max-width:420px"><b style="font-size:13px">Steering critics (read from the encoded molecule)</b>'+pair("brain",cp.bbb,c.jp.bbb,true)+pair("activity",cp.act,c.jp.act,true)+pair("low tox",cp.tox,c.jp.tox,false)+'<div class="small" style="font-size:11.5px">Largest disagreement between the five critic heads: '+fmt(sdv,2)+' logits'+(sdv>1?' (high: treat the critic numbers as unreliable)':'')+'. Bars compare the critic with the held-out judge.</div></div><p class="small" style="margin-top:8px">The critics read the molecule through the model\'s own encoder, so a molecule far from the training set is scored less reliably. Per-atom effects are not available.</p>'; }
    else box.innerHTML='<p class="small">Critic scores unavailable: '+esc((r&&r.why)||"no encoder in this build")+'. Judge scores above still apply.</p>'; }
  turn("score",$("#sc-in").value,"#sc-out"); }
/* ---------- photo: what the image reader found, and how to use it ---------- */
/* ---------- attachments (read on device; used silently as context for the model) ---------- */
S.files=[];
function fileFacts(a){ if(a._f) return a._f; const valid=[]; (a.smiles||[]).forEach(c=>{ for(const v of c.variants){ const an=analyse(v); if(an){ if(!valid.some(x=>x===an.canon)) valid.push(an.canon); break; } } }); const f={valid,dna:(a.dna&&a.dna.length>=20)?a.dna:"",ptprs:!!a.ptprs}; const bits=[]; if(valid.length) bits.push(valid.length+" molecule"+(valid.length>1?"s":"")); if(f.dna) bits.push(f.dna.length+" nt DNA"); if(f.ptprs) bits.push("PTPRS"); f.text=bits.length?("found "+bits.join(", ")):"no molecule or DNA strings found"; a._f=f; return f; }
function attachSeed(){ for(const a of S.files){ const f=fileFacts(a); if(f.valid.length) return f.valid[0]; } return ""; }
function attachDna(){ for(const a of S.files){ const f=fileFacts(a); if(f.dna.length>=40) return f.dna; } return ""; }
function syncPhoto(){ if(!S.files.length){ S.photo=null; return; } const first=S.files.find(a=>a.thumb)||S.files[0]; S.photo={name:S.files.map(a=>a.name).join(", "),kind:first.kind,kindLabel:S.files.length>1?S.files.length+" files":first.kindLabel,thumb:first.thumb||"",foundText:S.files.map(a=>a.name+": "+fileFacts(a).text).join("; ")}; }
function renderAtt(){ const att=$("#g-att"); att.querySelectorAll(".chipf").forEach(n=>n.remove()); S.files.forEach((a,i)=>{ const c=document.createElement("span"); c.className="chipf"; c.title=a.kindLabel+" · "+fileFacts(a).text; c.innerHTML=(a.thumb?'<img alt="" src="'+a.thumb+'">':'<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>')+'<span>'+esc(a.name.length>22?a.name.slice(0,20)+"…":a.name)+'</span><button type="button" class="att-x" data-i="'+i+'" aria-label="Remove '+esc(a.name)+'">×</button>'; att.insertBefore(c,att.firstChild?att.querySelector(".att-s"):null); }); att.hidden=!S.files.length&&!att.querySelector(".att-s"); }
$("#g-att").addEventListener("click",e=>{ const b=e.target.closest(".att-x[data-i]"); if(!b) return; S.files.splice(+b.dataset.i,1); syncPhoto(); renderAtt(); doRead(); });
document.addEventListener("ptp:photo",e=>{ if(S.files.length>=8){ $("#g-status").textContent="Up to 8 files per message."; return; } S.files.push(e.detail); syncPhoto(); renderAtt(); $("#g-status").textContent=e.detail.name+" read on this device ("+fileFacts(e.detail).text+"). It will be used as context."; doRead(); });
document.addEventListener("ptp:attach-error",e=>{ $("#g-status").textContent=e.detail; renderAtt(); });
function turn(kind,prompt,sel){ document.dispatchEvent(new CustomEvent("ptp:turn",{detail:{kind,prompt,html:$(sel).innerHTML}})); }
boot();

/* ---------- UX: Enter sends, click a SMILES in results to copy ---------- */
$("#g-prompt").addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){ const b=$("#g-make-go"); if(b&&!b.disabled){ e.preventDefault(); b.click(); } } });
document.addEventListener("click",e=>{ const t=e.target.closest(".smi[data-smi]"); if(!t||!t.closest("#thread,#g-out")||!navigator.clipboard) return; navigator.clipboard.writeText(t.dataset.smi).then(()=>{ if(t.dataset.busy) return; t.dataset.busy="1"; const o=t.textContent; t.textContent="Copied"; setTimeout(()=>{ t.textContent=o; delete t.dataset.busy; },900); }).catch(()=>{ const ta=document.createElement("textarea"); ta.value=t.dataset.smi; ta.style.position="fixed"; ta.style.opacity="0"; document.body.appendChild(ta); ta.select(); try{ document.execCommand("copy"); const o=t.textContent; t.textContent="Copied"; setTimeout(()=>{ t.textContent=o; },900); }catch(_){} ta.remove(); }); });
})();

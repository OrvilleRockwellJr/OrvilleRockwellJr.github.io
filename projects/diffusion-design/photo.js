/* Photo reader: runs entirely in the browser. The image is never uploaded.
   1) coarse image kind from pixel statistics (no model), 2) OCR with a locally hosted Tesseract (vendor/tess),
   3) pulls out SMILES-like strings, DNA runs and gene names. app.js validates and uses them ("ptp:photo" event). */
(function(){
"use strict";
const $=s=>document.querySelector(s);
const BASE=(function(){ const s=document.currentScript; return s?s.src.replace(/[^/]*$/,""):""; })();
let tessP=null, worker=null;
function loadScript(src){ return new Promise((res,rej)=>{ const s=document.createElement("script"); s.src=src; s.onload=res; s.onerror=()=>rej(new Error("could not load "+src)); document.head.appendChild(s); }); }
async function getWorker(onp){
  if(!tessP) tessP=loadScript(BASE+"vendor/tess/tesseract.min.js"); await tessP;
  if(worker) return worker;
  worker=await Tesseract.createWorker("eng",1,{workerPath:BASE+"vendor/tess/worker.min.js",corePath:BASE+"vendor/tess/",langPath:BASE+"vendor/tess/",gzip:true,workerBlobURL:false,logger:m=>{ if(onp&&m.status) onp(m.status,m.progress||0); }});
  return worker;
}
function loadImage(file){ return new Promise((res,rej)=>{ const u=URL.createObjectURL(file), im=new Image(); im.onload=()=>res({im,u}); im.onerror=()=>{ URL.revokeObjectURL(u); rej(new Error("That file did not open as an image.")); }; im.src=u; }); }
function stats(im){
  const W=192, k=W/Math.max(im.naturalWidth,im.naturalHeight), w=Math.max(1,Math.round(im.naturalWidth*k)), h=Math.max(1,Math.round(im.naturalHeight*k));
  const c=document.createElement("canvas"); c.width=w; c.height=h; const x=c.getContext("2d",{willReadFrequently:true}); x.fillStyle="#fff"; x.fillRect(0,0,w,h); x.drawImage(im,0,0,w,h);
  const d=x.getImageData(0,0,w,h).data; let gray=0,bright=0,ink=0,n=w*h;
  for(let i=0;i<d.length;i+=4){ const r=d[i],g=d[i+1],b=d[i+2], mx=Math.max(r,g,b), mn=Math.min(r,g,b), l=(r+g+b)/765; if(mx-mn<30) gray++; if(l>0.85) bright++; if(l<0.5) ink++; }
  return {gray:gray/n,bright:bright/n,ink:ink/n};
}
function thumb(im){ const W=160, k=Math.min(1,W/Math.max(im.naturalWidth,im.naturalHeight)), c=document.createElement("canvas"); c.width=Math.max(1,Math.round(im.naturalWidth*k)); c.height=Math.max(1,Math.round(im.naturalHeight*k)); c.getContext("2d").drawImage(im,0,0,c.width,c.height); return c.toDataURL("image/jpeg",0.7); }
function kindOf(st,words,conf){
  if(words>=8&&conf>=55) return {k:"text",label:"Text-heavy image (document, slide, table, screenshot)",note:"Read as text."};
  if(st.gray>0.9&&st.bright>0.7&&st.ink<0.15) return {k:"drawing",label:"Black-and-white line drawing",note:"Could be a chemical structure, a sketch or a diagram. This page cannot turn a drawn structure into a SMILES string, so type or paste one if the picture is a molecule."};
  if(st.gray>0.85) return {k:"gray",label:"Grayscale image (gel, blot, scan or microscopy-like)",note:"Recognized only by its colors. The page has no model that reads gel bands or microscopy."};
  return {k:"photo",label:"Colour photo or scene",note:"The page has no scene-recognition model, so it uses only text found in the image."};
}
const STOP=new Set(["THE","AND","FOR","WITH","FROM","THIS","THAT","ARE","NOT","BUT","YOU","ALL","CAN","WAS","HAS","HAD","ONE","TWO","FIG","TABLE","DATA","NAME","DATE","PAGE","TEXT","INFO","NOTE","TYPE","MEAN","FIGURE","RESULTS","METHODS"]);
function extract(text){
  const lines=text.split(/\r?\n/), toks=[]; lines.forEach(l=>l.split(/\s+/).forEach(t=>{ t=t.replace(/^[,;:'"“”]+|[,;'"“”]+$/g,""); if(t) toks.push(t); }));
  const smiles=[], seen=new Set();
  const consider=raw=>{ const t=raw.replace(/\s+/g,"").replace(/^[,;:'"“”]+|[,;'"“”]+$/g,""); if(t.length<6||t.length>100||seen.has(t)) return; if(!/^[A-Za-z0-9@+\-\[\]()=#$\/\\%.:|]+$/.test(t)||/^\d+(\.\d+)?$/.test(t)) return;
    const outside=t.replace(/\[[^\]]*\]/g,"X"); if(/[^BCNOPSFIclnosprX0-9@+\-()=#$\/\\%.:|]/.test(outside)) return; const atoms=(outside.match(/Cl|Br|[CNOSPFIcnosp]/g)||[]).length; if(atoms<3) return;
    if(!/[()=#\[\]\d]/.test(t)&&!(/^[CNOcnoSsFP]+$/.test(t)&&t.length>=8)) return; seen.add(t);
    const idx=[]; for(let i=0;i<t.length;i++) if("lI|0".includes(t[i])) idx.push(i); const vars=[t];
    if(idx.length&&idx.length<=8){ const subs=[]; for(let m=1;m<(1<<idx.length);m++) subs.push(m); subs.sort((x,y)=>pop(x)-pop(y)); subs.forEach(m=>{ const c=t.split(""); idx.forEach((pos,k)=>{ if(m&(1<<k)) c[pos]=t[pos]==="0"?"O":"1"; }); vars.push(c.join("")); }); }
    smiles.push({raw:t,variants:vars}); };
  const pop=x=>{ let c=0; while(x){ c+=x&1; x>>=1; } return c; };
  lines.forEach(l=>consider(l)); toks.forEach(t=>consider(t));
  let dna=""; { const runs=[]; let cur=""; lines.forEach(l=>{ const c=l.replace(/[\s\d]/g,""); if(c.length>=10&&/^[ACGTacgt]+$/.test(c)) cur+=c.toUpperCase(); else if(cur){ runs.push(cur); cur=""; } }); if(cur) runs.push(cur);
    toks.forEach(t=>{ if(/^[ACGTacgt]{20,}$/.test(t)) runs.push(t.toUpperCase()); }); runs.sort((a,b)=>b.length-a.length); if(runs[0]&&runs[0].length>=20) dna=runs[0]; }
  const ptprs=/\bPTPRS\b|PTP[-\s]?(σ|sigma|s)\b|PTPσ/i.test(text); const genes=[]; (text.match(/\b[A-Z][A-Z0-9]{2,7}\b/g)||[]).forEach(g=>{ if(!STOP.has(g)&&!/^[ACGT]+$/.test(g)&&/[A-Z]{2}/.test(g)&&!genes.includes(g)&&!/^\d/.test(g)) genes.push(g); });
  return {smiles:smiles.slice(0,8),dna,ptprs,genes:genes.slice(0,6)};
}
async function analyze(file,onp){
  if(!/^image\//.test(file.type)&&!/\.(png|jpe?g|webp|gif|bmp|heic)$/i.test(file.name)) throw new Error("Choose an image file (photo, screenshot, scan).");
  if(file.size>25e6) throw new Error("That image is over 25 MB.");
  const {im,u}=await loadImage(file); try{
    const st=stats(im), th=thumb(im); onp&&onp("loading text reader",0); const w=await getWorker(onp); onp&&onp("reading text",0);
    const r=await w.recognize(file); const text=(r.data.text||"").trim(); const words=(r.data.words||[]).filter(x=>x.confidence>=50&&/[A-Za-z0-9]{2,}/.test(x.text)).length;
    const conf=r.data.confidence||0, kind=kindOf(st,words,conf), ex=extract(text);
    return {name:file.name,thumb:th,w:im.naturalWidth,h:im.naturalHeight,kind:kind.k,kindLabel:kind.label,kindNote:kind.note,text,words,conf:Math.round(conf),...ex};
  } finally{ URL.revokeObjectURL(u); }
}
window.PTPPhoto={analyze,extract};
const btn=$("#g-photo"), inp=$("#g-file"); if(!btn||!inp) return;
btn.addEventListener("click",()=>inp.click());
async function handle(file){
  if(!file) return; const att=$("#g-att"); att.hidden=false; att.innerHTML='<span class="att-s">Reading image on this device…</span>';
  try{ const a=await analyze(file,(s,p)=>{ att.querySelector(".att-s").textContent=(s==="recognizing text"?"Reading text":s.charAt(0).toUpperCase()+s.slice(1))+(p?" "+Math.round(p*100)+"%":"…"); });
    document.dispatchEvent(new CustomEvent("ptp:photo",{detail:a})); }
  catch(e){ att.innerHTML='<span class="att-s">'+String(e.message||e).replace(/</g,"&lt;")+'</span> <button type="button" class="att-x" aria-label="Dismiss">×</button>'; att.querySelector(".att-x").onclick=()=>{ att.hidden=true; att.innerHTML=""; }; }
  inp.value="";
}
inp.addEventListener("change",()=>handle(inp.files&&inp.files[0]));
const dock=btn.closest(".composer")||document; ["dragover","drop"].forEach(ev=>dock.addEventListener(ev,e=>{ if(!e.dataTransfer||!e.dataTransfer.files||!e.dataTransfer.files.length) return; e.preventDefault(); if(ev==="drop") handle(e.dataTransfer.files[0]); }));
document.addEventListener("paste",e=>{ const it=[...(e.clipboardData&&e.clipboardData.items||[])].find(i=>i.type.startsWith("image/")); if(it&&$("#go-gen")&&!$("#go-gen").hidden){ const f=it.getAsFile(); if(f) handle(f); } });
})();

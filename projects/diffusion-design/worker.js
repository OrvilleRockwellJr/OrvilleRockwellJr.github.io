/* Generation worker: runs the guided-diffusion engine off the UI thread. */
importScripts("engine.js");
const E=self.GDEngine; let M=null, Dn=null, cancelled=new Set();
const tick=()=>new Promise(r=>setTimeout(r,0));
const PG={};
async function load(name){ const rj=await fetch("models/"+name+".json"); if(!rj.ok) throw new Error("models/"+name+".json: HTTP "+rj.status); const J=JSON.parse(await rj.text()); const rb=await fetch("models/"+name+".bin"); if(!rb.ok) throw new Error("models/"+name+".bin: HTTP "+rb.status); const tot=+rb.headers.get("content-length")||{mol:13772624,dna:5369844}[name]||0; let buf; if(rb.body&&rb.body.getReader){ const rd=rb.body.getReader(), parts=[]; let got=0; for(;;){ const {done,value}=await rd.read(); if(done) break; parts.push(value); got+=value.length; PG[name]=[got,tot]; let g=0,t=0; for(const k in PG){ g+=PG[k][0]; t+=PG[k][1]; } postMessage({evt:"prog",pct:t?Math.min(99,Math.round(100*g/t)):0}); } const u=new Uint8Array(got); let o=0; for(const p of parts){ u.set(p,o); o+=p.length; } buf=u.buffer; } else buf=await rb.arrayBuffer(); return {J,W:new E.Weights(J.manifest,buf)}; }
async function init(){
  const m=await load("mol"), d=await load("dna");
  M={J:m.J,W:m.W,den:new E.Denoiser(m.W,"den."),ens:{bbb:new E.Ensemble(m.W,"bbb."),act:new E.Ensemble(m.W,"act."),tox:new E.Ensemble(m.W,"tox.")},dec:new E.MolDecoder(m.W,m.J.vocab,m.J.L),enc:m.W.has("enc.emb")?new E.MolEncoder(m.W,m.J.vocab,m.J.L):null,mean:m.W.get("stat.mean"),std:m.W.get("stat.std")};
  Dn={J:d.J,W:d.W,den:new E.Denoiser(d.W,"den."),ens:new E.Ensemble(d.W,"eff.")};
  self.postMessage({evt:"ready",info:{molD:M.den.D,molHeads:M.ens.bbb.K,dnaD:Dn.den.D,dnaHeads:Dn.ens.K}});
}
function unstd(z){ const o=new Float32Array(z.length); for(let i=0;i<z.length;i++) o[i]=z[i]*M.std[i]+M.mean[i]; return o; }
function molCrit(x){ const r={}; r.bbb=M.ens.bbb.eval(x,0,1,false); r.act=M.ens.act.eval(x,0,1,false); r.tox=M.ens.tox.eval(x,0,0,false); return r; }
function molRun(job,i,gamma,trace){
  const rnd=E.rng((job.seed*7919+i*104729)>>>0); const D=M.den.D, x=new Float32Array(D); for(let k=0;k<D;k++) x[k]=rnd.n();
  const gate=job.gate?"ens":"none"; const objs=[{ens:M.ens.bbb,want:1,weight:job.w[0]},{ens:M.ens.act,want:1,weight:job.w[1]},{ens:M.ens.tox,want:0,weight:job.w[2]}];
  const snaps=new Set(); if(trace) for(let s=0;s<8;s++) snaps.add(Math.round((job.steps-1)*s/7));
  const xf=E.sample({den:M.den,objectives:objs,gamma,tau:job.tau,gate,eta:0,steps:job.steps,x,rnd,
    onStep:(si,n,x0,info)=>{ if(trace&&snaps.has(si)){ const dec=M.dec.decode(unstd(x0)); self.postMessage({evt:"trace",id:job.id,kind:"mol",si,n,smiles:dec.smiles,gate:info.gate}); } }});
  const alts=M.dec.decodeMulti(unstd(xf),8,0.06,rnd); const c=molCrit(xf);
  return {smiles:alts[0],alts:alts.slice(1),crit:{bbb:c.bbb,act:c.act,tox:c.tox}};
}
function stdz(mu){ const z=new Float32Array(mu.length); for(let i=0;i<mu.length;i++) z[i]=(mu[i]-M.mean[i])/M.std[i]; return z; }
/* analog design (SDEdit-style): encode the seed molecule, noise it to a partial step, then run the same guided reverse process and decode */
function analogRun(job,i){
  const e=M.enc.encode(job.smiles); if(!e.ok) throw new Error("cannot encode: "+e.why);
  const z0=stdz(e.mu), D=z0.length, rnd=E.rng((job.seed*7919+i*104729)>>>0), tS=Math.max(1,Math.min(E.T-1,Math.round(job.strength*(E.T-1)))), ab=E.AB[tS];
  const x=new Float32Array(D); for(let k=0;k<D;k++) x[k]=Math.sqrt(ab)*z0[k]+Math.sqrt(1-ab)*rnd.n();
  const objs=[{ens:M.ens.bbb,want:1,weight:job.w[0]},{ens:M.ens.act,want:1,weight:job.w[1]},{ens:M.ens.tox,want:0,weight:job.w[2]}];
  const xf=E.sample({den:M.den,objectives:objs,gamma:job.gamma,tau:job.tau,gate:job.gate?"ens":"none",eta:0,steps:job.steps,x,rnd,tStart:tS});
  const alts=M.dec.decodeMulti(unstd(xf),8,0.06,rnd); const c=molCrit(xf); return {smiles:alts[0],alts:alts.slice(1),crit:{bbb:c.bbb,act:c.act,tox:c.tox}}; }
const libCache={};
function getLib(job){ if(libCache.key===job.libKey) return libCache.crit;
  const lsig=z=>-Math.log(1+Math.exp(-z)); const lw=Float64Array.from(job.lib,s=>(job.tilt||0)*lsig(Dn.ens.eval(E.dnaToX(s),0,1,false).mean));
  libCache.key=job.libKey; libCache.crit=new E.LibCritic(job.lib,0.02,lw); return libCache.crit; }
function dnaRun(job,i,gamma,trace){
  const rnd=E.rng((job.seed*6151+i*92821)>>>0); const D=Dn.den.D, x=new Float32Array(D); for(let k=0;k<D;k++) x[k]=rnd.n();
  const snaps=new Set(); if(trace) for(let s=0;s<10;s++) snaps.add(Math.round((job.steps-1)*s/9));
  const objs=[]; if((job.ew==null?1:job.ew)>0) objs.push({ens:Dn.ens,want:1,weight:job.ew==null?1:job.ew}); if(job.lib&&job.lib.length&&job.wl>0&&gamma>0) objs.push({ens:getLib(job),want:1,weight:job.wl});
  const xf=E.sample({den:Dn.den,objectives:objs,gamma,tau:job.tau,gate:job.gate?"ens":"none",eta:0,steps:job.steps,clip:1.5,x,rnd,
    onStep:(si,n,x0,info)=>{ if(trace&&snaps.has(si)) self.postMessage({evt:"trace",id:job.id,kind:"dna",si,n,seq:E.dnaFromX(x0),conf:E.dnaConf(x0)}); }});
  // final sequence: clip like the training-time sampler, read argmax of the final x
  const seq=E.dnaFromX(xf); const r=Dn.ens.eval(E.dnaToX(seq),0,1,false);
  return {seq,crit:r};
}
/* per-base report for a 30-nt site: in-silico mutagenesis of the critic ensemble, and prior "surprise" (denoising error under the coding-sequence prior) */
const surpRef={};
function surprise(seq){ const x0=E.dnaToX(seq), D=120, pos=new Float64Array(30), eps=new Float32Array(D), xt=new Float32Array(D), out=new Float32Array(D), rnd=E.rng(12345); let n=0;
  for(const t of [20,60,100,140]){ const ab=E.AB[t], sa=Math.sqrt(ab), s1=Math.sqrt(1-ab); for(let d=0;d<3;d++){ for(let k=0;k<D;k++){ eps[k]=rnd.n(); xt[k]=sa*x0[k]+s1*eps[k]; } Dn.den.eps(xt,t,out); for(let i=0;i<30;i++){ let e=0; for(let c=0;c<4;c++){ const k=i*4+c; e+=(out[k]-eps[k])*(out[k]-eps[k]); } pos[i]+=e/4; } n++; } }
  let m=0; for(let i=0;i<30;i++){ pos[i]/=n; m+=pos[i]; } return {pos:Array.from(pos),mean:m/30}; }
function report(seq,lib,libKey){
  const NT="ACGT", cur=Dn.ens.eval(E.dnaToX(seq),0,1,false); const ism=[], sd=[];
  for(let i=0;i<30;i++){ const row=[],rs=[]; for(let b=0;b<4;b++){ if(NT[b]===seq[i]){ row.push(cur.mean); rs.push(cur.sd); continue; } const s=seq.slice(0,i)+NT[b]+seq.slice(i+1); const r=Dn.ens.eval(E.dnaToX(s),0,1,false); row.push(r.mean); rs.push(r.sd); } ism.push(row); sd.push(rs); }
  const su=surprise(seq); let pct=null;
  if(lib&&lib.length){ if(!surpRef[libKey]){ const rnd=E.rng(77), m=[]; for(let j=0;j<120;j++) m.push(surprise(lib[Math.floor(rnd.u()*lib.length)]).mean); m.sort((a,b)=>a-b); surpRef[libKey]=m; } const ref=surpRef[libKey]; pct=ref.filter(v=>v<=su.mean).length/ref.length; }
  return {seq,crit:{mean:cur.mean,sd:cur.sd},ism,sd,surp:su,pct}; }
self.onmessage=async(ev)=>{
  const m=ev.data;
  try{
    if(m.cmd==="init") return await init();
    if(m.cmd==="cancel"){ cancelled.add(m.id); return; }
    if(m.cmd==="genMol"||m.cmd==="genDna"){
      const run=m.cmd==="genMol"?molRun:dnaRun;
      for(let i=0;i<m.n;i++){
        if(cancelled.has(m.id)) break;
        const out={evt:"item",id:m.id,i,kind:m.cmd==="genMol"?"mol":"dna"};
        out.guided=run(m,i,m.gamma,i===0);
        if(m.compare) out.unguided=run(m,i,0,false);
        self.postMessage(out); await tick();
      }
      self.postMessage({evt:"done",id:m.id,cancelled:cancelled.has(m.id)}); return;
    }
    if(m.cmd==="genAnalog"){
      if(!M.enc) throw new Error("this model build has no encoder");
      { const pr=M.enc.tokenize(m.smiles); if(!pr.ok){ self.postMessage({evt:"seedError",id:m.id,why:pr.why}); return; }
        const e0=M.enc.encode(m.smiles); self.postMessage({evt:"seedCheck",id:m.id,recon:M.dec.decode(unstd(stdz(e0.mu))).smiles}); }
      for(let i=0;i<m.n;i++){ if(cancelled.has(m.id)) break; self.postMessage({evt:"item",id:m.id,i,kind:"mol",guided:analogRun(m,i)}); await tick(); }
      self.postMessage({evt:"done",id:m.id,cancelled:cancelled.has(m.id)}); return; }
    if(m.cmd==="scoreMol"){
      const out=m.smiles.map(s=>{ if(!M.enc) return {ok:false,why:"no encoder"}; const e=M.enc.encode(s); if(!e.ok) return {ok:false,why:e.why}; const z=stdz(e.mu); const c=molCrit(z); return {ok:true,crit:{bbb:c.bbb,act:c.act,tox:c.tox}}; });
      self.postMessage({evt:"molScores",id:m.id,scores:out}); return; }
    if(m.cmd==="report"){ const out=m.seqs.map(s=>report(s,m.lib,m.libKey)); self.postMessage({evt:"report",id:m.id,reports:out}); return; }
    if(m.cmd==="scoreDna"){
      const out=[]; for(let i=0;i<m.seqs.length;i++){ const r=Dn.ens.eval(E.dnaToX(m.seqs[i]),0,1,false); out.push([r.mean,r.sd]); if(i%200===0){ self.postMessage({evt:"scanprog",id:m.id,i,n:m.seqs.length}); await tick(); } }
      self.postMessage({evt:"scores",id:m.id,scores:out}); return;
    }
  }catch(err){ self.postMessage({evt:"error",message:String(err&&err.stack||err)}); }
};

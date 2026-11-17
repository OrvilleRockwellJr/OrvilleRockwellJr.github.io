/* Generation worker: runs the guided-diffusion engine off the UI thread. */
importScripts("engine.js");
const E=self.GDEngine; let M=null, Dn=null, cancelled=new Set();
const tick=()=>new Promise(r=>setTimeout(r,0));
async function load(name){ const J=await (await fetch("models/"+name+".json")).json(); const buf=await (await fetch("models/"+name+".bin")).arrayBuffer(); return {J,W:new E.Weights(J.manifest,buf)}; }
async function init(){
  const m=await load("mol"), d=await load("dna");
  M={J:m.J,W:m.W,den:new E.Denoiser(m.W,"den."),ens:{bbb:new E.Ensemble(m.W,"bbb."),act:new E.Ensemble(m.W,"act."),tox:new E.Ensemble(m.W,"tox.")},dec:new E.MolDecoder(m.W,m.J.vocab,m.J.L),mean:m.W.get("stat.mean"),std:m.W.get("stat.std")};
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
  const dec=M.dec.decode(unstd(xf)); const c=molCrit(xf);
  return {smiles:dec.smiles,crit:{bbb:c.bbb,act:c.act,tox:c.tox}};
}
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
    if(m.cmd==="scoreDna"){
      const out=[]; for(let i=0;i<m.seqs.length;i++){ const r=Dn.ens.eval(E.dnaToX(m.seqs[i]),0,1,false); out.push([r.mean,r.sd]); if(i%200===0){ self.postMessage({evt:"scanprog",id:m.id,i,n:m.seqs.length}); await tick(); } }
      self.postMessage({evt:"scores",id:m.id,scores:out}); return;
    }
  }catch(err){ self.postMessage({evt:"error",message:String(err&&err.stack||err)}); }
};

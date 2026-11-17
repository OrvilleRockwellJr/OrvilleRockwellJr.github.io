/* Guided-diffusion engine, mirrored 1:1 from scripts/gd.py (PyTorch reference). No dependencies.
   Used by worker.js in the browser and by parity.js under Node. */
(function(root){
"use strict";
const T=200;
function alphaBarTable(){ // cosine schedule, identical to gd.alpha_bar_table (float64 math, stored float32)
  const s=0.008, f=i=>Math.pow(Math.cos(((i/T)+s)/(1+s)*Math.PI/2),2), f0=f(0), ab=new Float32Array(T);
  for(let i=1;i<=T;i++){ let v=f(i)/f0; v=Math.min(Math.max(v,1e-5),0.9999); ab[i-1]=v; }
  return ab;
}
const AB=alphaBarTable();
const sigmoid=x=>1/(1+Math.exp(-x)), silu=x=>x/(1+Math.exp(-x));
const dsilu=x=>{const s=1/(1+Math.exp(-x)); return s*(1+x*(1-s));};

/* ---------- weights ---------- */
class Weights{
  constructor(manifest,buf){ this.m=manifest; this.f=new Float32Array(buf); }
  get(name){ const e=this.m[name]; if(!e) throw new Error("missing weight "+name); return this.f.subarray(e.off,e.off+e.n); }
  shape(name){ return this.m[name].shape; }
  has(name){ return name in this.m; }
}
/* y = W x + b ; W is [out][in] row-major */
function lin(W,b,x,y,out,inn){
  for(let o=0;o<out;o++){ let s=b?b[o]:0; const r=o*inn; for(let i=0;i<inn;i++) s+=W[r+i]*x[i]; y[o]=s; }
}

/* ---------- time features ---------- */
const FR=new Float64Array(16); for(let i=0;i<16;i++) FR[i]=Math.exp(-Math.log(10000)*i/16);
function tfeat(t,out){ const x=(t/T)*1000; for(let i=0;i<16;i++){ const a=x*FR[i]; out[i]=Math.sin(a); out[16+i]=Math.cos(a);} }

/* ---------- denoiser ---------- */
class Denoiser{
  constructor(W,prefix){
    this.W=W; this.p=prefix; const g=n=>W.get(prefix+n);
    this.t1w=g("t1.w"); this.t1b=g("t1.b"); this.inw=g("inp.w"); this.inb=g("inp.b"); this.outw=g("out.w"); this.outb=g("out.b");
    this.D=W.shape(prefix+"out.w")[0]; this.H=W.shape(prefix+"inp.w")[0]; this.nb=0;
    this.w1=[];this.b1=[];this.wt=[];this.w2=[];this.b2=[];
    while(W.has(prefix+"w1."+this.nb+".w")){ const i=this.nb++; this.w1.push(g("w1."+i+".w")); this.b1.push(g("w1."+i+".b")); this.wt.push(g("wt."+i+".w")); this.w2.push(g("w2."+i+".w")); this.b2.push(g("w2."+i+".b")); }
    this.tf=new Float32Array(32); this.te=new Float32Array(128); this.h=new Float32Array(this.H); this.u=new Float32Array(this.H); this.v=new Float32Array(this.H);
    this.xin=new Float32Array(this.D+128);
  }
  eps(x,t,out){
    const D=this.D,H=this.H; tfeat(t,this.tf); lin(this.t1w,this.t1b,this.tf,this.te,128,32); for(let i=0;i<128;i++) this.te[i]=silu(this.te[i]);
    for(let i=0;i<D;i++) this.xin[i]=x[i]; for(let i=0;i<128;i++) this.xin[D+i]=this.te[i];
    lin(this.inw,this.inb,this.xin,this.h,H,D+128);
    for(let b=0;b<this.nb;b++){
      lin(this.w1[b],this.b1[b],this.h,this.u,H,H);
      // wt has no bias
      const wt=this.wt[b]; for(let o=0;o<H;o++){ let s=0; const r=o*128; for(let i=0;i<128;i++) s+=wt[r+i]*this.te[i]; this.u[o]=silu(this.u[o]+s); }
      lin(this.w2[b],this.b2[b],this.u,this.v,H,H); for(let o=0;o<H;o++) this.h[o]+=this.v[o];
    }
    for(let o=0;o<H;o++) this.u[o]=silu(this.h[o]);
    lin(this.outw,this.outb,this.u,out,D,H);
  }
}
/* ---------- critic head with analytic input gradient ---------- */
class Critic{
  constructor(W,prefix){ const g=n=>W.get(prefix+n); this.l1w=g("l1.w"); this.l1b=g("l1.b"); this.l2w=g("l2.w"); this.l2b=g("l2.b"); this.l3w=g("l3.w"); this.l3b=g("l3.b");
    this.H=W.shape(prefix+"l1.w")[0]; this.D=W.shape(prefix+"l1.w")[1]-2;
    const H=this.H; this.inp=new Float32Array(this.D+2); this.a1=new Float32Array(H); this.z1=new Float32Array(H); this.a2=new Float32Array(H); this.z2=new Float32Array(H); this.d2=new Float32Array(H); this.d1=new Float32Array(H); }
  /* logit and (optionally) gradient of logit w.r.t. x accumulated into gacc with coefficient c */
  logit(x,t,gacc,c){
    const D=this.D,H=this.H,ab=AB[t]; const inp=this.inp; for(let i=0;i<D;i++) inp[i]=x[i]; inp[D]=Math.sqrt(ab); inp[D+1]=Math.sqrt(1-ab);
    lin(this.l1w,this.l1b,inp,this.z1,H,D+2); for(let i=0;i<H;i++) this.a1[i]=silu(this.z1[i]);
    lin(this.l2w,this.l2b,this.a1,this.z2,H,H); for(let i=0;i<H;i++) this.a2[i]=silu(this.z2[i]);
    let z=this.l3b[0]; for(let i=0;i<H;i++) z+=this.l3w[i]*this.a2[i];
    if(gacc){
      for(let i=0;i<H;i++) this.d2[i]=this.l3w[i]*dsilu(this.z2[i]);
      for(let j=0;j<H;j++){ let s=0; for(let i=0;i<H;i++) s+=this.d2[i]*this.l2w[i*H+j]; this.d1[j]=s*dsilu(this.z1[j]); }
      const w=this.l1w, n=D+2; for(let k=0;k<D;k++){ let s=0; for(let j=0;j<H;j++) s+=this.d1[j]*w[j*n+k]; gacc[k]+=c*s; }
    }
    return z;
  }
}
class Ensemble{ // K heads for one objective
  constructor(W,prefix){ this.heads=[]; let k=0; while(W.has(prefix+k+".l1.w")){ this.heads.push(new Critic(W,prefix+k+".")); k++; } this.K=k; this.D=this.heads[0].D; this.g=new Float32Array(this.D); }
  /* returns {mean, sd}; fills this.gradLogP with grad of log p(want|x) wrt x (mean-logit ensemble) */
  eval(x,t,want,withGrad){
    const K=this.K, D=this.D, lg=new Float64Array(K); const sgn=want===1?1:-1; this.g.fill(0);
    for(let k=0;k<K;k++) lg[k]=this.heads[k].logit(x,t,null,0);
    let m=0; for(let k=0;k<K;k++) m+=lg[k]; m/=K; let v=0; for(let k=0;k<K;k++) v+=(lg[k]-m)*(lg[k]-m); const sd=K>1?Math.sqrt(v/(K-1)):0;
    if(withGrad){ const coef=sgn*sigmoid(-sgn*m)/K; for(let k=0;k<K;k++) this.heads[k].logit(x,t,this.g,coef); }
    return {mean:m,sd:sd};
  }
}
/* ---------- rng ---------- */
function rng(seed){ let a=seed>>>0; const u=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
  return {u, n:()=>{let x=0,y=0;while(x===0)x=u();y=u();return Math.sqrt(-2*Math.log(x))*Math.cos(2*Math.PI*y);}}; }

/* ---------- guided DDIM sampler (mirror of gd.sample) ----------
   opts: den, objectives:[{ens,want,weight}], gamma, tau, gate('ens'|'none'), eta, steps, clip, x (Float32Array D, initial noise), rnd, onStep(si,nsteps,x0hat,info) */
function timeSeq(steps){ const ts=new Set(); for(let i=0;i<steps;i++){ const v=Math.round((T-1)+(0-(T-1))*(steps>1?i/(steps-1):0)); ts.add(v);} return Array.from(ts).sort((a,b)=>b-a); }
function sample(o){
  const D=o.den.D, x=Float32Array.from(o.x), eps=new Float32Array(D), gs=new Float32Array(D), x0=new Float32Array(D); const ts=timeSeq(o.steps).filter(v=>o.tStart==null||v<=o.tStart);
  const info={scores:[],gates:[]};
  for(let si=0;si<ts.length;si++){
    const ti=ts[si], ab=AB[ti], abp=si+1<ts.length?AB[ts[si+1]]:1.0;
    o.den.eps(x,ti,eps);
    let gateAvg=1; 
    if(o.gamma>0&&o.objectives&&o.objectives.length){
      gs.fill(0); let gsum=0,ng=0;
      for(const ob of o.objectives){ if(!(ob.weight>0)) continue; const r=ob.ens.eval(x,ti,ob.want,true); let gate=1; if(o.gate==="ens") gate=1/(1+(r.sd/o.tau)*(r.sd/o.tau));
        gsum+=gate; ng++; for(let k=0;k<D;k++) gs[k]+=ob.weight*gate*ob.ens.g[k]; ob.last={mean:r.mean,sd:r.sd,gate}; }
      gateAvg=ng?gsum/ng:1; const sc=Math.sqrt(1-ab)*o.gamma; for(let k=0;k<D;k++) eps[k]-=sc*gs[k];
    }
    const sa=Math.sqrt(ab), s1=Math.sqrt(1-ab);
    for(let k=0;k<D;k++){ let v=(x[k]-s1*eps[k])/sa; if(o.clip!=null&&o.clip>0){ v=Math.max(-o.clip,Math.min(o.clip,v)); eps[k]=(x[k]-sa*v)/s1; } x0[k]=v; }
    const sig=(o.eta||0)*Math.sqrt((1-abp)/(1-ab))*Math.sqrt(Math.max(0,1-ab/abp)); const c2=Math.sqrt(Math.max(0,1-abp-sig*sig)), sap=Math.sqrt(abp);
    for(let k=0;k<D;k++){ x[k]=sap*x0[k]+c2*eps[k]+(sig>0?sig*o.rnd.n():0); }
    if(o.onStep) o.onStep(si,ts.length,x0,{gate:gateAvg,t:ti});
  }
  return x;
}
/* ---------- molecule decoder (GRU) ---------- */
class MolDecoder{
  constructor(W,vocab,L){ this.vocab=vocab; this.L=L||48; this.zh=W.get("dec.zh.w"); this.zhb=W.get("dec.zh.b"); this.demb=W.get("dec.demb"); this.wih=W.get("dec.gru.wih"); this.whh=W.get("dec.gru.whh"); this.bih=W.get("dec.gru.bih"); this.bhh=W.get("dec.gru.bhh"); this.ow=W.get("dec.out.w"); this.ob=W.get("dec.out.b");
    this.Z=W.shape("dec.zh.w")[1]; this.H=W.shape("dec.zh.w")[0]; this.E=W.shape("dec.demb")[1]; this.V=W.shape("dec.demb")[0]; }
  decode(z){
    const {Z,H,E,V,L}=this; const h=new Float32Array(H), nh=new Float32Array(H), xin=new Float32Array(E+Z), gi=new Float32Array(3*H), gh=new Float32Array(3*H), lg=new Float32Array(V);
    lin(this.zh,this.zhb,z,h,H,Z); for(let i=0;i<H;i++) h[i]=Math.tanh(h[i]);
    let tok=1; const out=[]; const IN=E+Z;
    for(let t=0;t<L;t++){
      for(let i=0;i<E;i++) xin[i]=this.demb[tok*E+i]; for(let i=0;i<Z;i++) xin[E+i]=z[i];
      lin(this.wih,this.bih,xin,gi,3*H,IN); lin(this.whh,this.bhh,h,gh,3*H,H);
      for(let i=0;i<H;i++){ const r=sigmoid(gi[i]+gh[i]), zz=sigmoid(gi[H+i]+gh[H+i]), n=Math.tanh(gi[2*H+i]+r*gh[2*H+i]); nh[i]=(1-zz)*n+zz*h[i]; }
      h.set(nh); lin(this.ow,this.ob,h,lg,V,H); let b=0; for(let i=1;i<V;i++) if(lg[i]>lg[b]) b=i; tok=b; out.push(b); }
    let s=""; for(const i of out){ if(i===2) break; if(i>2) s+=this.vocab[i]; } return {tokens:out,smiles:s};
  }
}
/* ---------- molecule encoder (mirror of train_ae.AE.encode, mean only) ---------- */
const SMI_TOK=/(\[[^\]]+]|Br|Cl|%\d\d|[A-Za-z]|\d|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$)/g;
class MolEncoder{
  constructor(W,vocab,L){ this.L=L||48; this.idx={}; vocab.forEach((t,i)=>{this.idx[t]=i;});
    this.emb=W.get("enc.emb"); this.E=W.shape("enc.emb")[1];
    this.c=[1,2,3].map(k=>({w:W.get("enc.c"+k+".w"),b:W.get("enc.c"+k+".b"),sh:W.shape("enc.c"+k+".w")}));
    this.fcw=W.get("enc.fc.w"); this.fcb=W.get("enc.fc.b"); this.muw=W.get("enc.mu.w"); this.mub=W.get("enc.mu.b");
    this.H=W.shape("enc.fc.w")[0]; this.Z=W.shape("enc.mu.w")[0]; }
  tokenize(smi){ const t=smi.match(SMI_TOK)||[]; if(t.join("")!==smi) return {ok:false,why:"unsupported characters"};
    const ids=[]; for(const x of t){ if(!(x in this.idx)) return {ok:false,why:"token "+x+" not in the model vocabulary"}; ids.push(this.idx[x]); }
    ids.push(2); if(ids.length>this.L) return {ok:false,why:"longer than "+(this.L-1)+" tokens"};
    while(ids.length<this.L) ids.push(0); return {ok:true,ids}; }
  conv(x,Tin,Cin,l,stride){ const [Co,Ci,K]=l.sh, Tout=Math.floor((Tin+4-K)/stride)+1, y=new Float32Array(Co*Tout);
    for(let o=0;o<Co;o++){ for(let t=0;t<Tout;t++){ let s=l.b[o]; for(let i=0;i<Ci;i++){ const wb=(o*Ci+i)*K, xb=i*Tin; for(let k=0;k<K;k++){ const p=t*stride+k-2; if(p>=0&&p<Tin) s+=l.w[wb+k]*x[xb+p]; } } y[o*Tout+t]=s>0?s:0; } }
    return {y,T:Tout}; }
  encode(smi){ const tk=this.tokenize(smi); if(!tk.ok) return tk; const L=this.L,E=this.E; let x=new Float32Array(E*L);
    for(let t=0;t<L;t++){ const id=tk.ids[t]; for(let e=0;e<E;e++) x[e*L+t]=this.emb[id*E+e]; }
    let T=L, C=E; const strides=[1,2,2]; for(let k=0;k<3;k++){ const r=this.conv(x,T,C,this.c[k],strides[k]); x=r.y; T=r.T; C=this.c[k].sh[0]; }
    const h=new Float32Array(this.H); lin(this.fcw,this.fcb,x,h,this.H,C*T); for(let i=0;i<this.H;i++) if(h[i]<0) h[i]=0;
    const mu=new Float32Array(this.Z); lin(this.muw,this.mub,h,mu,this.Z,this.H); return {ok:true,mu}; }
}
/* ---------- DNA helpers ---------- */
const NT="ACGT";
function dnaFromX(x0){ let s=""; for(let i=0;i<30;i++){ let b=0; for(let j=1;j<4;j++) if(x0[i*4+j]>x0[i*4+b]) b=j; s+=NT[b]; } return s; }
function dnaToX(s){ const x=new Float32Array(120).fill(-1); for(let i=0;i<30;i++) x[i*4+NT.indexOf(s[i])]=1; return x; }
function dnaConf(x0){ const c=[]; for(let i=0;i<30;i++){ let m=-1e9,e=0; for(let j=0;j<4;j++) if(x0[i*4+j]>m) m=x0[i*4+j]; for(let j=0;j<4;j++) e+=Math.exp(3*(x0[i*4+j]-m)); c.push(1/e); } return c; }
/* DNA judge: logistic regression on positional one-hot + dinucleotides + GC (mirror of train_dna.feats) */
function dnaJudge(s,J){ const f=[]; for(let i=0;i<30;i++) for(let j=0;j<4;j++) f.push(NT[j]===s[i]?1:0);
  for(let i=0;i<29;i++) for(let a=0;a<4;a++) for(let b=0;b<4;b++) f.push(NT[a]===s[i]&&NT[b]===s[i+1]?1:0);
  let gc=0; for(let i=4;i<24;i++) if(s[i]==="G"||s[i]==="C") gc++; gc/=20; f.push(gc,gc*gc);
  let z=J.b; for(let i=0;i<f.length;i++) z+=J.coef[i]*f[i]; return z; }

/* ---------- target-site library critic ----------
   Exact score of a Gaussian mixture centred on the library sites, evaluated on the noised input x_t:
   log p(x_t) = logsumexp_j( logw_j - ||x_t - sqrt(ab) s_j||^2 / (2 v) ), v = max(1-ab, floor). Gradient = posterior-weighted pull toward the sites.
   It is a retrieval score, not a learned one, so it has no ensemble spread (sd = 0 and the trust gate is a no-op for it). */
class LibCritic{
  constructor(seqs,floor,logw){ this.N=seqs.length; this.lw=logw||new Float64Array(seqs.length); this.D=120; this.floor=floor||0.02; this.S=new Float32Array(this.N*120); for(let j=0;j<this.N;j++){ const x=dnaToX(seqs[j]); this.S.set(x,j*120); } this.g=new Float32Array(120); this.lg=new Float64Array(this.N); }
  eval(x,t,want,wantGrad){ const ab=AB[t], sa=Math.sqrt(ab), v=Math.max(1-ab,this.floor), N=this.N, S=this.S; let xx=0; for(let k=0;k<120;k++) xx+=x[k]*x[k];
    let mx=-Infinity; for(let j=0;j<N;j++){ let d=0; const o=j*120; for(let k=0;k<120;k++) d+=x[k]*S[o+k]; const l=-(xx-2*sa*d+ab*120)/(2*v)+this.lw[j]; this.lg[j]=l; if(l>mx) mx=l; }
    let z=0; for(let j=0;j<N;j++){ const e=Math.exp(this.lg[j]-mx); this.lg[j]=e; z+=e; }
    if(wantGrad){ const g=this.g; g.fill(0); for(let j=0;j<N;j++){ const w=this.lg[j]/z; if(w<1e-6) continue; const o=j*120; for(let k=0;k<120;k++) g[k]+=w*S[o+k]; } for(let k=0;k<120;k++) g[k]=(sa*g[k]-x[k])/v; }
    return {mean:mx+Math.log(z),sd:0}; }
}
const COMPL={A:"T",C:"G",G:"C",T:"A"}; const rc=s=>[...s].reverse().map(c=>COMPL[c]||"N").join("");
/* every NGG site with its full 30-nt context (4 nt, 20-nt protospacer, NGG, 3 nt), both strands; cutIn = predicted cut inside [exonLo,exonHi) */
function findSites(seq,label,exonLo,exonHi,rows){
  for(const strand of [1,-1]){ const s=strand===1?seq:rc(seq); const lo=strand===1?exonLo:seq.length-exonHi, hi=strand===1?exonHi:seq.length-exonLo;
    for(let w=0;w+30<=s.length;w++){ const win=s.slice(w,w+30); if(win[25]!=="G"||win[26]!=="G"||/[^ACGT]/.test(win)) continue; const cut=w+4+17; rows.push({win,label,strand,cutIn:cut>=lo&&cut<hi}); } }
}
const API={LibCritic,findSites,rc,T,AB,Weights,Denoiser,Critic,Ensemble,rng,sample,timeSeq,MolDecoder,MolEncoder,dnaFromX,dnaToX,dnaConf,dnaJudge,sigmoid};
if(typeof module!=="undefined"&&module.exports) module.exports=API; else root.GDEngine=API;
})(typeof self!=="undefined"?self:globalThis);

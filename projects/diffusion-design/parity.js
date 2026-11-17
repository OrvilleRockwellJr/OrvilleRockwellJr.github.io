// Node parity check: engine.js vs PyTorch reference (gd.py). Run: node web/parity.js
const fs=require("fs"), E=require("./engine.js"); const dir=__dirname+"/models/";
const P=JSON.parse(fs.readFileSync(__dirname+"/./results/parity_reference.json"));
let fail=0; const chk=(name,a,b,tol)=>{ let m=0; for(let i=0;i<a.length;i++) m=Math.max(m,Math.abs(a[i]-b[i])); const ok=m<=tol; if(!ok) fail++; console.log((ok?"PASS":"FAIL"),name,"max|diff|="+m.toExponential(2)); };
function load(n){ const J=JSON.parse(fs.readFileSync(dir+n+".json")); const b=fs.readFileSync(dir+n+".bin"); return {J,W:new E.Weights(J.manifest,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength))}; }
for(const name of ["mol","dna"]){
  const {J,W}=load(name); const den=new E.Denoiser(W,"den."); const R=P[name]; const D=den.D;
  const objNames=name==="mol"?["bbb","act","tox"]:["eff"]; const ens={}; for(const o of objNames) ens[o]=new E.Ensemble(W,o+".");
  for(let i=0;i<R.x.length;i++){ const o=new Float32Array(D); den.eps(Float32Array.from(R.x[i]),R.t[i],o); chk(name+" denoiser eps["+i+"]",o,R.eps[i],2e-3);
    for(const c of R.crit){ const want=(name==="mol"&&c.name==="tox")?0:1; const r=ens[c.name].eval(Float32Array.from(R.x[i]),R.t[i],want,true); chk(name+" "+c.name+" mean["+i+"]",[r.mean],[c.mean[i]],2e-3); chk(name+" "+c.name+" sd["+i+"]",[r.sd],[c.sd[i]],2e-3); chk(name+" "+c.name+" grad["+i+"]",ens[c.name].g,c.grad[i],2e-3); } }
  for(const gate of ["none","ens"]){ for(let s=0;s<2;s++){
    const objs=objNames.map(o=>({ens:ens[o],want:(name==="mol"&&o==="tox")?0:1,weight:1.0}));
    const x=E.sample({den,objectives:objs,gamma:3.0,tau:1.5,gate,eta:0,steps:20,clip:name==="mol"?null:1.5,x:Float32Array.from(R.noise[s]),rnd:E.rng(1)});
    chk(name+" sample("+gate+")["+s+"]",x,R["sample_"+gate][s],5e-2); } }
}
{ const {J,W}=load("mol"); const dec=new E.MolDecoder(W,J.vocab,J.L); for(let i=0;i<P.dec.z.length;i++){ const r=dec.decode(Float32Array.from(P.dec.z[i])); const same=r.smiles===P.dec.smiles[i]; if(!same) fail++; console.log(same?"PASS":"FAIL","decoder smiles["+i+"]",r.smiles,same?"":("expected "+P.dec.smiles[i])); } }
console.log(fail?("FAILURES: "+fail):"ALL PARITY CHECKS PASSED"); process.exit(fail?1:0);

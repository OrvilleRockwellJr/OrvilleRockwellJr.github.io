// Locus-conditioned guide generation vs scan-and-rank. Run: node scripts/experiments_locus.js [N]
// Fixed before looking: steps=50, eta=0, clip=1.5, gamma=4, no ensemble gate (first smoke run at N=60 showed the untilted library pull swamps efficiency, so softer weights and a tilted library were added), N per (setting, seed), seeds 0-2.
const fs=require("fs"),E=require("../engine.js"),R=__dirname+"/../";
const N=+process.argv[2]||200, SEEDS=[0,1,2], STEPS=50, GAMMA=4;
const J=JSON.parse(fs.readFileSync(R+"models/dna.json")), buf=fs.readFileSync(R+"models/dna.bin"); const W=new E.Weights(J.manifest,buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
const den=new E.Denoiser(W,"den."), eff=new E.Ensemble(W,"eff."); const judge=JSON.parse(fs.readFileSync(R+"models/judges.json")).dna; const P=JSON.parse(fs.readFileSync(R+"models/ptprs.json"));
const usable=s=>{ let gc=0; for(let i=4;i<24;i++) if(s[i]==="G"||s[i]==="C") gc++; gc/=20; return s[25]==="G"&&s[26]==="G"&&gc>=0.4&&gc<=0.7&&!s.slice(4,24).includes("TTTT"); };
let rows=[]; for(const ex of P.exons) E.findSites(ex.seq,"exon "+ex.exon,30,30+ex.len,rows); const seen=new Set(); const lib=rows.filter(r=>r.cutIn&&!seen.has(r.win)&&seen.add(r.win)).map(r=>r.win);
const set=new Set(lib); const ham=(a,b)=>{let d=0;for(let i=0;i<30;i++) if(a[i]!==b[i]) d++; return d;}; const minHam=s=>{let m=99;for(const l of lib){const d=ham(s,l); if(d<m) m=d; if(m===0) break;} return m;};
const jz=s=>E.dnaJudge(s,judge); const ok=s=>jz(s)>0&&usable(s); const ctrl=E.dnaToX; const effAll=lib.map(s=>eff.eval(ctrl(s),0,1,false).mean);
console.log("library sites (cut in exon, unique 30-mers):",lib.length);
const lsig=z=>-Math.log(1+Math.exp(-z)); const mk=b=>new E.LibCritic(lib,0.02,Float64Array.from(effAll,z=>b*lsig(z))); const LIB0=mk(0), LIB2=mk(2); const out={N,seeds:SEEDS,steps:STEPS,gamma:GAMMA,library_size:lib.length,results:[]};
const settings=[{name:"unguided",gamma:0,we:0,wl:0},{name:"efficiency only",gamma:GAMMA,we:1,wl:0},{name:"locus only",gamma:GAMMA,we:0,wl:1},{name:"efficiency + locus 0.1",gamma:GAMMA,we:1,wl:0.1},{name:"efficiency + locus 0.25",gamma:GAMMA,we:1,wl:0.25},{name:"tilted locus (beta 2)",gamma:GAMMA,we:0,wl:1,lib:"t"},{name:"efficiency + tilted locus",gamma:GAMMA,we:1,wl:1,lib:"t"}];
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length, sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(v=>(v-m)**2)));};
for(const st of settings){ const acc={exact:[],within2:[],distinct:[],success:[],judgeHit:[],usableAll:[]};
  for(const seed of SEEDS){ const seqs=[]; for(let i=0;i<N;i++){ const rnd=E.rng((seed*6151+i*92821)>>>0); const x=new Float32Array(120); for(let k=0;k<120;k++) x[k]=rnd.n();
      const objs=[]; if(st.we>0) objs.push({ens:eff,want:1,weight:st.we}); if(st.wl>0) objs.push({ens:st.lib==='t'?LIB2:LIB0,want:1,weight:st.wl});
      const xf=E.sample({den,objectives:objs,gamma:st.gamma,tau:1,gate:"none",eta:0,steps:STEPS,clip:1.5,x,rnd}); seqs.push(E.dnaFromX(xf)); }
    const ex=seqs.filter(s=>set.has(s)); acc.exact.push(ex.length/N); acc.within2.push(seqs.filter(s=>minHam(s)<=2).length/N); acc.distinct.push(new Set(ex).size/N);
    acc.success.push(seqs.filter(s=>set.has(s)&&ok(s)).length/N); acc.judgeHit.push(ex.length?mean(ex.map(s=>1/(1+Math.exp(-jz(s))))):NaN); acc.usableAll.push(seqs.filter(usable).length/N); }
  const r={setting:st.name,we:st.we,wl:st.wl}; for(const k in acc){ const v=acc[k].filter(x=>!Number.isNaN(x)); r[k]=v.length?[mean(v),sd(v)]:null; } out.results.push(r);
  console.log(st.name.padEnd(24),"exact",r.exact[0].toFixed(3),"within2",r.within2[0].toFixed(3),"distinct",r.distinct[0].toFixed(3),"on-target&judge&usable",r.success[0].toFixed(3),"judge P of exact",r.judgeHit?r.judgeHit[0].toFixed(3):"-"); }
// baselines at equal N: scan-and-rank (top-N library sites by critic) and a uniform random draw from the library
const top=lib.map((s,i)=>[effAll[i],s]).sort((a,b)=>b[0]-a[0]).slice(0,N).map(x=>x[1]);
const rndS=[]; for(const seed of SEEDS){ const rnd=E.rng(seed+99); const d=[]; for(let i=0;i<N;i++) d.push(lib[Math.floor(rnd.u()*lib.length)]); rndS.push(d.filter(ok).length/N); }
out.baselines={scan_and_rank_top_by_critic:{success:top.filter(ok).length/N,distinct:new Set(top).size/N},random_library_draw:{success:[mean(rndS),sd(rndS)]}};
console.log("baselines",JSON.stringify(out.baselines)); fs.writeFileSync(R+"results/locus_experiments.json",JSON.stringify(out,null,1));

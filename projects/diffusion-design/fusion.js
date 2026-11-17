/* PTPσ site mapping: where does a guide's cut land in the real protein? Uses the exon sequences in models/ptprs.json
   and UniProt Q13332 domain boundaries (models/ptprs_map.json). Pure lookup, no model. */
(function(root){
"use strict";
const BASE=new URL("./",(document.currentScript&&document.currentScript.src)||location.href).href;
const rc=s=>s.split("").reverse().map(c=>({A:"T",C:"G",G:"C",T:"A"}[c]||"N")).join("");
const F={ok:false,map:null,exons:null,cache:new Map()};
F.ready=Promise.all([fetch(BASE+"models/ptprs.json").then(r=>r.json()),fetch(BASE+"models/ptprs_map.json").then(r=>r.json())]).then(([p,m])=>{ F.exons=p.exons; F.map=m; F.ok=true; }).catch(()=>{});
const D1=[1393,1648], D2=[1680,1939];
function domainAt(aa){
  const m=F.map; if(!m) return null; let best=null;
  for(const f of m.features){ if(f.type!=="Domain") continue; if(aa>=f.start&&aa<=f.end){ if(!best||(f.end-f.start)<(best.end-best.start)) best=f; } }
  const comp=m.features.find(f=>f.type==="Topological domain"&&aa>=f.start&&aa<=f.end), tm=m.features.find(f=>f.type==="Transmembrane"&&aa>=f.start&&aa<=f.end);
  return {domain:best?best.desc:(tm?"Transmembrane helix":"Linker between domains"),compartment:tm?"Membrane":(comp?comp.desc:""),catalytic:aa>=D1[0]&&aa<=D1[1]?"D1":aa>=D2[0]&&aa<=D2[1]?"D2":""};
}
/* win = 30-nt context (protospacer at 4..23, PAM 25..26). Returns null when not found in a PTPRS exon. */
function siteInfo(win,exonHint){
  if(!F.ok) return null; if(F.cache.has(win)) return F.cache.get(win);
  let res=null;
  const order=F.exons.slice().sort((a,b)=>(a.exon===exonHint?-1:0)-(b.exon===exonHint?-1:0));
  for(const ex of order){
    const s=ex.seq, L=s.length;
    for(const strand of [1,-1]){
      const t=strand===1?s:rc(s), w=t.indexOf(win); if(w<0) continue;
      const cut=w+21, fwd=strand===1?cut:L-cut, p=fwd-30; if(p<0||p>ex.len) continue;
      const mrna=F.map.exonStart[ex.exon-1]+p, q=mrna-F.map.cds0;
      if(q<0){ res={exon:ex.exon,strand,region:"5′ UTR",coding:false,text:"5′ UTR of exon "+ex.exon+": a cut here does not remove coding sequence"}; break; }
      if(q>=3*F.map.protLen){ res={exon:ex.exon,strand,region:"3′ UTR",coding:false,text:"3′ UTR of exon "+ex.exon+": a cut here does not remove coding sequence"}; break; }
      const aa=Math.floor(q/3)+1, d=domainAt(aa), last=ex.exon===F.exons.length;
      let eff;
      if(d.catalytic) eff="lands inside the catalytic phosphatase domain "+d.catalytic;
      else if(aa<D1[0]) eff="a frameshift here truncates PTPσ before its catalytic domain (aa "+D1[0]+")";
      else if(aa>D2[1]) eff="C-terminal tail past the second phosphatase domain; a frameshift may leave both domains intact";
      else eff="between the phosphatase domains";
      res={exon:ex.exon,strand,aa,phase:q%3,residue:F.map.prot[aa-1],domain:d.domain,compartment:d.compartment,catalytic:d.catalytic,coding:true,lastExon:last,
        text:"exon "+ex.exon+", aa "+aa+" ("+d.domain+(d.compartment?", "+d.compartment.toLowerCase():"")+"): "+eff+(last?"; last exon, so the transcript may escape nonsense-mediated decay":"")};
      break;
    }
    if(res) break;
  }
  F.cache.set(win,res); return res;
}
F.siteInfo=siteInfo; F.domainAt=domainAt; root.PTPFusion=F;
})(window);

/* Shared: header that recedes on scroll, theme toggle. */
(function(){ const hd=document.querySelector(".top"), root=document.documentElement; let last=window.scrollY, tick=false;
  function upd(){ const y=window.scrollY, d=y-last; if(y<80){ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); } else if(d>6){ hd.classList.add("hide"); root.style.setProperty("--hh","8px"); } else if(d<-6){ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); } if(Math.abs(d)>6||y<80) last=y; tick=false; }
  window.addEventListener("scroll",()=>{ if(!tick){ tick=true; requestAnimationFrame(upd); } },{passive:true});
  hd.addEventListener("focusin",()=>{ hd.classList.remove("hide"); root.style.setProperty("--hh","58px"); });
  const btn=document.getElementById("theme"); if(!hd||!btn) return; function paint(){ const dark=root.dataset.theme==="dark"; btn.setAttribute("aria-pressed",String(dark)); btn.title=dark?"Switch to light theme":"Switch to dark theme"; const m=document.querySelector('meta[name=theme-color]'); if(m) m.content=dark?"#0B1020":"#14305A"; }
  btn.addEventListener("click",()=>{ const dark=root.dataset.theme!=="dark"; root.dataset.theme=dark?"dark":"light"; try{ localStorage.setItem("ptp-theme",root.dataset.theme); }catch(e){} paint(); document.dispatchEvent(new CustomEvent("ptp:theme")); }); paint(); })();

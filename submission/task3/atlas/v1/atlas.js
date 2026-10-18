"use strict";
(() => {
  const D = window.MICROCTURE_ATLAS;
  const host = document.getElementById("browser");
  if (!host || !D) return;
  const base = document.body.dataset.base || "";
  host.innerHTML = `<p id="region-title"></p><div class="controls"><button id="prev" aria-label="上一页">← 上一页</button><label>页码 <input id="page" type="number" min="1" max="${D.pages.length}" step="1" value="1"></label><button id="next" aria-label="下一页">下一页 →</button><label>坐标(bp) <input id="coordinate" type="number" min="0" max="${D.length-1}" step="1" value="0"></label><button id="jump">定位</button><button id="overview">交互全基因组</button><button id="fixed">固定1.05–1.10 Mb</button></div><div class="controls"><input id="gene-query" placeholder="基因名、别名或locus tag" aria-label="搜索参考基因"><button id="search">检索注释</button><span id="search-status" role="status"></span></div><div id="gene-results" class="gene-results"></div><p id="page-status" class="status" role="status"></p><div class="canvas-wrap"><canvas id="tracks" width="1160" height="760" aria-label="三个条件、作者结构及RefSeq基因轨道"></canvas></div><p class="legend">细线＝两个生物学重复；粗线＝条件均值；三条件共享y轴。信号＝sqrt(每百万接触数／有效接触对)。灰色背景＝共同无效位置。作者区间保留嵌套与原坐标；RefSeq独立注释层为0-based半开区间。轨道不是分类预测。</p><p class="download-row"><a id="page-link">独立页面</a> · <a id="original-link">下载保留的原始PNG</a> · <a href="${base}reference_genes.csv">参考基因表</a> · <a href="${base}supplied_gene_audit.csv">原表坐标核查</a></p><details id="original"><summary>查看保留的原始PNG（作者补充表注释版）</summary><p class="notice">原图保留原作者基因坐标和按页绘图尺度，未重绘为RefSeq版本。其基因端点与本页默认RefSeq轨不能混用；多数精确匹配原表行符合1-based闭区间，原图未减1。所有465原图完整保留，不以此认证作者结构坐标。</p><img id="source-image" alt="原始作者注释轨道" loading="lazy"></details>`;
  const $ = id => document.getElementById(id);
  const colors = ["#2463a7", "#cf8427", "#a53e6b"];
  const canvas = $("tracks"), ctx = canvas.getContext("2d");
  let page = 0, range = [0, D.pageWidth];
  const fmt = value => Math.round(value).toLocaleString("en-US");
  function bins(lo, hi) {
    const first = Math.floor(lo / D.step), stop = Math.min(D.signals[0].length, Math.ceil(hi / D.step));
    const stride = Math.max(1, Math.ceil((stop-first)/1100));
    const result = [];
    for (let i=first; i<stop; i+=stride) {
      const stopBin=Math.min(stop, i+stride), values=D.signals.map(y => {
        let sum=0,n=0; for(let k=i;k<stopBin;k++){if(y[k]!==null){sum+=y[k];n++;}}
        return n ? sum/n : null;
      });
      result.push({x:(i+stopBin-1)/2*D.step,y:values});
    }
    return result;
  }
  function render(lo, hi, label) {
    range=[lo,hi]; const W=1160, H=760, left=125, right=1130, width=right-left;
    ctx.clearRect(0,0,W,H);ctx.fillStyle="white";ctx.fillRect(0,0,W,H);
    const points=bins(lo,hi), peak=Math.max(0,...points.flatMap(p=>p.y.filter(v=>v!==null).map(v=>Math.sqrt(v))));
    const ymax=peak>0?peak*1.08:1, x = value => left+(value-lo)/(hi-lo)*width;
    ctx.font="12px sans-serif";
    for(let j=0;j<3;j++) {
      const top=45+j*145,bottom=top+112;
      ctx.fillStyle="#18314c";ctx.fillText(D.labels[j],12,top+42);
      for(let tick=0;tick<=2;tick++) {
        const y=bottom-tick/2*112;ctx.strokeStyle="#e7edf3";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
        ctx.fillStyle="#52657a";ctx.fillText((ymax*tick/2).toPrecision(3),left-54,y+4);
      }
      for(let k=0;k<3;k++) {
        ctx.strokeStyle=colors[j];ctx.globalAlpha=k===2?1:.35;ctx.lineWidth=k===2?1.6:.7;ctx.beginPath();let active=false;
        for(const point of points) {
          const pair=point.y.slice(j*2,j*2+2);
          const v=k===2?(pair.every(v=>v!==null)?(pair[0]+pair[1])/2:null):pair[k];
          if(v===null){active=false;continue;}
          const px=x(point.x),py=bottom-Math.sqrt(v)/ymax*112;
          if(active)ctx.lineTo(px,py);else ctx.moveTo(px,py);active=true;
        }
        ctx.stroke();ctx.globalAlpha=1;
      }
    }
    ctx.fillStyle="#18314c";ctx.fillText("Author structures",12,505);
    const kinds=["OPCID","CHIN","CHID"];
    kinds.forEach((kind,j)=>{ctx.fillStyle="#52657a";ctx.fillText(kind,left-53,490+j*24);});
    D.structures.filter(r=>r.start<hi&&r.end>lo).forEach(r=>{
      const j=kinds.indexOf(r.type),a=x(Math.max(lo,r.start)),z=x(Math.min(hi,r.end));
      ctx.fillStyle=colors[j];ctx.globalAlpha=.65;ctx.fillRect(a,480+j*24,Math.max(1,z-a),11);ctx.globalAlpha=1;
      if(hi-lo<=50000&&z-a>35){ctx.font="9px sans-serif";ctx.fillText(r.id,a,477+j*24);ctx.font="12px sans-serif";}
    });
    ctx.fillStyle="#18314c";ctx.fillText("RefSeq genes",12,602);ctx.fillText("+",left-20,581);ctx.fillText("−",left-20,627);
    let last=[-1000,-1000];
    D.genes.filter(r=>r.start<hi&&r.end>lo).forEach(r=>{
      const j=r.strand==="+"?0:1,a=x(Math.max(lo,r.start)),z=x(Math.min(hi,r.end)),y=570+j*46;
      ctx.fillStyle="#7793a5";ctx.fillRect(a,y,Math.max(1,z-a),10);
      if(hi-lo<=50000){ctx.strokeStyle="#26475d";ctx.beginPath();const end=j===0?z:a,dir=j===0?1:-1;ctx.moveTo(end-dir*4,y+1);ctx.lineTo(end,y+5);ctx.lineTo(end-dir*4,y+9);ctx.stroke();
        if(a>last[j]+6){ctx.fillStyle="#26475d";ctx.font="10px sans-serif";ctx.fillText(r.gene,a,y+26);last[j]=a+ctx.measureText(r.gene).width;ctx.font="12px sans-serif";}
      }
    });
    ctx.fillStyle="#52657a";ctx.fillText("Genome coordinate (bp), 0-based half-open",left,714);
    for(let j=0;j<=6;j++){const value=lo+(hi-lo)*j/6;ctx.textAlign=j===6?"right":"left";ctx.fillText(fmt(value),x(value),690);}ctx.textAlign="left";
    // Common invalidity is shown in a separate QC strip, never bridged by a line.
    ctx.fillStyle="#52657a";ctx.fillText("Common QC",12,660);
    const first=Math.floor(lo/D.step),stop=Math.min(D.signals[0].length,Math.ceil(hi/D.step));
    ctx.fillStyle="#e2e8f0";
    for(let i=first;i<stop;i++){if(D.signals[0][i]===null)ctx.fillRect(x(Math.max(lo,i*D.step)),650,Math.max(1,width*D.step/(hi-lo)),10);}
    $("region-title").textContent=`${label} · ${fmt(lo)}–${fmt(hi)} bp`;
    $("page-status").textContent=`${points.length}显示点 · 原数值100 bp · ${D.genes.filter(r=>r.start<hi&&r.end>lo).length}条RefSeq基因／假基因 · ${D.structures.filter(r=>r.start<hi&&r.end>lo).length}条作者区间（端点约定暂定）`;
  }
  function go(index) {
    page=Math.max(0,Math.min(D.pages.length-1,Math.trunc(index)||0)); const r=D.pages[page];
    $("page").value=page+1;$("coordinate").value=r.start;
    $("prev").disabled=page===0;$("next").disabled=page===D.pages.length-1;
    $("page-link").href=base+r.html;$("original-link").href=base+r.file;
    // Do not load all 465 PNGs; this selected historical image is lazy-loaded.
    $("source-image").src=base+r.file;
    render(+r.start,+r.end,`Level 2 / ${page+1} of ${D.pages.length}`);
  }
  $("prev").onclick=()=>go(page-1);$("next").onclick=()=>go(page+1);
  $("page").onchange=()=>go(Number($("page").value)-1);
  $("jump").onclick=()=>go(Math.floor(Number($("coordinate").value)/D.pageWidth));
  $("overview").onclick=()=>render(0,D.length,"Level 1 / Whole genome");
  $("fixed").onclick=()=>render(...D.fixed,"Level 3 / Predefined region");
  $("search").onclick=()=>{
    const query=$("gene-query").value.trim().toLowerCase();$("gene-results").replaceChildren();
    const hits=query?D.genes.filter(r=>[r.gene,r.locus_tag,...r.aliases.split(",")].some(n=>n.toLowerCase()===query)):[];
    $("search-status").textContent=hits.length?`${hits.length}条参考记录，所有同名／别名位置均列出。`:"未找到精确基因名／别名／locus tag；未自动选择近似名称。";
    hits.forEach(r=>{const button=document.createElement("button");button.textContent=`${r.gene} / ${r.locus_tag} · ${fmt(r.start)}–${fmt(r.end)} (${r.strand})`;button.onclick=()=>go(Math.floor(r.start/D.pageWidth));$("gene-results").appendChild(button);});
  };
  $("gene-query").onkeydown=e=>{if(e.key==="Enter")$("search").click();};
  document.querySelectorAll("[data-page-link]").forEach(a=>a.addEventListener("click",e=>{e.preventDefault();go(+a.dataset.pageLink);host.scrollIntoView({block:"start"});}));
  document.addEventListener("keydown",e=>{if(["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName))return;if(e.key==="ArrowLeft")go(page-1);if(e.key==="ArrowRight")go(page+1);});
  const param=new URLSearchParams(window.location.search).get("page");
  go(param===null?Number(document.body.dataset.page||0):Number(param));
  window.MicroCTureAtlas={go,render,bins,getRange:()=>range.slice()};
})();

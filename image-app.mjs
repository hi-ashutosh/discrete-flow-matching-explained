import {schedule} from './model.mjs';
import {IMAGE_END,makeDataset,imagePosterior,pixelBeliefs,pixelRate,nearestDistance,simulateImagesFlow,simulateImagesDiffusion} from './image-model.mjs';
const $=s=>document.querySelector(s),flow=document.body.dataset.kind==='flow';
const catalog=await (await fetch('digits.json')).json();
const config={source:'uniform',selection:'all',firstShare:.1,schedule:'linear',steps:40,count:24,seed:41};
let dataset=makeDataset(catalog),frames=[],frame=0,chosen=0,pixel=27,timer=null,forward=false;
function image(x,label='Binary pixel image'){
 return `<svg class="pixel-art" viewBox="0 0 8 8" role="img" aria-label="${label}"><rect width="8" height="8" fill="white"/>${x.map((v,i)=>v===0?'':`<rect x="${i%8}" y="${Math.floor(i/8)}" width="1" height="1" fill="${v===-1?'#b5b9b7':'#202d32'}"/>`).join('')}</svg>`;
}
function stop(){if(timer)clearInterval(timer);timer=null;$('#image-play').textContent='▶ Play';$('#image-play').setAttribute('aria-pressed','false');}
function phase(){if(flow){const t=frame/100*IMAGE_END;return {a:schedule(t,config.schedule)[0],t,k:null};}const k=forward?frame:config.steps-frame;return {a:1-k/config.steps,t:null,k};}
function rebuild(reset=false){
 stop();dataset=makeDataset(catalog,config.selection,config.firstShare);
 const start=performance.now();
 frames=flow?simulateImagesFlow(config,dataset):simulateImagesDiffusion(config,dataset,forward);
 frame=reset?0:Math.min(frame,frames.length-1);$('#image-time').max=frames.length-1;
 $('#image-build').textContent=`${config.count} independent stochastic samples · ${(performance.now()-start).toFixed(0)} ms to simulate`;
 $('#image-targets').innerHTML=dataset.labels.map(label=>{const z=dataset.items.find(z=>z.label===label);return `<div>${image(z.pixels,`Example digit ${label}`)}<span>${label} · ${Math.round((label===dataset.labels[0]?config.firstShare:(1-config.firstShare)/(dataset.labels.length-1))*100)}%</span></div>`;}).join('');
 $('#image-first-label').textContent=`Digit ${dataset.labels[0]} target share`;
 render();
}
function render(){
 const {a,t,k}=phase(),population=frames[frame],x=population[chosen];
 const weights=imagePosterior(x,a,dataset,config.source),beliefs=pixelBeliefs(weights,dataset);
 $('#image-time').value=frame;$('#image-clock').textContent=flow?`t = ${t.toFixed(3)}`:`k = ${k} / ${config.steps}`;
 $('#image-grid').innerHTML=x.map((v,i)=>`<button class="image-pixel ${v===1?'black':v===-1?'masked':''} ${i===pixel?'chosen':''}" data-pixel="${i}" aria-label="Pixel row ${Math.floor(i/8)+1} column ${i%8+1}: ${v===1?'black':v===0?'white':'masked'}" aria-pressed="${i===pixel}"></button>`).join('');
 $('#image-gallery').innerHTML=population.map((p,i)=>`<button class="gallery-sample ${i===chosen?'chosen':''}" data-sample="${i}" aria-label="Follow generated sample ${i+1}" aria-pressed="${i===chosen}">${image(p,`Sample ${i+1}`)}</button>`).join('');
 $('#image-sample-label').textContent=`Sample ${chosen+1} of ${config.count}`;
 $('#image-gallery-caption').textContent='Click any sample to follow it. Each starts independently; no destination digit is selected in advance.';
 $('#image-time-start').textContent=flow?'t = 0 · source':forward?'k = 0 · data':`k = ${config.steps} · source`;
 $('#image-time-end').textContent=flow?'t = 0.999 · near target':forward?`k = ${config.steps} · source`:'k = 0 · generated data';
 $('#image-current-pixel').textContent=`Row ${Math.floor(pixel/8)+1}, column ${pixel%8+1} · ${x[pixel]===-1?'masked':x[pixel]===1?'black':'white'}`;
 $('#pixel-black').style.width=(100*beliefs[pixel])+'%';$('#pixel-probability').textContent=`P(clean pixel is black | current image) = ${(100*beliefs[pixel]).toFixed(1)}%`;
 if(flow){const r=pixelRate(x,pixel,t,config,dataset);$('#pixel-transition').textContent=config.source==='mask'?`Reveal black: ${r.black.toFixed(2)} / time\nReveal white: ${r.white.toFixed(2)} / time`:`Flip to ${x[pixel]===1?'white':'black'}: ${(r.black+r.white).toFixed(2)} / time\nSmall-step probability ≈ rate × Δt`;}else{
  if(k===0)$('#pixel-transition').textContent=forward?'No corruption has been applied.':'Generation has finished.';
  else {
   const b=1-(k-1)/config.steps,r=a/b;
   let nextBlack=0,nextMask=0;
   if(config.source==='mask'){
    if(x[pixel]===-1){const reveal=(b-a)/(1-a);nextBlack=reveal*beliefs[pixel];nextMask=1-reveal;}
    else nextBlack=x[pixel];
   }else for(let j=0;j<weights.length;j++){
    const z=dataset.items[j].pixels[pixel];
    nextBlack+=weights[j]*((1-r)/2+r*Number(x[pixel]===1))*((1-b)/2+b*z)/((1-a)/2+a*Number(x[pixel]===z));
   }
   $('#pixel-transition').textContent=`ONE reverse step at this noise level:\nBlack ${(100*nextBlack).toFixed(1)}% · white ${(100*(1-nextBlack-nextMask)).toFixed(1)}%${config.source==='mask'?` · mask ${(100*nextMask).toFixed(1)}%`:''}`;
  }
 }
 const ranked=weights.map((w,i)=>({w,i})).sort((a,b)=>b.w-a.w).slice(0,3);
 $('#image-candidates').innerHTML=ranked.map(({w,i})=>`<div>${image(dataset.items[i].pixels,`Possible endpoint digit ${dataset.items[i].label}`)}<span>${(100*w).toFixed(1)}%</span></div>`).join('');
 const distance=population.reduce((s,p)=>s+nearestDistance(p,dataset),0)/population.length;
 $('#image-distance').textContent=distance.toFixed(1);$('#image-match').textContent=population.filter(p=>nearestDistance(p,dataset)===0).length+'/'+population.length;
 $('#image-source-weight').textContent=(100*(1-a)).toFixed(1)+'%';
 const indices=Array.from({length:6},(_,i)=>Math.round(frame*i/5));
 $('#image-trace').innerHTML=indices.map((f,i)=>`<div>${image(frames[f][chosen],`Selected sample at frame ${f}`)}<span>${flow?(f/100*IMAGE_END).toFixed(2):forward?f:config.steps-f}</span></div>${i<5?'<span class="image-arrow" aria-hidden="true">→</span>':''}`).join('');
 let explanation=flow?(frame===0?(config.source==='mask'?'Every pixel starts masked. This source distribution puts all its mass on one masked image.':'The source is a distribution of independent random pixels. The model has not been given a clean destination.'):`At this time, samples follow the ${config.source==='mask'?'mask-reveal':'uniform-to-data'} probability path. Each update consults the whole current image, then samples categorical pixel changes.`):(forward?'Watch the known corruption process erase the data. The pixel inspector shows what a reverse model would predict at this same noise level.':'At each step, use the current image to infer possible clean endpoints, then sample a less corrupted image. The endpoint posterior is recomputed at every step.');
 if(flow&&frame===100)explanation+=' We stop just before t = 1; residual errors are expected. Uniform-source sampling also has finite-step approximation error. There is no nearest-digit correction.';
 $('#image-explanation').textContent=explanation;
 $('#image-source-name').textContent=config.source==='mask'?'All pixels masked':'Independent fair-coin pixels';
}
$('#image-time').addEventListener('input',e=>{stop();frame=Number(e.target.value);render();});
$('#image-play').addEventListener('click',()=>{if(timer){stop();return;}if(frame===frames.length-1)frame=0;$('#image-play').textContent='Ⅱ Pause';$('#image-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{frame++;render();if(frame>=frames.length-1)stop();},flow?85:180);});
$('#image-step').addEventListener('click',()=>{stop();frame=Math.min(frames.length-1,frame+1);render();});
$('#image-restart').addEventListener('click',()=>{stop();frame=0;render();});
$('#image-new').addEventListener('click',()=>{config.seed++;rebuild();});
$('#image-source').addEventListener('change',e=>{config.source=e.target.value;rebuild(true);});
$('#image-selection').addEventListener('change',e=>{config.selection=e.target.value;config.firstShare=e.target.value==='all'?.1:.5;$('#image-share').value=config.firstShare*100;$('#image-share-value').textContent=Math.round(config.firstShare*100)+'%';rebuild(true);});
$('#image-share').addEventListener('change',e=>{config.firstShare=Number(e.target.value)/100;$('#image-share-value').textContent=e.target.value+'%';rebuild();});
if(flow)$('#image-schedule').addEventListener('change',e=>{config.schedule=e.target.value;rebuild();});
else {$('#image-steps').addEventListener('change',e=>{config.steps=Number(e.target.value);rebuild(true);});$('#image-direction').addEventListener('change',e=>{forward=e.target.value==='forward';rebuild(true);});}
$('#image-grid').addEventListener('click',e=>{const p=e.target.closest('[data-pixel]');if(p){pixel=Number(p.dataset.pixel);render();}});
$('#image-gallery').addEventListener('click',e=>{const p=e.target.closest('[data-sample]');if(p){chosen=Number(p.dataset.sample);render();}});
$('#image-present').addEventListener('click',()=>{const on=document.body.classList.toggle('image-presentation');$('#image-present').textContent=on?'Exit presentation':'Presentation view';$('#image-present').setAttribute('aria-pressed',String(on));if(on)window.scrollTo({top:0});});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('image-presentation'))$('#image-present').click();});
$('#open-toy').addEventListener('click',()=>{$('#toy-lab').open=true;});
rebuild(true);

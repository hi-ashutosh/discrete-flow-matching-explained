import {STATES,END,patterns,bit,data,schedule,marginal,posterior,Denoiser,rates,reverseKernel,histogram,tv,simulateFlow,simulateDiffusion} from './model.mjs';
const $=s=>document.querySelector(s), flow=document.body.dataset.kind==='flow';
const config={pattern:'solid',weight:.65,schedule:'linear',steps:20,seed:7,count:192};
let model=new Denoiser(), useLearned=false, forward=false, frames=[], position=0, playing=null, selected=0, totalEpochs=0;
function tile(x,size=''){return `<span class="pattern ${size}" role="img" aria-label="${[0,1,2,3].map(i=>bit(x,i)).join('')}">${[0,1,2,3].map(i=>`<span class="tile ${bit(x,i)?'one':''}"></span>`).join('')}</span>`;}
function stop(){if(playing)clearInterval(playing);playing=null;$('#play').textContent='▶ Play';$('#play').setAttribute('aria-pressed','false');}
function resetModel(){model=new Denoiser();totalEpochs=0;updateLearning();}
function rebuild(keepPosition=true){stop();const old=position;frames=flow?simulateFlow(config,useLearned?model:null):simulateDiffusion(config,useLearned?model:null,forward);position=keepPosition?Math.min(old,frames.length-1):0;$('#time').max=frames.length-1;$('#time').value=position;render();}
function phase(){if(flow){const t=position/100*END;return {a:schedule(t,config.schedule)[0],t,k:null};}const k=forward?position:config.steps-position;return {a:1-k/config.steps,t:null,k};}
function render(){const {a,t,k}=phase(), samples=frames[position], empirical=histogram(samples), exact=marginal(a,config), target=data(config), first=samples[0];
 $('#time').value=position;$('#time-value').textContent=flow?`t = ${t.toFixed(3)}`:`k = ${k} / ${config.steps}`;
 $('#population').innerHTML=samples.slice(0,72).map(x=>tile(x)).join('');
 $('#population-caption').textContent=`72 of ${config.count} simulated patterns · outlined pattern is followed below`;
 $('#mode-tag').textContent=!flow&&forward?'FIXED CORRUPTION · NO MODEL':useLearned?'TRAINABLE TABLE · APPROXIMATE':'EXACT POSTERIOR · ORACLE';
 $('#oracle').disabled=!flow&&forward; $('#learned').disabled=!flow&&forward;
 $('#sim-title').textContent=flow?'Noise → structured patterns':forward?'Corrupting data → noise':'Generating noise → data';
 $('#time-start').textContent=flow?'t = 0 · random tiles':forward?'k = 0 · clean data':`k = ${config.steps} · random tiles`;
 $('#time-end').textContent=flow?'t = 0.995 · almost data':forward?`k = ${config.steps} · pure noise`:'k = 0 · generated data';
 const indices=Array.from({length:9},(_,i)=>Math.round(i*position/8));
 $('#trace').innerHTML='<span class="trace-label">One sample<br>through time</span>'+indices.map((f,i)=>(i?'<span class="arrow" aria-hidden="true">→</span>':'')+tile(frames[f][3])).join('');
 $('#histogram').innerHTML=STATES.map(x=>`<button class="bar-group" aria-label="Inspect state ${x.toString(2).padStart(4,'0')}: simulated ${(100*empirical[x]).toFixed(1)} percent, exact ${(100*exact[x]).toFixed(1)} percent" aria-pressed="${selected===x}" data-state="${x}"><span class="bar-space"><span class="bar-actual" style="height:${empirical[x]*100}%"></span><span class="bar-theory" style="height:${exact[x]*100}%"></span></span>${tile(x)}<span class="bar-label">${x.toString(2).padStart(4,'0')}</span></button>`).join('');
 $('#metric-tv').textContent=tv(empirical,exact).toFixed(3);$('#metric-target').textContent=(100*samples.filter(x=>target[x]>0).length/config.count).toFixed(1)+'%';$('#metric-noise').textContent=(100*(1-a)).toFixed(1)+'%';
 $('#pair').innerHTML=tile(patterns[config.pattern][0],'medium')+`<span>${Math.round(config.weight*100)}%</span>`+tile(patterns[config.pattern][1],'medium')+`<span>${Math.round((1-config.weight)*100)}%</span>`;
 let text=flow?(position===0?'Every tile starts as a fair coin flip. All 16 patterns are equally likely.':position===100?'Almost at the endpoint: most probability has reached the two target patterns. We stop at 0.995 because the jump-rate formula is singular at t = 1. Small errors also come from finite steps and only 192 samples.':'Each pattern can jump by changing one tile. The model uses the whole current pattern to decide which change is likely. The bars show the population becoming structured.'):(forward?'Corruption forgets the data on purpose. At the final step, every tile is a fair coin flip, independent of the original pattern.':'The denoiser predicts probabilities for clean patterns. A reverse transition uses those probabilities to sample an earlier, less noisy state. It never sees the original pattern.');
 if(useLearned&&!(forward&&!flow))text+=` You are using the trainable table after ${totalEpochs} epochs; any mismatch also includes model error.`;
 $('#live-explanation').textContent=text;$('#state-select').value=selected;updateInspector();
}
function updateInspector(){const {a,t,k}=phase();const w=posterior(selected,a,config);$('#inspector-state').innerHTML=tile(selected,'big');if(flow){const evalT=Math.min(t,END), rr=rates(selected,evalT,config,useLearned?model:null);$('#inspector-values').textContent=`At t = ${t.toFixed(3)}, state ${selected.toString(2).padStart(4,'0')}\nExact P(first clean pattern | current) = ${w.toFixed(4)}\n${rr.map((v,i)=>`Tile ${i+1}: flip rate ${v.toFixed(3)} / unit time`).join('\n')}\nRate × a small Δt ≈ probability of that jump.`;}else{if(k===0){$('#inspector-values').textContent=`At k = 0, state ${selected.toString(2).padStart(4,'0')}\nThe reverse chain has finished. There is no next denoising step.\nExact clean-state probability = ${marginal(1,config)[selected].toFixed(4)}`;}else{const r=reverseKernel(selected,k,config,useLearned?model:null);const top=STATES.slice().sort((a,b)=>r[b]-r[a]).slice(0,4);$('#inspector-values').textContent=`At k = ${k}, state ${selected.toString(2).padStart(4,'0')}\nExact P(first clean pattern | current) = ${w.toFixed(4)}\nMost likely states after ONE reverse step:\n${top.map(x=>`${x.toString(2).padStart(4,'0')}: ${(100*r[x]).toFixed(2)}%`).join('\n')}`;}}}
function updateLearning(){const loss=model.loss(config);$('#loss-value').textContent=loss.toFixed(4);$('#epochs').textContent=totalEpochs;$('#training-note').textContent=totalEpochs?`Trained for ${totalEpochs} full-batch epochs. Switch to “Trainable” in the explorer to sample with this model.`:'The table starts with 50–50 predictions. Train it on noisy patterns paired with their clean labels.';const history=model.history.length?model.history:[loss];const points=history.map((v,i)=>`${30+i/Math.max(1,history.length-1)*420},${120-v/Math.max(.75,...history)*100}`).join(' ');$('#loss-plot').innerHTML=`<line x1="30" y1="120" x2="455" y2="120" stroke="#cdd3cd"/><line x1="30" y1="15" x2="30" y2="120" stroke="#cdd3cd"/><polyline points="${points}" fill="none" stroke="var(--accent)" stroke-width="2.5"/><text x="35" y="138" fill="#566269" font-size="10">training epochs →</text><text x="36" y="12" fill="#566269" font-size="10">cross-entropy (lower is better)</text>`;}
$('#pattern').addEventListener('change',e=>{config.pattern=e.target.value;resetModel();rebuild();});
$('#weight').addEventListener('input',e=>{config.weight=Number(e.target.value)/100;$('#weight-value').textContent=e.target.value+'%';resetModel();rebuild();});
if(flow)$('#schedule').addEventListener('change',e=>{config.schedule=e.target.value;rebuild();});
else {$('#steps').addEventListener('change',e=>{config.steps=Number(e.target.value);rebuild(false);});$('#direction').addEventListener('change',e=>{forward=e.target.value==='forward';rebuild(false);});}
$('#oracle').addEventListener('click',()=>{useLearned=false;$('#oracle').setAttribute('aria-pressed','true');$('#learned').setAttribute('aria-pressed','false');rebuild();});
$('#learned').addEventListener('click',()=>{useLearned=true;$('#oracle').setAttribute('aria-pressed','false');$('#learned').setAttribute('aria-pressed','true');rebuild();});
$('#time').addEventListener('input',e=>{stop();position=Number(e.target.value);render();});
$('#play').addEventListener('click',()=>{if(playing){stop();return;}if(position===frames.length-1)position=0;$('#play').textContent='Ⅱ Pause';$('#play').setAttribute('aria-pressed','true');playing=setInterval(()=>{position++;render();if(position>=frames.length-1)stop();},flow?90:350);});
$('#step').addEventListener('click',()=>{stop();position=Math.min(position+1,frames.length-1);render();});
$('#restart').addEventListener('click',()=>{stop();position=0;render();});
$('#resample').addEventListener('click',()=>{config.seed++;rebuild();});
$('#histogram').addEventListener('click',e=>{const b=e.target.closest('[data-state]');if(b){selected=Number(b.dataset.state);render();}});
$('#state-select').innerHTML=STATES.map(x=>`<option value="${x}">${x.toString(2).padStart(4,'0')}</option>`).join('');
$('#state-select').addEventListener('change',e=>{selected=Number(e.target.value);render();});
$('#train').addEventListener('click',()=>{model.train(config,40);totalEpochs+=40;updateLearning();if(useLearned)rebuild();});
$('#reset-model').addEventListener('click',()=>{resetModel();if(useLearned)rebuild();});
$('#present').addEventListener('click',()=>{const on=document.body.classList.toggle('presentation');$('#present').textContent=on?'Exit presentation':'Presentation view';$('#present').setAttribute('aria-pressed',String(on));if(on)window.scrollTo({top:0});});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('presentation'))$('#present').click();});
updateLearning();rebuild(false);

/** Binary 8x8 image generation over a small finite empirical distribution.
 * Exact endpoint posterior; no neural model and no chosen clean destination.
 * Pixels: 0=white, 1=black, -1=masked. Images remain categorical throughout.
 */
import {rng,categorical,schedule} from './model.mjs';
export const IMAGE_END=0.999;
export function makeDataset(catalog,selection='all',firstShare=.1){
 const labels=selection==='all'?[0,1,2,3,4,5,6,7,8,9]:selection==='binary'?[0,1]:[2,7];
 const items=catalog.filter(z=>labels.includes(z.label));
 const weights=items.map(z=>(z.label===labels[0]?firstShare:(1-firstShare)/(labels.length-1))/items.filter(q=>q.label===z.label).length);
 return {items,weights,labels};
}
export function imagePosterior(x,a,dataset,source='uniform'){
 const logs=dataset.items.map((z,j)=>{
  let distance=0;
  for(let i=0;i<64;i++){
   if(x[i]===-1)continue;
   if(x[i]!==z.pixels[i])distance++;
  }
  if(source==='mask')return distance===0?Math.log(dataset.weights[j]):-Infinity;
  return Math.log(dataset.weights[j])+(64-distance)*Math.log((1+a)/2)+distance*Math.log(Math.max(1e-300,(1-a)/2));
 });
 const max=Math.max(...logs);
 if(!Number.isFinite(max))throw new Error('Observed pixels are inconsistent with the finite dataset.');
 const weights=logs.map(v=>Math.exp(v-max)),sum=weights.reduce((a,b)=>a+b,0);
 return weights.map(v=>v/sum);
}
export function pixelBeliefs(posterior,dataset){
 const result=Array(64).fill(0);
 for(let j=0;j<dataset.items.length;j++)for(let i=0;i<64;i++)result[i]+=posterior[j]*dataset.items[j].pixels[i];
 return result;
}
export function pixelRate(x,i,t,config,dataset){
 const [a,da]=schedule(t,config.schedule),w=imagePosterior(x,a,dataset,config.source),p=pixelBeliefs(w,dataset)[i];
 if(x[i]===-1)return {black:da/(1-a)*p,white:da/(1-a)*(1-p)};
 if(config.source==='mask')return {black:0,white:0};
 return {black:x[i]===0?da/(1-a)*p:0,white:x[i]===1?da/(1-a)*(1-p):0};
}
export function nearestDistance(x,dataset){
 return Math.min(...dataset.items.map(z=>x.reduce((s,v,i)=>s+Number(v!==z.pixels[i]),0)));
}
function initialImages(config,dataset,random,forward){
 return Array.from({length:config.count},()=>forward?dataset.items[categorical(dataset.weights,random)].pixels.slice():Array.from({length:64},()=>config.source==='mask'?-1:Number(random()<.5)));
}
/** CTMC for masked path is sampled exactly by independent reveal clocks.
 * Uniform path uses coordinate-parallel tau-leaping, an approximation. The
 * largest per-coordinate event probability is <=0.08; simultaneous flips use
 * beliefs from the same pre-step state. No final nearest-image snap is applied.
 */
export function simulateImagesFlow(config,dataset){
 const random=rng(config.seed),samples=initialImages(config,dataset,random,false);
 const frames=[samples.map(x=>x.slice())];
 const clocks=config.source==='mask'?samples.map(()=>Array.from({length:64},(_,i)=>({i,at:random()})).sort((a,b)=>a.at-b.at)):null;
 const cursors=samples.map(()=>0);
 let t=0;
 for(let f=1;f<=100;f++){
  const end=f/100*IMAGE_END;
  if(clocks){
   const signal=schedule(end,config.schedule)[0];
   for(let j=0;j<samples.length;j++){
    while(cursors[j]<64&&clocks[j][cursors[j]].at<=signal){
     const {i,at}=clocks[j][cursors[j]++];
     const weights=imagePosterior(samples[j],at,dataset,'mask');
     const p=pixelBeliefs(weights,dataset)[i];
     samples[j][i]=Number(random()<p);
    }
   }
  }else{
   while(t<end-1e-12){
    const [a,da]=schedule(t,config.schedule);
    const dt=Math.min(end-t,1/240,.08*(1-a)/Math.max(da,1e-12));
    const factor=dt*da/(1-a);
    for(const x of samples){
     const probs=pixelBeliefs(imagePosterior(x,a,dataset,'uniform'),dataset);
     for(let i=0;i<64;i++)if(random()<factor*(x[i]===0?probs[i]:1-probs[i]))x[i]=1-x[i];
    }
    t+=dt;
   }
  }
  frames.push(samples.map(x=>x.slice()));
 }
 return frames;
}
/** Exact finite-step reverse sampling: draw a fresh endpoint posterior sample
 * at EACH step and then its factorized conditional bridge. The latent endpoint
 * is integrated out by this sampling; no endpoint is fixed across the run.
 */
export function reverseImageStep(x,k,config,dataset,random){
 const a=1-k/config.steps,b=1-(k-1)/config.steps;
 const weights=imagePosterior(x,a,dataset,config.source);
 const z=dataset.items[categorical(weights,random)].pixels;
 if(config.source==='mask')return x.map((v,i)=>v!==-1?v:random()<(b-a)/(1-a)?z[i]:-1);
 const r=a/b;
 return x.map((v,i)=>{
  const numerator=((1-r)/2+r*Number(v===1))*((1-b)/2+b*z[i]);
  const denominator=(1-a)/2+a*Number(v===z[i]);
  return Number(random()<numerator/denominator);
 });
}
export function simulateImagesDiffusion(config,dataset,forward=false){
 const random=rng(config.seed),samples=initialImages(config,dataset,random,forward),frames=[samples.map(x=>x.slice())];
 for(let f=1;f<=config.steps;f++){
  const k=forward?f:config.steps-f+1;
  for(let j=0;j<samples.length;j++){
   if(!forward){samples[j]=reverseImageStep(samples[j],k,config,dataset,random);continue;}
   const ratio=(1-k/config.steps)/(1-(k-1)/config.steps);
   samples[j]=samples[j].map(v=>config.source==='mask'?(v===-1||random()>ratio?-1:v):(random()>ratio?Number(random()<.5):v));
  }
  frames.push(samples.map(x=>x.slice()));
 }
 return frames;
}

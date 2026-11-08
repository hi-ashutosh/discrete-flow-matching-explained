import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {makeDataset,imagePosterior,pixelBeliefs,pixelRate,nearestDistance,simulateImagesFlow,simulateImagesDiffusion} from '../image-model.mjs';
const catalog=JSON.parse(readFileSync(new URL('../digits.json',import.meta.url)));
const data=makeDataset(catalog);
const config={source:'uniform',schedule:'linear',steps:40,count:96,seed:41};
const near=(a,b,e=1e-9)=>assert.ok(Math.abs(a-b)<e,`${a} vs ${b}`);
test('image dataset is binary, distinct and normalized for each target selection',()=>{
 assert.equal(catalog.length,40);assert.equal(new Set(catalog.map(z=>z.pixels.join(''))).size,40);
 for(const z of catalog){assert.equal(z.pixels.length,64);assert.ok(z.pixels.every(v=>v===0||v===1));}
 for(const selection of ['all','binary','pair']){const d=makeDataset(catalog,selection,.65);near(d.weights.reduce((a,b)=>a+b),1);near(d.items.reduce((s,z,j)=>s+(z.label===d.labels[0]?d.weights[j]:0),0),.65);}
});
test('uninformative source yields target priors; visible masked pixels exclude incompatible endpoints',()=>{
 for(const source of ['uniform','mask']){const x=Array(64).fill(source==='mask'?-1:0);const w=imagePosterior(x,0,data,source);w.forEach((v,i)=>near(v,data.weights[i]));}
 const x=Array(64).fill(-1);x[20]=1;const w=imagePosterior(x,.4,data,'mask');near(w.reduce((a,b)=>a+b),1);w.forEach((v,j)=>{if(data.items[j].pixels[20]===0)near(v,0);});
 near(pixelBeliefs(w,data)[20],1);
});
test('posterior remains finite close to endpoint; rates are nonnegative',()=>{
 const x=Array.from({length:64},(_,i)=>i%2);const w=imagePosterior(x,.99999,data);near(w.reduce((a,b)=>a+b),1);assert.ok(w.every(v=>Number.isFinite(v)&&v>=0));
 for(let i=0;i<64;i++){const r=pixelRate(x,i,.7,config,data);assert.ok(r.black>=0&&r.white>=0);}
});
test('exact finite-step diffusion recovers dataset support for both source types',()=>{
 for(const source of ['uniform','mask']){const f=simulateImagesDiffusion({...config,source},data);assert.equal(f.length,41);assert.ok(f.at(-1).every(x=>nearestDistance(x,data)===0));assert.deepEqual(f,simulateImagesDiffusion({...config,source},data));}
 const d=makeDataset(catalog,'binary',.8),f=simulateImagesDiffusion({...config,count:256},d);const zeros=f.at(-1).filter(x=>d.items.find(z=>z.pixels.every((v,i)=>v===x[i])).label===0).length/256;near(zeros,.8,.09);
});
test('flow produces coherent images without a forced endpoint assignment',()=>{
 for(const source of ['uniform','mask']){const f=simulateImagesFlow({...config,source},data);assert.equal(f.length,101);const mean=f.at(-1).reduce((s,x)=>s+nearestDistance(x,data),0)/config.count;assert.ok(mean<.5,`mean distance ${mean}`);assert.deepEqual(f,simulateImagesFlow({...config,source},data));if(source==='mask'){const masks=f[50].flat().filter(v=>v===-1).length/(64*config.count);near(masks,.5005,.04);}}
});
test('forward corruption reaches independent noise or a completely masked image',()=>{
 for(const source of ['uniform','mask']){const f=simulateImagesDiffusion({...config,source},data,true);assert.ok(f[0].every(x=>nearestDistance(x,data)===0));if(source==='mask')assert.ok(f.at(-1).flat().every(v=>v===-1));else near(f.at(-1).flat().reduce((a,b)=>a+b)/config.count/64,.5,.04);}
});

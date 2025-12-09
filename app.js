// App: dataset generation, plotting and TF.js LSTM training

let dataset = null;
let currentIndex = 0;
let model = null;
let stopRequested = false;

function rng(seed){
  if(seed==null||seed=="") return Math.random;
  return new Math.seedrandom(seed);
}

function generateDataset(){
  const samples = +document.getElementById('samples').value;
  const n = +document.getElementById('seqLen').value;
  const noiseStd = +document.getElementById('noiseStd').value;
  const seed = document.getElementById('seed').value;
  const r = rng(seed);
  const ampMin = +document.getElementById('ampMin').value;
  const ampMax = +document.getElementById('ampMax').value;
  const freqMin = +document.getElementById('freqMin').value;
  const freqMax = +document.getElementById('freqMax').value;

  const out = [];
  const amps = [];
  const freqs = [];
  for(let i=0;i<samples;i++){
    const amp = ampMin + r()*(ampMax-ampMin);
    const freq = freqMin + r()*(freqMax-freqMin);
    const phase = r()*Math.PI*2;
    const seq = [];
    // t from 0..1 with n+1 samples; inputs are first n, target is next
    for(let j=0;j<=n;j++){
      const t = j/(n);
      const clean = amp * Math.sin(2*Math.PI*freq*t + phase);
      const noisy = clean + gaussRandom(r,0,noiseStd);
      seq.push(noisy);
    }
    const x = seq.slice(0,n);
    const y = seq[n];
    out.push({x,y,amp,freq,phase});
    amps.push(amp);
    freqs.push(freq);
  }
  dataset = {meta:{samples,n,noiseStd,seed,ampMin,ampMax,freqMin,freqMax},data:out};
  // update stats
  const stats = document.getElementById('datasetStats');
  const ampMinA = min(amps), ampMaxA = max(amps), ampMeanA = mean(amps);
  const freqMinA = min(freqs), freqMaxA = max(freqs), freqMeanA = mean(freqs);
  stats.innerText = `Samples = ${samples}\nSequence length (n) = ${n}\nNoise std = ${noiseStd}`;
  stats.innerText += `\nAmp range (actual): ${ampMinA.toFixed(3)} - ${ampMaxA.toFixed(3)} (avg ${ampMeanA.toFixed(3)})`;
  stats.innerText += `\nFreq range(actual): ${freqMinA.toFixed(3)} - ${freqMaxA.toFixed(3)} (avg ${freqMeanA.toFixed(3)})`;
  stats.innerText += `\nSeed: ${seed||'none'}`;
  currentIndex = +document.getElementById('startIdx').value || 0;
  renderSamples();
}

function gaussRandom(r,mu=0,sigma=1){
  // Box-Muller
  let u1=0,u2=0; while(u1===0)u1=r(); u2=r();
  const z = Math.sqrt(-2.0*Math.log(u1))*Math.cos(2*Math.PI*u2);
  return z*sigma + mu;
}

function min(arr){return Math.min(...arr)}
function max(arr){return Math.max(...arr)}
function mean(arr){return arr.reduce((a,b)=>a+b,0)/arr.length}

function downloadJSON(){
  if(!dataset) return alert('No dataset');
  const blob = new Blob([JSON.stringify(dataset,null,2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'synthetic_sine_dataset.json';
  a.click();
}

function renderSamples(){
  if(!dataset) return;
  const k = +document.getElementById('showK').value;
  const overlap = document.getElementById('overlap').checked;
  const start = currentIndex;
  const traces = [];
  const n = dataset.meta.n;
  if(overlap){
    for(let i=0;i<k;i++){
      const idx = (start + i) % dataset.data.length;
      const s = dataset.data[idx];
      traces.push({x: Array.from({length:n+1},(_,j)=>j), y: [...s.x, s.y], mode:'lines+markers', name:`#${idx} (amp ${s.amp.toFixed(3)}, f ${s.freq.toFixed(3)})`, hovertemplate: 't=%{x}<br>y=%{y:.4f}<extra></extra>'});
    }
  } else {
    for(let i=0;i<k;i++){
      const idx = (start + i) % dataset.data.length;
      const s = dataset.data[idx];
      traces.push({x: Array.from({length:n},(_,j)=>j), y: s.x, mode:'lines+markers', name:`#${idx} seq`, hovertemplate: 't=%{x}<br>y=%{y:.4f}<extra></extra>'});
      traces.push({x:[n], y:[s.y], mode:'markers', marker:{size:10}, name:`#${idx} target`, hovertemplate: 't=%{x}<br>target=%{y:.4f}<extra></extra>'});
    }
  }
  const layout = {title:'Sample(s)',xaxis:{title:'t'},yaxis:{title:'y'}};
  Plotly.newPlot('samplePlot', traces, layout, {responsive:true});
}

async function startTraining(){
  if(!dataset) return alert('Generate a dataset first');
  // prepare tensors
  const units = +document.getElementById('units').value;
  const batchSize = +document.getElementById('batchSize').value;
  const epochs = +document.getElementById('epochs').value;
  const lr = +document.getElementById('lr').value;
  const valSplit = +document.getElementById('valSplit').value;
  const data = dataset.data;
  const n = dataset.meta.n;
  const xs = tf.tensor3d(data.map(d=>d.x.map(v=>[v]))); // [samples,n,1]
  const ys = tf.tensor2d(data.map(d=>[d.y])); // [samples,1]

  // build model
  if(model) model.dispose();
  model = tf.sequential();
  model.add(tf.layers.lstm({units, inputShape:[n,1]}));
  model.add(tf.layers.dense({units:1}));
  const optimizer = tf.train.adam(lr);
  model.compile({optimizer, loss:'meanSquaredError'});

  stopRequested = false;
  model.stopTraining = false;

  // plot init with loss and val_loss
  const lossTrace = {x:[], y:[], mode:'lines', name:'loss'};
  const valTrace = {x:[], y:[], mode:'lines', name:'val_loss'};
  Plotly.newPlot('trainPlot', [lossTrace, valTrace], {title:'Training Loss', xaxis:{title:'epoch'}, yaxis:{title:'loss'}});

  const valCount = Math.floor(data.length * valSplit);
  const callbacks = {
    onEpochEnd: async (epoch, logs) => {
      // update training plot (loss and val_loss)
      const hasVal = typeof logs.val_loss !== 'undefined';
      await Plotly.extendTraces('trainPlot', {x:[[epoch+1],[epoch+1]], y:[[logs.loss],[hasVal?logs.val_loss:null]]}, [0,1]);
      // update predictions vs targets for validation samples (small set)
      if(valCount>0){
        const valStart = data.length - valCount;
        const vals = data.slice(valStart);
        const valX = vals.map(d=>d.x.map(v=>[v]));
        const valY = vals.map(d=>d.y);
        const valXs = tf.tensor3d(valX);
        const preds = model.predict(valXs);
        const predArr = await preds.data();
        valXs.dispose(); preds.dispose();
        const numPlot = Math.min(50, predArr.length);
        const idxs = Array.from({length:numPlot},(_,i)=>i);
        const tracePred = {x: idxs, y: idxs.map(i=>predArr[i]), mode:'markers', name:'pred', marker:{color:'orange'}};
        const traceTrue = {x: idxs, y: idxs.map(i=>valY[i]), mode:'markers', name:'true', marker:{color:'blue'}};
        Plotly.react('predPlot',[traceTrue, tracePred], {title:'Predictions vs Targets (validation)', xaxis:{title:'sample idx'}, yaxis:{title:'value'}});
      }
      if(stopRequested){ model.stopTraining = true; }
    }
  };

  await model.fit(xs, ys, {batchSize, epochs, validationSplit: valSplit, callbacks});
  xs.dispose(); ys.dispose();
  // display model architecture summary
  const arch = document.getElementById('modelArch');
  arch.textContent = `Input: sequence window of length ${n}, shape [batch, ${n}, 1]\nCore: single LSTM (units=${units}) -> output shape [batch, ${units}]\nHead: Dense(1) -> output shape [batch, 1]\nLoss: meanSquaredError | Optimizer: Adam (lr=${lr})`;
  alert('Training finished');
}

function stopTraining(){ stopRequested = true; }

document.getElementById('genBtn').addEventListener('click', ()=>{ generateDataset(); });
document.getElementById('downloadBtn').addEventListener('click', ()=>downloadJSON());
document.getElementById('renderSample').addEventListener('click', ()=>renderSamples());
document.getElementById('prevSample').addEventListener('click', ()=>{ currentIndex = Math.max(0,currentIndex-1); renderSamples(); });
document.getElementById('nextSample').addEventListener('click', ()=>{ currentIndex = Math.min((dataset?dataset.data.length-1:0), currentIndex+1); renderSamples(); });
document.getElementById('randomSample').addEventListener('click', ()=>{ if(!dataset) return; currentIndex = Math.floor(Math.random()*dataset.data.length); renderSamples(); });
document.getElementById('startTrain').addEventListener('click', ()=>{ startTraining().catch(e=>{console.error(e); alert('Training error: '+e.message)}); });
document.getElementById('stopTrain').addEventListener('click', ()=>{ stopTraining(); });

// initialize empty plots
Plotly.newPlot('samplePlot', [], {title:'Sample(s)'}, {responsive:true});
Plotly.newPlot('trainPlot', [], {title:'Training Loss'});
Plotly.newPlot('predPlot', [], {title:'Predictions vs Targets'});

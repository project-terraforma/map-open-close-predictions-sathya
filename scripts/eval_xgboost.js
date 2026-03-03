const files = {
  SF: '../src/data/test_data.json',
  LA: '../src/data/test_data_la.json',
  Chicago: '../src/data/test_data_chicago.json'
};

let allData = [];
for (const [city, f] of Object.entries(files)) {
  const d = require(f);
  d.forEach(x => { x._city = city; });
  allData.push(...d);
}

let tp=0, tn=0, fp=0, fn=0;
let details = [];
for (const poi of allData) {
  const truth = poi.yelp && poi.yelp.yelp_status;
  const xgb = poi.vision && poi.vision.layers && poi.vision.layers.xgboost && poi.vision.layers.xgboost.score;
  if (!truth || xgb === undefined || xgb === null) continue;
  const trueOpen = truth === 'open';
  const predOpen = xgb > 0.5;
  if (trueOpen && predOpen) tp++;
  else if (!trueOpen && !predOpen) tn++;
  else if (!trueOpen && predOpen) { fp++; details.push({name: poi.name, city: poi._city, truth, score: xgb, type: 'FP'}); }
  else { fn++; details.push({name: poi.name, city: poi._city, truth, score: xgb, type: 'FN'}); }
}

const total = tp+tn+fp+fn;
const acc = ((tp+tn)/total*100).toFixed(1);
const precOpen = tp/(tp+fp);
const recOpen = tp/(tp+fn);
const f1Open = (2*precOpen*recOpen/(precOpen+recOpen)*100).toFixed(1);
const precClosed = tn/(tn+fn);
const recClosed = tn/(tn+fp);
const f1Closed = (2*precClosed*recClosed/(precClosed+recClosed)*100).toFixed(1);

console.log('=== XGBoost Model Performance ===');
console.log('Total:', total);
console.log('Accuracy:', acc+'%', '('+(tp+tn)+'/'+total+')');
console.log('');
console.log('Confusion Matrix:');
console.log('                Pred OPEN   Pred CLOSED');
console.log('  True OPEN:     ', String(tp).padStart(3), '(TP)     ', String(fn).padStart(3), '(FN)');
console.log('  True CLOSED:   ', String(fp).padStart(3), '(FP)     ', String(tn).padStart(3), '(TN)');
console.log('');
console.log('Open class:   Precision='+(precOpen*100).toFixed(1)+'%  Recall='+(recOpen*100).toFixed(1)+'%  F1='+f1Open+'%');
console.log('Closed class: Precision='+(precClosed*100).toFixed(1)+'%  Recall='+(recClosed*100).toFixed(1)+'%  F1='+f1Closed+'%');
console.log('');

// Per-city breakdown
for (const city of ['SF','LA','Chicago']) {
  let ctp=0,ctn=0,cfp=0,cfn=0;
  for (const poi of allData.filter(x=>x._city===city)) {
    const truth = poi.yelp && poi.yelp.yelp_status;
    const xgb = poi.vision && poi.vision.layers && poi.vision.layers.xgboost && poi.vision.layers.xgboost.score;
    if (!truth || xgb === undefined || xgb === null) continue;
    const trueOpen = truth==='open';
    const predOpen = xgb>0.5;
    if(trueOpen&&predOpen)ctp++;
    else if(!trueOpen&&!predOpen)ctn++;
    else if(!trueOpen&&predOpen)cfp++;
    else cfn++;
  }
  const ct=ctp+ctn+cfp+cfn;
  const cacc = ((ctp+ctn)/ct*100).toFixed(0);
  console.log(city.padEnd(8)+': '+cacc+'% accuracy  (TP='+ctp+' TN='+ctn+' FP='+cfp+' FN='+cfn+')');
}

console.log('');
console.log('=== Misclassifications ===');
details.sort((a,b) => {
  if (a.type !== b.type) return a.type.localeCompare(b.type);
  return a.score - b.score;
});
for (const d of details) {
  console.log(d.type+' | '+d.city.padEnd(8)+' | score='+d.score.toFixed(3)+' | '+d.name+' (truly '+d.truth+')');
}

// Score distribution for misclassifications
console.log('');
console.log('=== Score Distribution ===');
const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
for (let i = 0; i < bins.length - 1; i++) {
  const lo = bins[i], hi = bins[i+1];
  const inBin = allData.filter(p => {
    const s = p.vision && p.vision.layers && p.vision.layers.xgboost && p.vision.layers.xgboost.score;
    return s !== undefined && s !== null && s >= lo && s < hi;
  });
  const openInBin = inBin.filter(p => p.yelp && p.yelp.yelp_status === 'open').length;
  const closedInBin = inBin.filter(p => p.yelp && p.yelp.yelp_status === 'closed').length;
  console.log('['+lo.toFixed(1)+'-'+hi.toFixed(1)+') : '+String(inBin.length).padStart(3)+' total  ('+openInBin+' open, '+closedInBin+' closed)');
}

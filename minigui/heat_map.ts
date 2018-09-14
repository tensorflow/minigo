function heatMapN(ns: Float32Array) {
  let result: Float32Array[] = [];
  let nSum = 0;
  ns.forEach((n) => {
    nSum += n;
  });
  nSum = Math.max(nSum, 1);
  ns.forEach((n) => {
    let a = Math.min(Math.sqrt(n / nSum), 0.6);
    if (a > 0) {
      a = 0.1 + 0.9 * a;
    }
    result.push(new Float32Array([0, 0, 0, a]));
  });
  return result;
}

function heatMapDq(qs: Float32Array) {
  let result: Float32Array[] = [];
  qs.forEach((q) => {
    let rgb = q > 0 ? 0 : 255;
    let a = Math.min(Math.abs(q / 100), 0.6);
    result.push(new Float32Array([rgb, rgb, rgb, a]));
  });
  return result;
}

export {
  heatMapN,
  heatMapDq,
}


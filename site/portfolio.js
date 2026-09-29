'use strict';
const comparison = document.getElementById('comparison');
function positionComparison() {
  document.getElementById('overlay').style.clipPath = `inset(0 0 0 ${comparison.value}%)`;
  document.getElementById('divider').style.left = `${comparison.value}%`;
}
comparison.addEventListener('input', positionComparison);
positionComparison();

// One monotonic color scale shared by both measured reconstructions.
const stops = [[0, 4, 7, 20], [.25, 64, 28, 98], [.5, 155, 48, 108], [.75, 227, 104, 88], [1, 251, 247, 188]];
function color(value) {
  const v = Math.min(1, Math.max(0, value));
  let upper = 1;
  while (upper < stops.length - 1 && v > stops[upper][0]) upper++;
  const lo = stops[upper - 1], hi = stops[upper], t = (v - lo[0]) / (hi[0] - lo[0]);
  return lo.slice(1).map((x, i) => Math.round(x + t * (hi[i + 1] - x)));
}
function render(id, matrix, mask) {
  const canvas = document.getElementById(id), ctx = canvas.getContext('2d');
  const n = matrix.length;
  canvas.width = n; canvas.height = n;
  const pixels = ctx.createImageData(n, n);
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    const offset = ((n - 1 - row) * n + col) * 4;
    const rgb = mask[row][col] ? color(matrix[row][col]) : [32, 42, 57];
    pixels.data.set([...rgb, 255], offset);
  }
  ctx.putImageData(pixels, 0, 0);
}
fetch('assets/matrix_preview.json').then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(data => {
  render('base', data.bicubic, data.mask);
  render('prediction', data.cnn, data.mask);
  document.getElementById('matrix-caption').textContent = `${data.structure_id} · rep${data.replicate} · 固定首个 CHID 测试案例 · 25.6 kb 窗口 · 统一 [0,1] 色标。灰色为无效像素。`;
}).catch(() => {
  document.getElementById('matrix-caption').textContent = '请在项目根目录运行 python3 -m http.server 8000，然后访问 http://localhost:8000/site/ 加载矩阵数据。';
});

// Run with: node tests/test_map_coordinates.js
// Exercise the conversion functions embedded in the Gradio map script.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "..", "app", "heye_map.js"), "utf8");
const start = source.indexOf("  function outChina(");
const end = source.indexOf("  function lon2x(", start);
assert.ok(start >= 0 && end > start, "map coordinate functions were not found");
const { wgsToGcj, gcjToWgs } = vm.runInNewContext(
  source.slice(start, end) + "\n({ wgsToGcj, gcjToWgs })"
);

// Clicked locations are converted from Amap's GCJ-02 back to WGS84 before
// satellite search. A one-step inverse leaves a visible position error.
for (const [lat, lon] of [
  [39.9042, 116.4074], // Beijing
  [29.55, 116.92],     // Poyang Lake
  [23.1291, 113.2644], // Guangzhou
  [30.5728, 104.0668], // Chengdu
  [45.8038, 126.5349], // Harbin
]) {
  const gcj = wgsToGcj(lat, lon);
  const restored = gcjToWgs(gcj[0], gcj[1]);
  assert.ok(Math.abs(restored[0] - lat) < 1e-7, `latitude round trip failed at ${lat}, ${lon}`);
  assert.ok(Math.abs(restored[1] - lon) < 1e-7, `longitude round trip failed at ${lat}, ${lon}`);
}

// Outside the GCJ-02 coverage area, both directions must remain unchanged.
for (const [lat, lon] of [[-27.47, 153.03], [40.71, -74.01]]) {
  assert.deepEqual(Array.from(wgsToGcj(lat, lon)), [lat, lon]);
  assert.deepEqual(Array.from(gcjToWgs(lat, lon)), [lat, lon]);
}

console.log("Map coordinate round trips passed");

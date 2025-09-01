// eval/compare.mjs
import fs from "node:fs";
import path from "node:path";

function listRuns(dir = "eval/runs") {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => f.endsWith(".json"))
    .map(f => {
      const p = path.join(dir, f);
      const st = fs.statSync(p);
      return { path: p, name: f, mtime: st.mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime); // newest first
}

function loadRun(p) {
  const j = JSON.parse(fs.readFileSync(p, "utf8"));
  const map = new Map(j.items.map(i => [i.id, i]));
  return { summary: j.summary, items: map };
}

function loadGolden(p) {
  if (!p) return null;
  const arr = JSON.parse(fs.readFileSync(p, "utf8"));
  const q = new Map(arr.map(r => [r.id, { q: r.question, ref: r.reference }]));
  return q;
}

function fmtPct(x) { return (x*100).toFixed(1) + "%"; }

const args = process.argv.slice(2);
let aPath, bPath, gPath, prev = 1;

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (!a.startsWith("-") && !aPath) { aPath = a; continue; }
  if (!a.startsWith("-") && !bPath) { bPath = a; continue; }
  if (a === "--golden") { gPath = args[++i]; continue; }
  if (a === "--prev")   { prev = Math.max(1, Number(args[++i])); continue; }
}

// Auto-pick latest two if none provided
if (!aPath && !bPath) {
  const runs = listRuns();
  if (runs.length < 2) {
    console.error("Need at least two run files in eval/runs to auto-compare.");
    process.exit(1);
  }
  bPath = runs[0].path;                     // newest
  aPath = runs[prev] ? runs[prev].path : runs[1].path; // Nth previous, fallback to previous
}

// Default golden path if present
if (!gPath && fs.existsSync("eval/tests/golden.json")) gPath = "eval/tests/golden.json";

const A = loadRun(aPath);
const B = loadRun(bPath);
const G = loadGolden(gPath);

const ids = new Set([...A.items.keys(), ...B.items.keys()]);
const regressions = [];
const improvements = [];
const errorsFixed = [];
const errorsNew = [];

for (const id of ids) {
  const a = A.items.get(id);
  const b = B.items.get(id);
  if (!a || !b) continue; // ignore ids not in both
  const aPass = !!a.pass && !a.error;
  const bPass = !!b.pass && !b.error;

  if (a.error && !b.error && bPass) errorsFixed.push(id);
  if (!a.error && b.error) errorsNew.push(id);

  if (aPass && !bPass) regressions.push(id);
  if (!aPass && bPass) improvements.push(id);
}

const accA = A.summary.accuracy ?? (A.summary.passed / A.summary.total);
const accB = B.summary.accuracy ?? (B.summary.passed / B.summary.total);
const dAcc = accB - accA;

console.log(`\nCompare`);
console.log(` old: ${path.basename(aPath)}  model=${A.summary.model}  acc=${fmtPct(accA)}  hash(golden)=${A.summary.golden_hash}`);
console.log(` new: ${path.basename(bPath)}  model=${B.summary.model}  acc=${fmtPct(accB)}  hash(golden)=${B.summary.golden_hash}`);
console.log(` Δacc: ${fmtPct(dAcc)}  Δpassed: ${B.summary.passed - A.summary.passed}  Δavg_latency: ${B.summary.avg_latency_ms - A.summary.avg_latency_ms}ms`);

function show(list, title) {
  if (!list.length) return;
  console.log(`\n${title} (${list.length})`);
  for (const id of list) {
    const info = G?.get(id);
    console.log(`  #${id}${info ? ` — ${info.q}` : ""}`);
  }
}
show(regressions, "Regressions (was pass, now fail/error)");
show(improvements, "Improvements (was fail/error, now pass)");
show(errorsNew, "New errors");
show(errorsFixed, "Fixed errors");
console.log("");

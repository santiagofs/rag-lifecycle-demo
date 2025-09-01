// eval/run.mjs
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b";
const goldenPath = process.argv[2] || "eval/tests/golden.json";
const resultsPath = process.argv[3] || "eval/results.json";

// --- I/O helpers ---
function readGoldenJSON(p) {
  const raw = fs.readFileSync(p, "utf8");
  const data = JSON.parse(raw);
  if (!Array.isArray(data)) throw new Error("golden file must be a JSON array");
  // Minimal validation; fail fast
  for (const [i, r] of data.entries()) {
    for (const key of ["id", "question", "context", "reference", "eval_type"]) {
      if (!(key in r)) throw new Error(`row ${i} missing '${key}'`);
    }
  }
  return data;
}

// --- Model call ---
async function generateOllama({ question, context }) {
  const prompt = `Use ONLY the <context> to answer.
If the answer is not present in context, reply exactly: "I don't know".
<context>
${context}
</context>
Question: ${question}
Answer:`;
  const res = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, prompt, stream: false, options: { temperature: 0, top_p: 1 } }),
  });
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
  const data = await res.json();
  return (data.response || "").trim();
}

// --- Scoring ---
function normalizeAggressive(s) {
  return (s || "")
    .normalize("NFKC")
    .replace(/[“”„‟]/g, '"')
    .replace(/[‘’‚‛]/g, "'")
    .replace(/[–—]/g, "-")
    .replace(/[^\S\r\n]+/g, " ")
    .replace(/[.,;:!?]+$/g, "")
    .toLowerCase()
    .trim();
}

function checkContains(out, reference) {
  return normalizeAggressive(out).includes(normalizeAggressive(reference));
}

function checkDontKnow(out) {
  return normalizeAggressive(out) === normalizeAggressive("I don't know");
}

function checkRegex(out, pattern) {
  const outTrim = (out || "").trim();

  // Strip anchors from the user's pattern, then try exact-ish first
  const core = String(pattern).replace(/^\^/, "").replace(/\$$/, "");
  const exactish = `^(?:${core})\\s*[.,;:!?]?$`;

  try {
    // 1) strict: must be exactly the phrase (+ optional trailing punctuation)
    if (new RegExp(exactish, "iu").test(outTrim)) {
      return { pass: true, error: null };
    }
    // 2) fallback: allow preceding text, but require the phrase as a word-bounded
    // tail (+ optional trailing punctuation). This fixes #3 ("... 45-degree.")
    const tailBounded = `\\b(?:${core})\\b\\s*[.,;:!?]?$`;
    if (new RegExp(tailBounded, "iu").test(outTrim)) {
      return { pass: true, error: null };
    }
    return { pass: false, error: null };
  } catch {
    return { pass: false, error: "BAD_REGEX" };
  }
}


// --- Main ---
async function main() {
  const started_at = new Date().toISOString();
  const testsRaw = fs.readFileSync(goldenPath, "utf8");
  const tests = JSON.parse(testsRaw);
  const golden_hash = crypto.createHash("sha256").update(testsRaw).digest("hex").slice(0, 8);

  console.log(`\nRunning ${tests.length} tests → ${MODEL} @ ${OLLAMA_URL}\n`);

  const items = [];
  let passed = 0;

  // Optional warmup
  try { await generateOllama({ question: "ping", context: "say pong" }); } catch {}

  for (const t of tests) {
    const id = t.id;
    const eval_type = t.eval_type || "contains";
    let actual = "";
    let error = null;
    let pass = false;

    const start = Date.now();
    try {
      actual = await generateOllama({ question: t.question, context: t.context });
      if (eval_type === "dontknow") {
        pass = checkDontKnow(actual);
      } else if (eval_type === "regex") {
        const res = checkRegex(actual, t.reference);
        pass = res.pass;
        error = res.error;
      } else {
        pass = checkContains(actual, t.reference);
      }
    } catch (e) {
      error = "PIPELINE_ERROR";
    }
    const latency_ms = Date.now() - start;

    if (error) pass = false;

    // Truncate pathological outputs
    if (actual && actual.length > 8192) actual = actual.slice(0, 8192);

    items.push({
      id,
      pass,
      latency_ms,
      error,              // null or MACHINE_CODE (e.g., BAD_REGEX, PIPELINE_ERROR)
      actual
    });

    if (pass) passed++;
    const status = pass ? "✅" : error ? "💥" : "❌";
    console.log(`${status} #${id} ${latency_ms}ms  ref="${t.reference}"  out="${(actual || "").slice(0,80)}"${error ? `  error=${error}` : ""}`);
  }

  const finished_at = new Date().toISOString();
  const total = tests.length;
  const accuracy = total ? Number((passed / total).toFixed(2)) : 0;
  const avg_latency_ms = items.length
    ? Math.round(items.reduce((a, x) => a + x.latency_ms, 0) / items.length)
    : 0;

  const summary = {
    run_id: `${started_at.slice(0,10)}--${Math.floor(Math.random()*1e6).toString().padStart(6,'0')}`,
    model: MODEL,
    pipeline_version: process.env.PIPELINE_VERSION || "dev",
    harness_version: process.env.HARNESS_VERSION || "dev",
    golden_hash,
      config_hash: crypto.createHash("sha256").update(JSON.stringify({
      MODEL, OLLAMA_URL,
      regex_mode: "exactish_tail",
      normalization: "NFKC_lower_trim_punctdash",
      dontknow_phrase: "i don't know",
      gen: { temperature: 0, top_p: 1 }    
    })).digest("hex").slice(0,8),
    started_at,
    finished_at,
    total,
    passed,
    accuracy,
    avg_latency_ms
  };

  fs.mkdirSync(path.dirname(resultsPath), { recursive: true });
  fs.writeFileSync(resultsPath, JSON.stringify({ summary, items }, null, 2));
  fs.mkdirSync("eval/runs", { recursive: true });
  fs.writeFileSync(`eval/runs/${summary.run_id}.json`, JSON.stringify({ summary, items }, null, 2));

  const failedIds = items.filter(r => !r.pass && !r.error).map(r => r.id);
  console.log(`\nSummary: ${passed}/${total} passed (${(accuracy*100).toFixed(1)}%). Report: ${resultsPath}`);
  console.log(`Failures: ${failedIds.length ? failedIds.join(", ") : "none"}\n`);
}

await main();

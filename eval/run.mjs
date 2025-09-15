// eval/run.mjs
import fs from "node:fs"
import path from "node:path"
import crypto from "node:crypto"
import { callRagAPI, healthCheck } from "./adapter.mjs"

// Prefer unified OLLAMA_BASE_URL, fall back to legacy OLLAMA_URL
const OLLAMA_BASE_URL =
  process.env.OLLAMA_BASE_URL ||
  process.env.OLLAMA_URL ||
  "http://localhost:11434"
const MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b"
const goldenPath = process.argv[2] || "eval/tests/golden.json"
const resultsPath = process.argv[3] || "eval/results.json"

// Check if API mode is enabled
const useAPI = process.argv.includes("--api")
const API_BASE_URL = process.env.RAG_API_URL || "http://localhost:8000"

// --- I/O helpers ---
function readGoldenJSON(p) {
  const raw = fs.readFileSync(p, "utf8")
  const data = JSON.parse(raw)
  if (!Array.isArray(data)) throw new Error("golden file must be a JSON array")
  // Minimal validation; fail fast
  for (const [i, r] of data.entries()) {
    for (const key of ["id", "question", "context", "reference", "eval_type"]) {
      if (!(key in r)) throw new Error(`row ${i} missing '${key}'`)
    }
  }
  return data
}

// --- Model call ---
async function generateOllama({ question, context }) {
  const prompt = `Use ONLY the <context> to answer.
If the answer is not present in context, reply exactly: "I don't know".
Respond with only the answer phrase, no extra words.
Copy the wording verbatim from the context when possible, including units and qualifiers exactly as written.
Never infer, assume, or list items not present in the context.
If simple math or unit conversion is needed, compute it explicitly using the context.
<context>
${context}
</context>
Question: ${question}
Answer:`
  const res = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      prompt,
      stream: false,
      options: { temperature: 0, top_p: 1 },
    }),
  })
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`)
  const data = await res.json()
  return (data.response || "").trim()
}

async function generateWithAPI(question) {
  return await callRagAPI(question)
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
    .trim()
}

function checkContains(out, reference) {
  return normalizeAggressive(out).includes(normalizeAggressive(reference))
}

function checkDontKnow(out) {
  return normalizeAggressive(out) === normalizeAggressive("I don't know")
}

function checkRegex(out, pattern) {
  const outTrim = (out || "").trim()

  // Normalize output similarly to contains() to avoid dash/quote issues
  function normalizeForRegex(s) {
    return (s || "")
      .normalize("NFKC")
      .replace(/[“”„‟]/g, '"')
      .replace(/[‘’‚‛]/g, "'")
      .replace(/[–—]/g, "-")
      .replace(/[^\S\r\n]+/g, " ")
      .trim()
  }

  const normalizedOut = normalizeForRegex(outTrim)
  const raw = String(pattern)

  // If the pattern explicitly uses leading and trailing wildcards or anchors,
  // try it as-is first against the normalized output.
  const hasWildcardEdges = /^\s*\.\*/.test(raw) && /\.\*\s*$/.test(raw)

  try {
    if (hasWildcardEdges || /\^|\$/.test(raw)) {
      if (new RegExp(raw, "iu").test(normalizedOut)) {
        return { pass: true, error: null }
      }
    }

    // Otherwise, strip anchors and apply exact-ish and tail-bounded checks
    const core = raw.replace(/^\^/, "").replace(/\$$/, "")
    const exactish = `^(?:${core})\\s*[.,;:!?]?$`
    if (new RegExp(exactish, "iu").test(normalizedOut)) {
      return { pass: true, error: null }
    }
    const tailBounded = `\\b(?:${core})\\b\\s*[.,;:!?]?$`
    if (new RegExp(tailBounded, "iu").test(normalizedOut)) {
      return { pass: true, error: null }
    }
    return { pass: false, error: null }
  } catch {
    return { pass: false, error: "BAD_REGEX" }
  }
}

// --- Main ---
async function main() {
  const started_at = new Date().toISOString()
  const testsRaw = fs.readFileSync(goldenPath, "utf8")
  const tests = JSON.parse(testsRaw)
  const golden_hash = crypto
    .createHash("sha256")
    .update(testsRaw)
    .digest("hex")
    .slice(0, 8)

  console.log(
    `\nRunning ${tests.length} tests → ${MODEL} @ ${OLLAMA_BASE_URL}${
      useAPI ? ` (API mode @ ${API_BASE_URL})` : ""
    }\n`
  )

  // Check API health if in API mode
  if (useAPI) {
    const apiHealthy = await healthCheck()
    if (!apiHealthy) {
      console.error(
        `❌ API health check failed. Make sure the Python API is running at ${API_BASE_URL}`
      )
      process.exit(1)
    }
    console.log("✅ API health check passed")
  }

  const items = []
  let passed = 0

  // Optional warmup
  try {
    if (useAPI) {
      await generateWithAPI("ping")
    } else {
      await generateOllama({ question: "ping", context: "say pong" })
    }
  } catch {}

  for (const t of tests) {
    const id = t.id
    const eval_type = t.eval_type || "contains"
    let actual = ""
    let error = null
    let pass = false

    const start = Date.now()
    try {
      if (useAPI) {
        actual = await generateWithAPI(t.question)
      } else {
        actual = await generateOllama({
          question: t.question,
          context: t.context,
        })
      }
      if (eval_type === "dontknow") {
        pass = checkDontKnow(actual)
      } else if (eval_type === "regex") {
        const res = checkRegex(actual, t.reference)
        pass = res.pass
        error = res.error
      } else {
        pass = checkContains(actual, t.reference)
      }
    } catch (e) {
      error = "PIPELINE_ERROR"
    }
    const latency_ms = Date.now() - start

    if (error) pass = false

    // Truncate pathological outputs
    if (actual && actual.length > 8192) actual = actual.slice(0, 8192)

    items.push({
      id,
      pass,
      latency_ms,
      error, // null or MACHINE_CODE (e.g., BAD_REGEX, PIPELINE_ERROR)
      actual,
    })

    if (pass) passed++
    const status = pass ? "✅" : error ? "💥" : "❌"
    console.log(
      `${status} #${id} ${latency_ms}ms  ref="${t.reference}"  out="${(
        actual || ""
      ).slice(0, 80)}"${error ? `  error=${error}` : ""}`
    )
  }

  const finished_at = new Date().toISOString()
  const total = tests.length
  const accuracy = total ? Number((passed / total).toFixed(2)) : 0
  const avg_latency_ms = items.length
    ? Math.round(items.reduce((a, x) => a + x.latency_ms, 0) / items.length)
    : 0

  const summary = {
    run_id: `${started_at.slice(0, 10)}--${Math.floor(Math.random() * 1e6)
      .toString()
      .padStart(6, "0")}`,
    model: MODEL,
    pipeline_version: process.env.PIPELINE_VERSION || "dev",
    harness_version: process.env.HARNESS_VERSION || "dev",
    golden_hash,
    config_hash: crypto
      .createHash("sha256")
      .update(
        JSON.stringify({
          MODEL,
          OLLAMA_BASE_URL,
          useAPI,
          API_BASE_URL,
          regex_mode: "exactish_tail",
          normalization: "NFKC_lower_trim_punctdash",
          dontknow_phrase: "i don't know",
          gen: { temperature: 0, top_p: 1 },
        })
      )
      .digest("hex")
      .slice(0, 8),
    started_at,
    finished_at,
    total,
    passed,
    accuracy,
    avg_latency_ms,
  }

  fs.mkdirSync(path.dirname(resultsPath), { recursive: true })
  fs.writeFileSync(resultsPath, JSON.stringify({ summary, items }, null, 2))
  fs.mkdirSync("eval/runs", { recursive: true })
  fs.writeFileSync(
    `eval/runs/${summary.run_id}.json`,
    JSON.stringify({ summary, items }, null, 2)
  )

  const failedIds = items.filter((r) => !r.pass && !r.error).map((r) => r.id)
  console.log(
    `\nSummary: ${passed}/${total} passed (${(accuracy * 100).toFixed(
      1
    )}%). Report: ${resultsPath}`
  )
  console.log(`Failures: ${failedIds.length ? failedIds.join(", ") : "none"}\n`)
}

await main()

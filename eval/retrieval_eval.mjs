// eval/retrieval_eval.mjs
// Retrieval-only evaluation for Hit@k metrics

import fs from "node:fs"
import path from "node:path"
import crypto from "node:crypto"
import {
  callRetrieveAPI,
  healthCheck,
  checkRetrievalHit,
} from "./retrieval_adapter.mjs"

const API_BASE_URL = process.env.RAG_API_URL || "http://localhost:8000"
const TOP_K = parseInt(process.env.TOP_K) || 5
const goldenPath = process.argv[2] || "eval/tests/golden.json"
const resultsPath = process.argv[3] || "eval/retrieval_results.json"

// --- I/O helpers ---
function readGoldenJSON(p) {
  const raw = fs.readFileSync(p, "utf8")
  const data = JSON.parse(raw)
  if (!Array.isArray(data)) throw new Error("golden file must be a JSON array")
  // Minimal validation; fail fast
  for (const [i, r] of data.entries()) {
    for (const key of ["id", "question", "reference"]) {
      if (!(key in r)) throw new Error(`row ${i} missing '${key}'`)
    }
  }
  return data
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
    `\nRunning retrieval evaluation on ${tests.length} tests → Hit@${TOP_K} @ ${API_BASE_URL}\n`
  )

  // Check API health
  const apiHealthy = await healthCheck()
  if (!apiHealthy) {
    console.error(
      `❌ API health check failed. Make sure the Python API is running at ${API_BASE_URL}`
    )
    process.exit(1)
  }
  console.log("✅ API health check passed")

  const items = []
  let hits = 0

  // Optional warmup
  try {
    await callRetrieveAPI("ping")
  } catch {}

  for (const t of tests) {
    const id = t.id
    const question = t.question
    const expectedSubstring = t.reference
    let results = []
    let error = null
    let hit = false

    const start = Date.now()
    try {
      results = await callRetrieveAPI(question)
      hit = checkRetrievalHit(results, expectedSubstring)
    } catch (e) {
      error = "PIPELINE_ERROR"
    }
    const latency_ms = Date.now() - start

    if (error) hit = false

    items.push({
      id,
      hit,
      latency_ms,
      error,
      question,
      expected_substring: expectedSubstring,
      retrieved_count: results.length,
      retrieved_snippets: results.map((r) => r.text).slice(0, 3), // Store first 3 for debugging
    })

    if (hit) hits++
    const status = hit ? "✅" : error ? "💥" : "❌"
    console.log(
      `${status} #${id} ${latency_ms}ms  expected="${expectedSubstring}"  retrieved=${
        results.length
      } snippets${error ? `  error=${error}` : ""}`
    )
  }

  const finished_at = new Date().toISOString()
  const total = tests.length
  const hitRate = total ? Number((hits / total).toFixed(2)) : 0
  const avg_latency_ms = items.length
    ? Math.round(items.reduce((a, x) => a + x.latency_ms, 0) / items.length)
    : 0

  const summary = {
    run_id: `${started_at.slice(0, 10)}--${Math.floor(Math.random() * 1e6)
      .toString()
      .padStart(6, "0")}`,
    evaluation_type: "retrieval_hit_at_k",
    top_k: TOP_K,
    api_url: API_BASE_URL,
    pipeline_version: process.env.PIPELINE_VERSION || "dev",
    harness_version: process.env.HARNESS_VERSION || "dev",
    golden_hash,
    config_hash: crypto
      .createHash("sha256")
      .update(
        JSON.stringify({
          TOP_K,
          API_BASE_URL,
          normalization: "NFKC_lower_trim",
        })
      )
      .digest("hex")
      .slice(0, 8),
    started_at,
    finished_at,
    total,
    hits,
    hit_rate: hitRate,
    avg_latency_ms,
  }

  fs.mkdirSync(path.dirname(resultsPath), { recursive: true })
  fs.writeFileSync(resultsPath, JSON.stringify({ summary, items }, null, 2))
  fs.mkdirSync("eval/runs", { recursive: true })
  fs.writeFileSync(
    `eval/runs/retrieval_${summary.run_id}.json`,
    JSON.stringify({ summary, items }, null, 2)
  )

  const missedIds = items.filter((r) => !r.hit && !r.error).map((r) => r.id)
  console.log(
    `\nRetrieval Summary: ${hits}/${total} hits (Hit@${TOP_K} = ${(
      hitRate * 100
    ).toFixed(1)}%). Report: ${resultsPath}`
  )
  console.log(
    `Missed retrievals: ${missedIds.length ? missedIds.join(", ") : "none"}\n`
  )
}

await main()

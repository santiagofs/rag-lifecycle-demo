import fs from "node:fs"
import path from "node:path"
import readline from "node:readline"

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434"
const MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b"
const csvPath = process.argv[2] || "eval/tests/golden.csv"
const resultsPath = "eval/results.json"

function parseCSV(text) {
  const [header, ...rows] = text.trim().split(/\r?\n/)
  const cols = header.split(",")
  return rows.map((r) => {
    // naive CSV split (fine for our simple rows)
    const parts = r.split(",")
    const obj = {}
    cols.forEach((c, i) => (obj[c.trim()] = (parts[i] ?? "").trim()))
    return obj
  })
}

async function generateOllama({ question, context }) {
  const prompt = `Use ONLY the <context> to answer.
If the answer is not present in context, reply exactly: "I don't know".
<context>
${context}
</context>
Question: ${question}
Answer:`
  const res = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, prompt, stream: false }),
  })
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`)
  const data = await res.json()
  return (data.response || "").trim()
}

function contains(haystack, needle) {
  return haystack.toLowerCase().includes(needle.toLowerCase())
}

async function main() {
  // warm model (optional, fast no-op)
  try {
    await generateOllama({ question: "ping", context: "say pong" })
  } catch {}

  const csv = fs.readFileSync(csvPath, "utf8")
  const tests = parseCSV(csv)
  const results = []
  let pass = 0

  console.log(
    "\nRunning",
    tests.length,
    "tests against",
    MODEL,
    "via",
    OLLAMA_URL,
    "\n"
  )

  for (const t of tests) {
    const { id, question, context, reference } = t
    let out = "",
      error = null,
      ok = false
    const start = Date.now()
    try {
      out = await generateOllama({ question, context })
      ok = contains(out, reference)
    } catch (e) {
      error = String(e)
    }
    const ms = Date.now() - start
    results.push({ id, question, reference, out, ok, ms, error })
    if (ok) pass++
    const status = ok ? "✅" : error ? "💥" : "❌"
    console.log(
      `${status} #${id} ${ms}ms  ref="${reference}"  out="${(out || "").slice(
        0,
        80
      )}"${error ? `  error=${error}` : ""}`
    )
  }

  fs.mkdirSync(path.dirname(resultsPath), { recursive: true })
  fs.writeFileSync(
    resultsPath,
    JSON.stringify(
      { model: MODEL, url: OLLAMA_URL, pass, total: tests.length, results },
      null,
      2
    )
  )
  console.log(
    `\nSummary: ${pass}/${tests.length} passed. Report: ${resultsPath}\n`
  )
}

await main()

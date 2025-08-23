import fs from "node:fs"
import path from "node:path"
import readline from "node:readline"

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434"
const MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b"
const csvPath = process.argv[2] || "eval/tests/golden.csv"
const resultsPath = "eval/results.json"

function parseCSV(text) {
  // Normalize CRLF, drop empty lines
  const lines = text.replace(/\r/g, "").split("\n").filter(l => l.trim().length)
  if (lines.length === 0) return []
  const header = splitCSVLine(lines[0])
  return lines.slice(1).map(line => {
    const cells = splitCSVLine(line)
    const row = {}
    header.forEach((h, i) => (row[h.trim()] = (cells[i] ?? "").trim()))
    return row
  })
}

function splitCSVLine(line) {
  return line
    // split on commas that are NOT inside quotes
    .split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/)
    .map(cell => {
      let s = cell.trim();
      // if quoted, strip outer quotes and unescape ""
      if (s.startsWith('"') && s.endsWith('"')) {
        s = s.slice(1, -1).replace(/""/g, '"');
      }
      return s;
    });
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

// ---- Normalization & checks ----
function normalizeAggressive(s) {
  return (s || "")
    .normalize("NFKC")           // Unicode compatibility fold
    .replace(/[“”„‟]/g, '"')     // smart double quotes -> "
    .replace(/[‘’‚‛]/g, "'")     // smart single quotes -> '
    .replace(/[–—]/g, "-")       // en/em dashes -> hyphen
    .replace(/[^\S\r\n]+/g, " ") // collapse all spaces
    .replace(/[.,;:!?]+$/g, "")  // strip trailing punctuation
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
  // Strip user anchors if present, then anchor ourselves and allow one trailing punctuation.
  const core = pattern.replace(/^\^/, "").replace(/\$$/, "");
  const wrapped = `^(?:${core})\\s*[.,;:!?]?$`;
  const re = new RegExp(wrapped, "i");
  return re.test((out || "").trim());
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
    const { id, question, context, reference, eval_type = "contains" } = t
    let out = "",
      error = null,
      ok = false
    const start = Date.now()
    try {
      out = await generateOllama({ question, context })
      switch (eval_type) {
        case "dontknow": ok = checkDontKnow(out); break
        case "regex":    ok = checkRegex(out, reference); break
        case "contains":
        default:         ok = checkContains(out, reference)
      }
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

  const total = tests.length
  const accuracy = total ? (pass / total) * 100 : 0
  const failedIds = results.filter(r => !r.ok && !r.error).map(r => r.id)

  console.log(`\nSummary: ${pass}/${total} passed (${accuracy.toFixed(1)}%). Report: ${resultsPath}`)
  console.log(`Failures: ${failedIds.length ? failedIds.join(", ") : "none"}\n`)
}

await main()

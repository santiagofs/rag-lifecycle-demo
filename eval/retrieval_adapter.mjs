// eval/retrieval_adapter.mjs
// Retrieval-only evaluation adapter for Hit@k metrics

const API_BASE_URL = process.env.RAG_API_URL || "http://localhost:8000"
const TOP_K = parseInt(process.env.TOP_K) || 5

export async function callRetrieveAPI(question) {
  try {
    const response = await fetch(`${API_BASE_URL}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        k: TOP_K,
        method: "cosine", // Can be "cosine", "hybrid", or "fts"
      }),
    })

    if (!response.ok) {
      throw new Error(`API HTTP ${response.status}: ${await response.text()}`)
    }

    const data = await response.json()
    return data.results || []
  } catch (error) {
    console.error(`Retrieve API call failed: ${error.message}`)
    throw error
  }
}

export async function healthCheck() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}

// Normalize text for case-insensitive comparison
function normalizeText(s) {
  return (s || "")
    .normalize("NFKC")
    .replace(/[^\S\r\n]+/g, " ")
    .toLowerCase()
    .trim()
}

// Check if any retrieved snippet contains the expected substring
export function checkRetrievalHit(results, expectedSubstring) {
  const normalizedExpected = normalizeText(expectedSubstring)

  for (const result of results) {
    const snippetText = result.text || ""
    const normalizedSnippet = normalizeText(snippetText)

    if (normalizedSnippet.includes(normalizedExpected)) {
      return true
    }
  }

  return false
}

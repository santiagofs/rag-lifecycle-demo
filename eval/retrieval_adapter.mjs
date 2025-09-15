// eval/retrieval_adapter.mjs
// Retrieval-only evaluation adapter for Hit@k metrics

const API_BASE_URL = process.env.RAG_API_URL || "http://localhost:8000"
const TOP_K = parseInt(process.env.TOP_K) || 5
const RETRIEVAL_METHOD = (
  process.env.RAG_RETRIEVAL_METHOD || "cosine"
).toLowerCase()

export async function callRetrieveAPI(question) {
  try {
    const response = await fetch(`${API_BASE_URL}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        k: TOP_K,
        method: RETRIEVAL_METHOD, // "cosine", "hybrid", "fts", or "faiss"
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

// Check if any retrieved snippet contains the expected substring or matches regex
export function checkRetrievalHit(results, expectedSubstring) {
  for (const result of results) {
    const snippetText = result.text || ""
    const normalizedSnippet = normalizeText(snippetText)

    // Check if expectedSubstring looks like a regex pattern
    if (
      expectedSubstring.startsWith("(?=") ||
      expectedSubstring.startsWith(".*") ||
      expectedSubstring.startsWith("^")
    ) {
      try {
        // For strict regex patterns like ^LLAMA-3\.1:8B$, we need to extract the core content
        let regexPattern = expectedSubstring
        if (
          expectedSubstring.startsWith("^") &&
          expectedSubstring.endsWith("$")
        ) {
          // Extract the core pattern without anchors for more flexible matching
          const corePattern = expectedSubstring.slice(1, -1)
          // For anchored patterns, just search for the core content with word boundaries
          regexPattern = `\\b${corePattern}\\b`
        }

        const regex = new RegExp(regexPattern, "i")
        if (regex.test(normalizedSnippet)) {
          return true
        }
      } catch (e) {
        // If regex is invalid, fall back to substring match
        const normalizedExpected = normalizeText(expectedSubstring)
        if (normalizedSnippet.includes(normalizedExpected)) {
          return true
        }
      }
    } else {
      // Simple substring match
      const normalizedExpected = normalizeText(expectedSubstring)
      if (normalizedSnippet.includes(normalizedExpected)) {
        return true
      }
    }
  }

  return false
}

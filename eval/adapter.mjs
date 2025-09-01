// eval/adapter.mjs
// Use global fetch available in modern Node; avoid non-portable 'node:fetch' import

const API_BASE_URL = process.env.RAG_API_URL || "http://localhost:8000"

export async function callRagAPI(question, context) {
  try {
    const response = await fetch(`${API_BASE_URL}/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        k: 3,
        method: "cosine", // Can be "cosine", "hybrid", or "fts"
      }),
    })

    if (!response.ok) {
      throw new Error(`API HTTP ${response.status}: ${await response.text()}`)
    }

    const data = await response.json()
    return data.answer || ""
  } catch (error) {
    console.error(`API call failed: ${error.message}`)
    throw error
  }
}

export async function callRetrieveAPI(question) {
  try {
    const response = await fetch(`${API_BASE_URL}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        k: 3,
        method: "cosine",
      }),
    })

    if (!response.ok) {
      throw new Error(`API HTTP ${response.status}: ${await response.text()}`)
    }

    const data = await response.json()
    return data.context || ""
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

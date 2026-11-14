
const OLLAMA_URL = "http://localhost:11434";

const MODEL = "nomic-embed-text:latest";

// Generate an embedding from text
async function generateEmbedding(text, task) {

    if (typeof text !== "string" || !text.trim()) {
        throw new Error("Text cannot be empty.");
    }

    const response = await fetch(
        `${OLLAMA_URL}/api/embeddings`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                model: MODEL,
                prompt: `${task}: ${text.trim()}`
            })
        }
    );

    if (!response.ok) {
        throw new Error(
            `Ollama embedding failed: ${response.status} ${response.statusText}`
        );
    }

    const data = await response.json();

    if (
        !Array.isArray(data.embedding) ||
        data.embedding.length === 0 ||
        !data.embedding.every(Number.isFinite)
    ) {
        throw new Error("Invalid embedding received from Ollama.");
    }

    return data.embedding;
}

// Generate embedding for stored documents
export async function embedDocument(text) {

    return generateEmbedding(
        text,
        "search_document"
    );
}

// Generate embedding for user questions
export async function embedQuery(text) {

    return generateEmbedding(
        text,
        "search_query"
    );
}


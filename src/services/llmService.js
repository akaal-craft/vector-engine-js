
const OLLAMA_URL = "http://localhost:11434";

const MODEL = "qwen3.5:2b-q4_K_M";

// Generate an answer using retrieved documents
export async function generateAnswer(
    question,
    retrievedDocuments
) {

    if (!question || typeof question !== "string") {
        throw new Error("Question cannot be empty.");
    }

    if (!Array.isArray(retrievedDocuments) ||
        retrievedDocuments.length === 0) {
        return "I could not find relevant information to answer your question.";
    }

    // Extract the original text from retrieved documents.
    const context = retrievedDocuments
        .map((document, index) => {
            return `Document ${index + 1}:\n${document.metadata?.text ?? ""}`;
        })
        .join("\n\n");

    const prompt = `
You are a helpful question-answering assistant.

Answer the user's question using only the information
provided in the documents below.

If the documents do not contain enough information,
clearly say that the information is insufficient.

DOCUMENTS:
${context}

USER QUESTION:
${question}

ANSWER:
`;

    const response = await fetch(
        `${OLLAMA_URL}/api/generate`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                model: MODEL,
                prompt,
                stream: false,
                options: {
                    temperature: 0.2
                }
            })
        }
    );

    if (!response.ok) {
        throw new Error(
            `Ollama answer generation failed: ${response.status}`
        );
    }

    const data = await response.json();

    if (typeof data.response !== "string") {
        throw new Error("Invalid response from Ollama.");
    }

    return data.response.trim();
}

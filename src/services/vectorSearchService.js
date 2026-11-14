
import {
    embedDocument,
    embedQuery
} from "./embeddingService.js";

// ==========================================
// STORE TEXT
// ==========================================

export async function insertText(
    index,
    id,
    text,
    metadata = {}
) {

    if (!text || typeof text !== "string" || !text.trim()) {
        throw new Error("Text cannot be empty.");
    }

    // Convert document text into a vector.
    const vector = await embedDocument(text);

    // Store vector along with original text.
    return index.insert(id, vector, {
        ...metadata,
        text: text.trim()
    });
}

// ==========================================
// SEARCH USING TEXT
// ==========================================

export async function searchText(
    index,
    query,
    k = 5,
    options
) {

    if (!query || typeof query !== "string" || !query.trim()) {
        throw new Error("Query cannot be empty.");
    }

    // Convert user question into a vector.
    const queryVector = await embedQuery(query);

    // Search for nearest vectors.
    if (options === undefined) {
        return index.search(queryVector, k);
    }

    return index.search(queryVector, k, options);
}


import { searchText } from "./vectorSearchService.js";
import { generateAnswer } from "./llmService.js";

export async function askDocuments(index, question, k = 3) {

    // Find relevant document chunks.
    const retrievedDocuments = await searchText(
        index,
        question,
        k
    );

    // Generate an answer using the retrieved information.
    const answer = await generateAnswer(
        question,
        retrievedDocuments
    );

    // Prepare readable source information.
    const sources = retrievedDocuments.map((document, index) => ({
        number: index + 1,
        id: document.id,
        documentId: document.metadata?.documentId ?? "Unknown",
        chunkNumber: document.metadata?.chunkNumber ?? "Unknown",
        file: document.metadata?.source ?? "Unknown",
        distance: document.distance,
        text: document.metadata?.text ?? ""
    }));

    return {
        question,
        answer,
        sources
    };
}

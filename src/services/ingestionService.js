
import { extractPdfText } from "./pdfService.js";
import { chunkText } from "./chunkService.js";
import { insertText } from "./vectorSearchService.js";

// ==========================================
// INGEST PDF DOCUMENT
// ==========================================

export async function ingestPdf(
    index,
    filePath,
    documentId
) {

    if (!documentId || typeof documentId !== "string") {
        throw new Error("A valid document ID is required.");
    }

    // Step 1: Extract PDF text.
    console.log("Reading PDF...");

    const text = await extractPdfText(filePath);

    // Step 2: Divide text into smaller chunks.
    console.log("Creating text chunks...");

    const chunks = chunkText(text, 120, 20);

    console.log(`Total chunks: ${chunks.length}`);

    // Step 3: Generate embeddings and store them.
    console.log("Generating embeddings and storing vectors...");

    for (let i = 0; i < chunks.length; i++) {

        const chunkId = `${documentId}-chunk-${i + 1}`;

        await insertText(
            index,
            chunkId,
            chunks[i],
            {
                documentId,
                chunkNumber: i + 1,
                totalChunks: chunks.length,
                source: filePath
            }
        );

        console.log(`Stored chunk ${i + 1}/${chunks.length}`);
    }

    return {
        documentId,
        totalCharacters: text.length,
        totalChunks: chunks.length,
        storedVectors: chunks.length
    };
}

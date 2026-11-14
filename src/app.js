
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { VectorIndexManager } from "./core/indexManager.js";
import { ingestPdf } from "./services/ingestionService.js";
import { askDocuments } from "./services/ragService.js";

const STORAGE_PATH = "./data/vectors.json";

async function main() {

    const rl = readline.createInterface({
        input: stdin,
        output: stdout
    });

    try {

        const index = new VectorIndexManager({
            algorithm: "bruteForce",
            options: {
                metric: "cosine"
            }
        });

        // Restore previously saved documents.
        try {

            await index.load(STORAGE_PATH);

            console.log(
                `Restored ${index.getDocuments().length} documents.`
            );

        } catch (error) {

            if (error.code === "ENOENT") {
                console.log("Starting a new collection.");
            } else {
                throw error;
            }
        }

        console.log("\n================================");
        console.log("       PERSONAL AI ASSISTANT");
        console.log("================================");

        while (true) {

            console.log("\n1. Add PDF document");
            console.log("2. Ask a question");
            console.log("3. View documents");
            console.log("4. Delete a document");
            console.log("5. Exit");

            const choice = (
                await rl.question("\nChoose an option: ")
            ).trim();

            // ======================================
            // ADD PDF
            // ======================================

            if (choice === "1") {

                const filePath = (
                    await rl.question("Enter PDF path: ")
                ).trim();

                const documentId = (
                    await rl.question("Enter document ID: ")
                ).trim();

                if (!filePath || !documentId) {
                    console.log("Both fields are required.");
                    continue;
                }

                if (index.hasDocument(documentId)) {
                    console.log("This document ID already exists.");
                    continue;
                }

                if (index.hasSource(filePath)) {
                    console.log("This PDF has already been uploaded.");
                    continue;
                }

                try {

                    console.log("\nProcessing PDF...\n");

                    const summary = await ingestPdf(
                        index,
                        filePath,
                        documentId
                    );

                    await index.save(STORAGE_PATH);

                    console.log("\nDocument saved successfully!");
                    console.log(summary);

                } catch (error) {
                    console.log(
                        "\nCould not process document:",
                        error.message
                    );
                }
            }

            // ======================================
            // ASK QUESTION
            // ======================================

            else if (choice === "2") {

                if (index.getDocuments().length === 0) {
                    console.log("Please add a PDF first.");
                    continue;
                }

                const question = (
                    await rl.question("Enter your question: ")
                ).trim();

                if (!question) {
                    console.log("Question cannot be empty.");
                    continue;
                }

                try {

                    console.log(
                        "\nSearching documents and generating answer...\n"
                    );

                    const result = await askDocuments(
                        index,
                        question,
                        3
                    );

                    console.log("\n========== ANSWER ==========\n");
                    console.log(result.answer);

                    console.log("\n========== SOURCES ==========\n");

                    for (const source of result.sources) {

                        console.log(`Document: ${source.documentId}`);
                        console.log(`Chunk: ${source.chunkNumber}`);
                        console.log(`Content: ${source.text}`);
                        console.log("--------------------------------");
                    }

                } catch (error) {
                    console.log(
                        "Could not generate answer:",
                        error.message
                    );
                }
            }

            // ======================================
            // VIEW DOCUMENTS
            // ======================================

            else if (choice === "3") {

                const documents = index.getDocuments();

                console.log("\n========== YOUR DOCUMENTS ==========\n");

                if (documents.length === 0) {
                    console.log("No documents stored.");
                    continue;
                }

                for (const document of documents) {

                    console.log(`Document ID: ${document.documentId}`);
                    console.log(`File: ${document.source}`);
                    console.log(
                        `Chunks: ${document.chunks}/${document.totalChunks}`
                    );
                    console.log("--------------------------------");
                }

                console.log(`Total stored vectors: ${index.size}`);
            }

            // ======================================
            // DELETE DOCUMENT
            // ======================================

            else if (choice === "4") {

                const documents = index.getDocuments();

                if (documents.length === 0) {
                    console.log("No documents available to delete.");
                    continue;
                }

                console.log("\n========== YOUR DOCUMENTS ==========\n");

                for (const document of documents) {
                    console.log(
                        `${document.documentId} (${document.chunks} chunks)`
                    );
                }

                const documentId = (
                    await rl.question("\nEnter document ID to delete: ")
                ).trim();

                const deleted = index.deleteDocument(documentId);

                if (deleted === 0) {
                    console.log("Document not found.");
                    continue;
                }

                await index.save(STORAGE_PATH);

                console.log("\nDocument deleted successfully!");
                console.log(`Removed ${deleted} chunks.`);
            }

            // ======================================
            // EXIT
            // ======================================

            else if (choice === "5") {

                console.log("\nGoodbye!");
                break;
            }

            else {
                console.log("Invalid choice. Please try again.");
            }
        }

    } finally {
        rl.close();
    }
}

main().catch(error => {
    console.error("\nApplication error:", error.message);
    process.exitCode = 1;
});

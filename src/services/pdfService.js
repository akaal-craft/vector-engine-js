
import { readFile } from "node:fs/promises";
import { PDFParse } from "pdf-parse";

// Extract text from a PDF file.
export async function extractPdfText(filePath) {

    if (!filePath || typeof filePath !== "string") {
        throw new Error("A valid PDF file path is required.");
    }

    // Read PDF from disk.
    const buffer = await readFile(filePath);

    if (buffer.length === 0) {
        throw new Error("PDF file is empty.");
    }

    const parser = new PDFParse({
        data: buffer
    });

    try {

        // Extract all available text.
        const result = await parser.getText();

        const text = result.text?.trim();

        if (!text) {
            throw new Error(
                "No readable text found in this PDF."
            );
        }

        return text;

    } finally {

        // Release parser resources.
        await parser.destroy();
    }
}

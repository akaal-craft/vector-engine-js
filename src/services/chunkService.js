
export function chunkText(
    text,
    chunkSize = 120,
    overlap = 20
) {

    if (typeof text !== "string" || !text.trim()) {
        throw new Error("Text cannot be empty.");
    }

    if (
        !Number.isInteger(chunkSize) ||
        chunkSize <= 0
    ) {
        throw new Error("Chunk size must be positive.");
    }

    if (
        !Number.isInteger(overlap) ||
        overlap < 0 ||
        overlap >= chunkSize
    ) {
        throw new Error(
            "Overlap must be non-negative and smaller than chunk size."
        );
    }

    const words = text.trim().split(/\s+/);

    const chunks = [];

    let start = 0;

    while (start < words.length) {

        const end = Math.min(
            start + chunkSize,
            words.length
        );

        const chunk = words
            .slice(start, end)
            .join(" ");

        chunks.push(chunk);

        start += chunkSize - overlap;
    }

    return chunks;
}

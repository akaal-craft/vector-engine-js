// ==========================================
// VECTOR VALIDATION
// ==========================================

function validateVector(vector, name = "vector") {
    if (!Array.isArray(vector)) {
        throw new TypeError(`${name} must be an array.`);
    }

    if (vector.length === 0) {
        throw new RangeError(`${name} cannot be empty.`);
    }

    for (let i = 0; i < vector.length; i++) {
        if (
            typeof vector[i] !== "number" ||
            !Number.isFinite(vector[i])
        ) {
            throw new TypeError(
                `${name}[${i}] must be a finite number.`
            );
        }
    }
}


function validatePair(a, b) {
    validateVector(a, "Vector A");
    validateVector(b, "Vector B");

    if (a.length !== b.length) {
        throw new RangeError(
            "Vectors must have the same number of dimensions."
        );
    }
}


// ==========================================
// 1. DOT PRODUCT
// ==========================================

export function dotProduct(a, b) {
    validatePair(a, b);

    let result = 0;

    for (let i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }

    return result;
}


// ==========================================
// 2. VECTOR MAGNITUDE
// ==========================================

export function magnitude(vector) {
    validateVector(vector);

    return Math.sqrt(
        dotProduct(vector, vector)
    );
}


// ==========================================
// 3. EUCLIDEAN DISTANCE
// ==========================================

export function euclideanDistance(a, b) {
    validatePair(a, b);

    let sum = 0;

    for (let i = 0; i < a.length; i++) {
        const difference = a[i] - b[i];

        sum += difference * difference;
    }

    return Math.sqrt(sum);
}


// ==========================================
// 4. MANHATTAN DISTANCE
// ==========================================

export function manhattanDistance(a, b) {
    validatePair(a, b);

    let sum = 0;

    for (let i = 0; i < a.length; i++) {
        sum += Math.abs(a[i] - b[i]);
    }

    return sum;
}


// ==========================================
// 5. COSINE SIMILARITY
// ==========================================

export function cosineSimilarity(a, b) {
    validatePair(a, b);

    const magnitudeA = magnitude(a);
    const magnitudeB = magnitude(b);

    const denominator = magnitudeA * magnitudeB;

    if (denominator === 0) {
        throw new RangeError(
            "Cosine similarity is undefined for zero vectors."
        );
    }

    const similarity =
        dotProduct(a, b) / denominator;

    // Protect against tiny floating-point errors.
    return Math.max(
        -1,
        Math.min(1, similarity)
    );
}


// ==========================================
// 6. COSINE DISTANCE
// ==========================================

export function cosineDistance(a, b) {
    return 1 - cosineSimilarity(a, b);
}
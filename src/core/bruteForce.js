import {
    euclideanDistance,
    manhattanDistance,
    cosineDistance,
    magnitude
} from "./distance.js";

// Supported distance metrics
const DISTANCE_METRICS = {
    euclidean: euclideanDistance,
    manhattan: manhattanDistance,
    cosine: cosineDistance
};


// Validate vector input
function validateVector(vector, name = "vector") {
    if (!Array.isArray(vector) || vector.length === 0) {
        throw new TypeError(
            `${name} must be a non-empty array.`
        );
    }

    for (const value of vector) {
        if (
            typeof value !== "number" ||
            !Number.isFinite(value)
        ) {
            throw new TypeError(
                `${name} must contain only finite numbers.`
            );
        }
    }
}


// ==========================================
// BRUTE FORCE VECTOR INDEX
// ==========================================

export class BruteForceIndex {

    constructor({ metric = "cosine" } = {}) {

        if (!DISTANCE_METRICS[metric]) {
            throw new Error(
                `Unsupported distance metric: ${metric}`
            );
        }

        this.metric = metric;

        this.distanceFunction =
            DISTANCE_METRICS[metric];

        this.records = new Map();

        this.dimensions = null;
    }


    // ======================================
    // INSERT VECTOR
    // ======================================

    insert(id, vector, metadata = {}) {

        if (
            typeof id !== "string" ||
            id.trim().length === 0
        ) {
            throw new TypeError(
                "Vector ID must be a non-empty string."
            );
        }

        validateVector(vector);

        if (
            metadata === null ||
            typeof metadata !== "object" ||
            Array.isArray(metadata)
        ) {
            throw new TypeError(
                "Metadata must be an object."
            );
        }

        // Every vector must have the same dimensions.
        if (
            this.dimensions !== null &&
            vector.length !== this.dimensions
        ) {
            throw new Error(
                "Vector dimensions do not match the index."
            );
        }

        // Cosine distance is undefined for zero vectors.
        if (
            this.metric === "cosine" &&
            magnitude(vector) === 0
        ) {
            throw new Error(
                "Zero vectors are not supported with cosine distance."
            );
        }

        if (this.dimensions === null) {
            this.dimensions = vector.length;
        }

        // Map.set replaces an existing ID.
        this.records.set(id, {
            vector: [...vector],
            metadata: { ...metadata }
        });

        return id;
    }


    // ======================================
    // SEARCH NEAREST VECTORS
    // ======================================

    search(queryVector, k = 5) {

        validateVector(queryVector, "Query vector");

        if (
            !Number.isInteger(k) ||
            k <= 0
        ) {
            throw new Error(
                "K must be a positive integer."
            );
        }

        if (
            this.dimensions !== null &&
            queryVector.length !== this.dimensions
        ) {
            throw new Error(
                "Query vector dimensions do not match the index."
            );
        }

        if (
            this.metric === "cosine" &&
            magnitude(queryVector) === 0
        ) {
            throw new Error(
                "Query vector cannot be zero for cosine distance."
            );
        }

        const results = [];

        // Compare query against EVERY stored vector.
        for (const [id, record] of this.records) {

            const distance = this.distanceFunction(
                queryVector,
                record.vector
            );

            results.push({
                id,
                distance,
                metadata: { ...record.metadata }
            });
        }

        // Smaller distance means closer vector.
        results.sort(
            (a, b) => a.distance - b.distance
        );

        // Return only the nearest K results.
        return results.slice(0, k);
    }


    // ======================================
    // DELETE VECTOR
    // ======================================

    delete(id) {
        return this.records.delete(id);
    }


    // ======================================
    // CLEAR INDEX
    // ======================================

    clear() {
        this.records.clear();
        this.dimensions = null;
    }


    // ======================================
    // GET VECTOR COUNT
    // ======================================

    get size() {
        return this.records.size;
    }
}

import { performance } from "node:perf_hooks";

import { HNSWIndex } from "../src/core/hnsw.js";
import { BruteForceIndex } from "../src/core/bruteForce.js";

// ==========================================
// CONFIGURATION
// ==========================================

const VECTOR_COUNT = 5000;
const DIMENSIONS = 768;
const QUERY_COUNT = 50;
const K = 10;

const EF_VALUES = [50, 100, 150];

// ==========================================
// DETERMINISTIC RANDOM GENERATOR
// ==========================================

function createRandom(seed = 12345) {
    let state = seed;

    return function () {
        state = (
            Math.imul(state, 1664525) + 1013904223
        ) >>> 0;

        return state / 4294967296;
    };
}

// ==========================================
// VECTOR GENERATION
// ==========================================

function generateVector(random, dimensions) {
    const vector = [];

    for (let i = 0; i < dimensions; i++) {
        vector.push(random() * 2 - 1);
    }

    return vector;
}

// ==========================================
// NORMALIZE VECTOR
// ==========================================

function normalize(vector) {
    const magnitude = Math.sqrt(
        vector.reduce(
            (sum, value) => sum + value * value,
            0
        )
    );

    if (magnitude === 0) {
        throw new Error("Cannot normalize a zero vector.");
    }

    return vector.map(value => value / magnitude);
}

// ==========================================
// RECALL CALCULATION
// ==========================================

function calculateRecall(actual, predicted) {
    const actualIds = new Set(
        actual.map(item => item.id)
    );

    let correct = 0;

    for (const item of predicted) {
        if (actualIds.has(item.id)) {
            correct++;
        }
    }

    return actual.length === 0
        ? 0
        : correct / actual.length;
}

// ==========================================
// STATISTICS
// ==========================================

function average(values) {
    return values.reduce(
        (sum, value) => sum + value,
        0
    ) / values.length;
}

function percentile(values, p) {
    const sorted = [...values].sort(
        (a, b) => a - b
    );

    const index = Math.ceil(p * sorted.length) - 1;

    return sorted[Math.max(0, index)];
}

// ==========================================
// MEASURE SEARCH LATENCY
// ==========================================

function measureSearch(index, query, k, options) {
    const start = performance.now();

    const results = options
        ? index.search(query, k, options)
        : index.search(query, k);

    const end = performance.now();

    return {
        results,
        latency: end - start
    };
}

// ==========================================
// GENERATE DATASET
// ==========================================

console.log("\n===== GENERATING DATASET =====");

console.log(`Vectors: ${VECTOR_COUNT}`);
console.log(`Dimensions: ${DIMENSIONS}`);
console.log(`Queries: ${QUERY_COUNT}`);

const dataRandom = createRandom(12345);

const vectors = [];

for (let i = 0; i < VECTOR_COUNT; i++) {
    const vector = normalize(
        generateVector(dataRandom, DIMENSIONS)
    );

    vectors.push({
        id: `V${i}`,
        vector
    });
}

// ==========================================
// GENERATE QUERY VECTORS
// ==========================================

const queryRandom = createRandom(98765);

const queries = [];

for (let i = 0; i < QUERY_COUNT; i++) {
    const query = normalize(
        generateVector(queryRandom, DIMENSIONS)
    );

    queries.push(query);
}

// ==========================================
// INITIALIZE INDEXES
// ==========================================

const layerRandom = createRandom(45678);

const hnsw = new HNSWIndex({
    metric: "cosine",
    M: 16,
    efConstruction: 100,
    efSearch: 100,
    random: layerRandom
});
const brute = new BruteForceIndex({
    metric: "cosine"
});

// ==========================================
// BUILD HNSW INDEX
// ==========================================

console.log("\n===== INDEX CONSTRUCTION =====");

let start = performance.now();

for (const item of vectors) {
    hnsw.insert(
        item.id,
        item.vector
    );
}

const hnswBuildTime = performance.now() - start;

console.log(
    `HNSW build time: ${hnswBuildTime.toFixed(2)} ms`
);

// ==========================================
// BUILD BRUTE FORCE INDEX
// ==========================================

start = performance.now();

for (const item of vectors) {
    brute.insert(
        item.id,
        item.vector
    );
}

const bruteBuildTime = performance.now() - start;

console.log(
    `Brute Force build time: ${bruteBuildTime.toFixed(2)} ms`
);

// ==========================================
// GENERATE EXACT GROUND TRUTH
// ==========================================

console.log("\n===== GENERATING EXACT RESULTS =====");

const exactResults = queries.map(query =>
    brute.search(query, K)
);

// ==========================================
// WARM-UP
// ==========================================

console.log("\n===== SEARCH WARM-UP =====");

for (let i = 0; i < Math.min(5, QUERY_COUNT); i++) {
    brute.search(queries[i], K);

    hnsw.search(
        queries[i],
        K,
        { efSearch: 100 }
    );
}

// ==========================================
// BRUTE FORCE BENCHMARK
// ==========================================

console.log("\n===== BRUTE FORCE SEARCH BASELINE =====");

const bruteLatencies = [];

for (const query of queries) {
    const measurement = measureSearch(
        brute,
        query,
        K,
        null
    );

    bruteLatencies.push(measurement.latency);
}

const bruteAverage = average(bruteLatencies);

const bruteP95 = percentile(
    bruteLatencies,
    0.95
);

console.log(
    `Average latency: ${bruteAverage.toFixed(4)} ms`
);

console.log(
    `P95 latency: ${bruteP95.toFixed(4)} ms`
);

// ==========================================
// HNSW BENCHMARK
// ==========================================

console.log("\n===== HNSW SEARCH BENCHMARK =====");

console.log(`Vectors: ${VECTOR_COUNT}`);
console.log(`Dimensions: ${DIMENSIONS}`);
console.log(`Queries: ${QUERY_COUNT}`);
console.log(`K: ${K}`);

for (const ef of EF_VALUES) {

    const latencies = [];

    let totalRecall = 0;

    for (let i = 0; i < QUERY_COUNT; i++) {

        const measurement = measureSearch(
            hnsw,
            queries[i],
            K,
            { efSearch: ef }
        );

        latencies.push(measurement.latency);

        totalRecall += calculateRecall(
            exactResults[i],
            measurement.results
        );
    }

    const averageRecall = (
        totalRecall / QUERY_COUNT
    );

    const averageLatency = average(latencies);

    const p95Latency = percentile(
        latencies,
        0.95
    );

    const speedup = (
        bruteAverage / averageLatency
    );

    console.log("\n--------------------------------");

    console.log(`efSearch: ${ef}`);

    console.log(
        `Recall@${K}: ${(averageRecall * 100).toFixed(2)}%`
    );

    console.log(
        `Average latency: ${averageLatency.toFixed(4)} ms`
    );

    console.log(
        `P95 latency: ${p95Latency.toFixed(4)} ms`
    );

    console.log(
        `Speedup vs Brute Force: ${speedup.toFixed(2)}x`
    );
}

// ==========================================
// GRAPH STATISTICS
// ==========================================

console.log("\n===== HNSW GRAPH STATISTICS =====");

console.log(
    JSON.stringify(
        hnsw.getStats(),
        null,
        2
    )
);

console.log("\n===== BENCHMARK COMPLETED =====");

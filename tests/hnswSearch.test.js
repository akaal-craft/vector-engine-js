
import test from "node:test";
import assert from "node:assert/strict";

import { HNSWIndex } from "../src/core/hnsw.js";
import { BruteForceIndex } from "../src/core/bruteForce.js";


function createIndexes() {

    const hnsw = new HNSWIndex({
        metric: "euclidean",
        M: 8,
        efConstruction: 30,
        efSearch: 20,
        random: () => 0.5
    });

    const brute = new BruteForceIndex({
        metric: "euclidean"
    });

    const vectors = [
        { id: "A", vector: [0, 0] },
        { id: "B", vector: [5, 0] },
        { id: "C", vector: [2, 0] },
        { id: "D", vector: [1, 0] },
        { id: "E", vector: [3, 1] }
    ];

    for (const item of vectors) {

        hnsw.insert(
            item.id,
            item.vector,
            { label: item.id }
        );

        brute.insert(
            item.id,
            item.vector,
            { label: item.id }
        );
    }

    return { hnsw, brute };
}


// ==========================================
// NEAREST NEIGHBOR
// ==========================================

test("Find nearest vector", () => {

    const { hnsw } = createIndexes();

    const results = hnsw.search(
        [1.1, 0],
        1
    );

    assert.equal(results[0].id, "D");
});


// ==========================================
// TOP K
// ==========================================

test("Return exactly K nearest vectors", () => {

    const { hnsw } = createIndexes();

    const results = hnsw.search(
        [1.1, 0],
        3
    );

    assert.equal(results.length, 3);

    assert.deepEqual(
        results.map(result => result.id),
        ["D", "C", "A"]
    );
});


// ==========================================
// DISTANCE ORDERING
// ==========================================

test("Return results in ascending distance order", () => {

    const { hnsw } = createIndexes();

    const results = hnsw.search(
        [1.1, 0],
        4
    );

    for (let i = 1; i < results.length; i++) {

        assert.ok(
            results[i - 1].distance <=
            results[i].distance
        );
    }
});


// ==========================================
// BRUTE FORCE COMPARISON
// ==========================================

test("HNSW matches Brute Force on small test dataset", () => {

    const { hnsw, brute } = createIndexes();

    const query = [1.1, 0];

    const hnswResults = hnsw.search(
        query,
        3
    );

    const bruteResults = brute.search(
        query,
        3
    );

    assert.deepEqual(
        hnswResults.map(result => result.id),
        bruteResults.map(result => result.id)
    );
});


// ==========================================
// METADATA
// ==========================================

test("Preserve metadata in search results", () => {

    const { hnsw } = createIndexes();

    const results = hnsw.search(
        [0, 0],
        1
    );

    assert.equal(results[0].metadata.label, "A");
});


// ==========================================
// EMPTY INDEX
// ==========================================

test("Return empty results for empty index", () => {

    const index = new HNSWIndex();

    assert.deepEqual(
        index.search([1, 2], 5),
        []
    );
});


// ==========================================
// INVALID K
// ==========================================

test("Reject invalid K", () => {

    const { hnsw } = createIndexes();

    assert.throws(
        () => hnsw.search([1, 0], 0),
        /positive integer/
    );
});


// ==========================================
// INVALID efSearch
// ==========================================

test("Reject invalid efSearch", () => {

    const { hnsw } = createIndexes();

    assert.throws(
        () => hnsw.search(
            [1, 0],
            2,
            { efSearch: 0 }
        ),
        /efSearch must be/
    );
});


// ==========================================
// DIMENSION VALIDATION
// ==========================================

test("Reject incorrect query dimensions", () => {

    const { hnsw } = createIndexes();

    assert.throws(
        () => hnsw.search([1, 2, 3], 2),
        /dimensions/
    );
});


// ==========================================
// EF SEARCH BEHAVIOUR
// ==========================================

test("Allow efSearch smaller than K by expanding search width", () => {

    const { hnsw } = createIndexes();

    const results = hnsw.search(
        [1.1, 0],
        3,
        { efSearch: 1 }
    );

    assert.equal(results.length, 3);
});

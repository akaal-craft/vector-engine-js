import test from "node:test";
import assert from "node:assert/strict";

import { KDTreeIndex } from "../src/core/kdTree.js";
import { BruteForceIndex } from "../src/core/bruteForce.js";


// ==========================================
// TEST DATA
// ==========================================

const sampleVectors = [
    { id: "A", vector: [1, 0] },
    { id: "B", vector: [0, 1] },
    { id: "C", vector: [0.9, 0.1] },
    { id: "D", vector: [-1, 0] },
    { id: "E", vector: [0.5, 0.5] },
    { id: "F", vector: [2, 3] }
];


function createIndex(IndexClass, metric) {

    const index = new IndexClass({ metric });

    for (const item of sampleVectors) {
        index.insert(item.id, item.vector);
    }

    return index;
}


// ==========================================
// BASIC INSERTION
// ==========================================

test("Insert vectors into KD-Tree", () => {

    const index = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    assert.equal(index.size, 6);
});


// ==========================================
// NEAREST NEIGHBOR
// ==========================================

test("Find nearest vector", () => {

    const index = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    const results = index.search([1, 0], 1);

    assert.equal(results[0].id, "A");
});


// ==========================================
// TOP K RESULTS
// ==========================================

test("Return top K nearest vectors", () => {

    const index = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    const results = index.search([1, 0], 2);

    assert.equal(results.length, 2);

    assert.deepEqual(
        results.map(result => result.id),
        ["A", "C"]
    );
});


// ==========================================
// COMPARE WITH BRUTE FORCE
// ==========================================

test("KD-Tree matches Brute Force for Euclidean search", () => {

    const kdTree = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    const bruteForce = createIndex(
        BruteForceIndex,
        "euclidean"
    );

    const queries = [
        [1, 0],
        [0.2, 0.4],
        [3, 2],
        [-1, 0]
    ];

    for (const query of queries) {

        const kdResults = kdTree.search(query, 3);

        const bruteResults =
            bruteForce.search(query, 3);

        assert.deepEqual(
            kdResults.map(result => result.id),
            bruteResults.map(result => result.id)
        );
    }
});


// ==========================================
// MANHATTAN DISTANCE
// ==========================================

test("KD-Tree supports Manhattan distance", () => {

    const kdTree = createIndex(
        KDTreeIndex,
        "manhattan"
    );

    const bruteForce = createIndex(
        BruteForceIndex,
        "manhattan"
    );

    const query = [0.5, 0.3];

    assert.deepEqual(
        kdTree.search(query, 3).map(x => x.id),
        bruteForce.search(query, 3).map(x => x.id)
    );
});


// ==========================================
// METADATA
// ==========================================

test("Preserve vector metadata", () => {

    const index = new KDTreeIndex({
        metric: "euclidean"
    });

    index.insert(
        "A",
        [1, 0],
        { text: "Binary Search Tree", page: 30 }
    );

    const result = index.search([1, 0], 1);

    assert.equal(
        result[0].metadata.text,
        "Binary Search Tree"
    );
});


// ==========================================
// INSERT OR UPDATE
// ==========================================

test("Update an existing vector ID", () => {

    const index = new KDTreeIndex({
        metric: "euclidean"
    });

    index.insert("A", [10, 10]);

    index.insert("A", [0, 0]);

    assert.equal(index.size, 1);

    assert.equal(
        index.search([0, 0], 1)[0].id,
        "A"
    );
});


// ==========================================
// DELETE
// ==========================================

test("Delete a vector and rebuild the tree", () => {

    const index = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    assert.equal(index.delete("A"), true);

    assert.equal(index.size, 5);

    assert.notEqual(
        index.search([1, 0], 1)[0].id,
        "A"
    );
});


// ==========================================
// DIMENSION VALIDATION
// ==========================================

test("Reject vectors with incorrect dimensions", () => {

    const index = new KDTreeIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    assert.throws(
        () => index.insert("B", [1, 2, 3]),
        /dimensions/
    );
});


// ==========================================
// COSINE NOT SUPPORTED
// ==========================================

test("Reject unsupported cosine metric", () => {

    assert.throws(
        () => new KDTreeIndex({ metric: "cosine" }),
        /does not support/
    );
});


// ==========================================
// INVALID K
// ==========================================

test("Reject invalid K", () => {

    const index = createIndex(
        KDTreeIndex,
        "euclidean"
    );

    assert.throws(
        () => index.search([1, 0], 0),
        /positive integer/
    );
});


// ==========================================
// PRUNING
// ==========================================

test("KD-Tree prunes distant branches", () => {

    const index = new KDTreeIndex({
        metric: "euclidean"
    });

    // Create 1000 points in one dimension.
    for (let i = 0; i < 1000; i++) {
        index.insert(`V${i}`, [i]);
    }

    const result = index.search([0], 1);

    assert.equal(result[0].id, "V0");

    assert.ok(
        index.lastSearchStats.visitedNodes < 1000
    );
});


// ==========================================
// EMPTY INDEX
// ==========================================

test("Return empty results for empty index", () => {

    const index = new KDTreeIndex();

    assert.deepEqual(
        index.search([1, 2], 3),
        []
    );
});
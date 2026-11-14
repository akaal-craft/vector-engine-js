import test from "node:test";
import assert from "node:assert/strict";

import {
    dotProduct,
    magnitude,
    euclideanDistance,
    manhattanDistance,
    cosineSimilarity,
    cosineDistance
} from "../src/core/distance.js";


// Helper for floating-point comparisons
function approximatelyEqual(actual, expected) {
    assert.ok(
        Math.abs(actual - expected) < 1e-10,
        `Expected ${expected}, received ${actual}`
    );
}


// DOT PRODUCT
test("Dot product", () => {
    assert.equal(
        dotProduct([1, 2, 3], [4, 5, 6]),
        32
    );
});


// MAGNITUDE
test("Vector magnitude", () => {
    approximatelyEqual(
        magnitude([3, 4]),
        5
    );
});


// EUCLIDEAN DISTANCE
test("Euclidean distance", () => {
    approximatelyEqual(
        euclideanDistance([1, 0], [0, 1]),
        Math.sqrt(2)
    );
});


// MANHATTAN DISTANCE
test("Manhattan distance", () => {
    assert.equal(
        manhattanDistance([1, 2], [4, 6]),
        7
    );
});


// COSINE SIMILARITY
test("Cosine similarity", () => {
    assert.equal(
        cosineSimilarity([1, 0], [0, 1]),
        0
    );

    assert.equal(
        cosineSimilarity([1, 0], [1, 0]),
        1
    );
});


// COSINE DISTANCE
test("Cosine distance", () => {
    assert.equal(
        cosineDistance([1, 0], [0, 1]),
        1
    );

    assert.equal(
        cosineDistance([1, 0], [-1, 0]),
        2
    );
});


// DIMENSION VALIDATION
test("Reject vectors with different dimensions", () => {
    assert.throws(
        () => dotProduct([1, 2], [1, 2, 3]),
        /same number of dimensions/
    );
});


// ZERO VECTOR VALIDATION
test("Reject zero vectors for cosine similarity", () => {
    assert.throws(
        () => cosineSimilarity([0, 0], [1, 2]),
        /zero vectors/
    );
});


// EMPTY VECTOR VALIDATION
test("Reject empty vectors", () => {
    assert.throws(
        () => magnitude([]),
        /cannot be empty/
    );
});


// INVALID NUMBER VALIDATION
test("Reject NaN values", () => {
    assert.throws(
        () => magnitude([1, NaN]),
        /finite number/
    );
});
import test from "node:test";
import assert from "node:assert/strict";

import { BruteForceIndex } from "../src/core/bruteForce.js";


test("Insert vectors and check count", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);
    index.insert("B", [0, 1]);

    assert.equal(index.size, 2);
});


test("Find nearest vector", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);
    index.insert("B", [0, 1]);
    index.insert("C", [0.9, 0.1]);

    const results = index.search([1, 0], 1);

    assert.equal(results[0].id, "A");
});


test("Return top K nearest vectors", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);
    index.insert("B", [0, 1]);
    index.insert("C", [0.9, 0.1]);
    index.insert("D", [-1, 0]);

    const results = index.search([1, 0], 2);

    assert.equal(results.length, 2);
    assert.deepEqual(
        results.map(result => result.id),
        ["A", "C"]
    );
});


test("Results must be sorted by distance", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [5, 5]);
    index.insert("B", [1, 1]);
    index.insert("C", [2, 2]);

    const results = index.search([0, 0], 3);

    assert.ok(
        results[0].distance < results[1].distance
    );

    assert.ok(
        results[1].distance < results[2].distance
    );
});


test("Support cosine distance", () => {

    const index = new BruteForceIndex({
        metric: "cosine"
    });

    index.insert("A", [1, 0]);
    index.insert("B", [0, 1]);

    const results = index.search([1, 0], 1);

    assert.equal(results[0].id, "A");
    assert.equal(results[0].distance, 0);
});


test("Support metadata", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert(
        "A",
        [1, 0],
        { text: "Binary Search Tree", page: 30 }
    );

    const results = index.search([1, 0], 1);

    assert.equal(
        results[0].metadata.text,
        "Binary Search Tree"
    );
});


test("Delete vector", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    assert.equal(index.delete("A"), true);
    assert.equal(index.size, 0);
});


test("Reject incorrect dimensions", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    assert.throws(
        () => index.insert("B", [1, 2, 3]),
        /dimensions/
    );
});


test("Reject invalid K", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    assert.throws(
        () => index.search([1, 0], 0),
        /positive integer/
    );
});


test("Update existing vector ID without duplication", () => {

    const index = new BruteForceIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    index.insert("A", [5, 5]);

    assert.equal(index.size, 1);

    const results = index.search([5, 5], 1);

    assert.equal(results[0].id, "A");
});
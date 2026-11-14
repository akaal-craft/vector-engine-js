
import test from "node:test";
import assert from "node:assert/strict";

import { VectorIndexManager } from "../src/core/indexManager.js";

// All three algorithms should support the common interface.
const algorithms = [
    "bruteForce",
    "kdTree",
    "hnsw"
];

// ==========================================
// TEST 1: INITIALIZATION
// ==========================================

test("Initialize manager with all supported algorithms", () => {

    for (const algorithm of algorithms) {

        const manager = new VectorIndexManager({
            algorithm,
            options: {
                metric: "euclidean",
                random: () => 0.5
            }
        });

        assert.equal(manager.algorithm, algorithm);
        assert.equal(manager.size, 0);
    }
});

// ==========================================
// TEST 2: INSERTION AND SEARCH
// ==========================================

test("Insert and search vectors through manager", () => {

    for (const algorithm of algorithms) {

        const manager = new VectorIndexManager({
            algorithm,
            options: {
                metric: "euclidean",
                M: 8,
                efConstruction: 100,
                efSearch: 50,
                random: () => 0.5
            }
        });

        manager.insert("A", [0, 0], {
            label: "Origin"
        });

        manager.insert("B", [1, 1], {
            label: "Near"
        });

        manager.insert("C", [8, 8], {
            label: "Far"
        });

        assert.equal(manager.size, 3);

        const results = manager.search([0.2, 0.1], 2);

        assert.equal(results.length, 2);

        assert.equal(results[0].id, "A");
        assert.equal(results[1].id, "B");

        assert.ok(
            results[0].distance <= results[1].distance
        );
    }
});

// ==========================================
// TEST 3: METADATA
// ==========================================

test("Preserve metadata through manager", () => {

    const manager = new VectorIndexManager({
        algorithm: "bruteForce",
        options: {
            metric: "euclidean"
        }
    });

    manager.insert("A", [1, 2], {
        text: "Apple",
        category: "Fruit"
    });

    const results = manager.search([1, 2], 1);

    assert.equal(results[0].metadata.text, "Apple");
    assert.equal(results[0].metadata.category, "Fruit");
});

// ==========================================
// TEST 4: STATISTICS
// ==========================================

test("Return manager statistics", () => {

    const manager = new VectorIndexManager({
        algorithm: "hnsw",
        options: {
            metric: "euclidean",
            random: () => 0.5
        }
    });

    manager.insert("A", [1, 2]);

    const stats = manager.getStats();

    assert.equal(stats.algorithm, "hnsw");
    assert.equal(stats.size, 1);
});

// ==========================================
// TEST 5: INVALID ALGORITHM
// ==========================================

test("Reject unsupported algorithm", () => {

    assert.throws(
        () => {
            new VectorIndexManager({
                algorithm: "invalid"
            });
        },
        /Unsupported index algorithm/
    );
});

// ==========================================
// TEST 6: INVALID OPTIONS
// ==========================================

test("Reject invalid index options", () => {

    assert.throws(
        () => {
            new VectorIndexManager({
                algorithm: "bruteForce",
                options: []
            });
        },
        /Index options must be an object/
    );
});

test("Hide internal index implementation", () => {

    const manager = new VectorIndexManager({
        algorithm: "bruteForce"
    });

    assert.equal(manager.index, undefined);

    assert.equal(manager.size, 0);

    assert.equal(
        typeof manager.insert,
        "function"
    );
});

test("Update and delete vectors through all algorithms", () => {

    const algorithms = [
        "bruteForce",
        "kdTree",
        "hnsw"
    ];

    for (const algorithm of algorithms) {

        const manager = new VectorIndexManager({
            algorithm,
            options: {
                metric: "euclidean",
                random: () => 0.5
            }
        });

        manager.insert("A", [0, 0], {
            label: "Original"
        });

        manager.insert("B", [5, 5]);

        // Update A's vector and metadata.
        manager.update("A", [4.9, 4.9], {
            label: "Updated"
        });

        const results = manager.search([4.9, 4.9], 1);

        assert.equal(results[0].id, "A");
        assert.equal(results[0].metadata.label, "Updated");
        assert.equal(manager.size, 2);

        // Delete B.
        assert.equal(manager.delete("B"), true);
        assert.equal(manager.size, 1);

        // Deleting a missing ID returns false.
        assert.equal(manager.delete("Missing"), false);
    }
});

test("Reject update for missing vector ID", () => {

    const manager = new VectorIndexManager({
        algorithm: "bruteForce"
    });

    assert.throws(
        () => manager.update("Missing", [1, 2]),
        /does not exist/
    );
});

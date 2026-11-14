
import test from "node:test";
import assert from "node:assert/strict";

import { HNSWIndex } from "../src/core/hnsw.js";


// ==========================================
// HELPER: MANUAL TEST GRAPH
// ==========================================

function createTestGraph() {

    const index = new HNSWIndex({
        metric: "euclidean"
    });

    const definitions = [
        ["A", [0, 0]],
        ["B", [5, 0]],
        ["C", [2, 0]],
        ["D", [1, 0]]
    ];

    for (const [id, vector] of definitions) {
        index.nodes.set(
            id,
            index._createNode(id, vector, {}, 0)
        );
    }

    index.dimensions = 2;
    index.entryPoint = "A";
    index.maxLevel = 0;

    const connections = [
        ["A", "B"],
        ["A", "D"],
        ["B", "C"],
        ["C", "D"]
    ];

    for (const [a, b] of connections) {
        index.nodes.get(a).neighbors[0].add(b);
        index.nodes.get(b).neighbors[0].add(a);
    }

    return index;
}


// ==========================================
// CONFIGURATION
// ==========================================

test("Initialize HNSW index", () => {

    const index = new HNSWIndex();

    assert.equal(index.size, 0);
    assert.equal(index.maxLevel, -1);
    assert.equal(index.entryPoint, null);
});


test("Reject invalid M", () => {

    assert.throws(
        () => new HNSWIndex({ M: 1 }),
        /M must be/
    );
});


test("Reject unsupported metric", () => {

    assert.throws(
        () => new HNSWIndex({ metric: "invalid" }),
        /Unsupported metric/
    );
});


// ==========================================
// RANDOM LAYER ASSIGNMENT
// ==========================================

test("Assign level zero for random value 0.5", () => {

    const index = new HNSWIndex({
        M: 4,
        random: () => 0.5
    });

    assert.equal(index._randomLevel(), 0);
});


test("Assign level three for random value 0.01", () => {

    const index = new HNSWIndex({
        M: 4,
        random: () => 0.01
    });

    assert.equal(index._randomLevel(), 3);
});


// ==========================================
// NODE CREATION
// ==========================================

test("Create graph node", () => {

    const index = new HNSWIndex();

    const node = index._createNode(
        "A",
        [1, 2, 3],
        { text: "Binary Search Tree" },
        2
    );

    assert.equal(node.id, "A");
    assert.equal(node.level, 2);
    assert.equal(node.neighbors.length, 3);

    assert.equal(
        node.metadata.text,
        "Binary Search Tree"
    );
});


test("Every layer has an independent neighbor set", () => {

    const index = new HNSWIndex();

    const node = index._createNode(
        "A",
        [1, 2],
        {},
        2
    );

    assert.equal(node.neighbors.length, 3);

    assert.notEqual(
        node.neighbors[0],
        node.neighbors[1]
    );
});


test("Node stores its own vector copy", () => {

    const index = new HNSWIndex();

    const vector = [1, 2, 3];

    const node = index._createNode(
        "A",
        vector,
        {},
        0
    );

    vector[0] = 100;

    assert.equal(node.vector[0], 1);
});


// ==========================================
// GREEDY SEARCH
// ==========================================

test("Greedy search finds a closer neighbor", () => {

    const index = createTestGraph();

    const result = index._greedySearch(
        [1, 0],
        "A",
        0
    );

    assert.equal(result.id, "D");
    assert.equal(result.distance, 0);
});


// ==========================================
// LAYER SEARCH
// ==========================================

test("Layer search returns nearest candidate", () => {

    const index = createTestGraph();

    const results = index._searchLayer(
        [1.1, 0],
        ["A"],
        1,
        0
    );

    assert.equal(results[0].id, "D");
});


test("Layer search returns ordered top K candidates", () => {

    const index = createTestGraph();

    const results = index._searchLayer(
        [1.1, 0],
        ["A"],
        3,
        0
    );

    assert.deepEqual(
        results.map(result => result.id),
        ["D", "C", "A"]
    );
});


test("Layer search respects ef", () => {

    const index = createTestGraph();

    const results = index._searchLayer(
        [1, 0],
        ["A"],
        2,
        0
    );

    assert.equal(results.length, 2);
});


test("Layer search handles duplicate entry points", () => {

    const index = createTestGraph();

    const results = index._searchLayer(
        [1, 0],
        ["A", "A"],
        3,
        0
    );

    assert.equal(
        results.filter(result => result.id === "A").length,
        1
    );
});


// ==========================================
// NAVIGATION
// ==========================================

test("Navigate from upper layer to layer zero", () => {

    const index = new HNSWIndex({
        metric: "euclidean"
    });

    const entry = index._createNode(
        "A",
        [10, 0],
        {},
        2
    );

    const closer = index._createNode(
        "B",
        [0, 0],
        {},
        2
    );

    index.nodes.set("A", entry);
    index.nodes.set("B", closer);

    entry.neighbors[2].add("B");
    closer.neighbors[2].add("A");

    entry.neighbors[1].add("B");
    closer.neighbors[1].add("A");

    index.dimensions = 2;
    index.entryPoint = "A";
    index.maxLevel = 2;

    assert.equal(
        index._navigateToLayerZero([0.5, 0]),
        "B"
    );
});


// ==========================================
// AUTOMATIC INSERTION
// ==========================================

test("Insert first node and set entry point", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        random: () => 0.5
    });

    index.insert("A", [1, 2]);

    assert.equal(index.size, 1);
    assert.equal(index.entryPoint, "A");
    assert.equal(index.maxLevel, 0);
    assert.equal(index.dimensions, 2);
});


test("Insert multiple nodes automatically", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        M: 4,
        efConstruction: 20,
        random: () => 0.5
    });

    index.insert("A", [0, 0]);
    index.insert("B", [1, 0]);
    index.insert("C", [0, 1]);
    index.insert("D", [1, 1]);

    assert.equal(index.size, 4);

    for (const node of index.nodes.values()) {
        assert.ok(node.neighbors[0] instanceof Set);
    }
});


test("Connect inserted nodes bidirectionally", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        random: () => 0.5
    });

    index.insert("A", [0, 0]);
    index.insert("B", [1, 0]);

    const nodeA = index.nodes.get("A");
    const nodeB = index.nodes.get("B");

    assert.ok(nodeA.neighbors[0].has("B"));
    assert.ok(nodeB.neighbors[0].has("A"));
});


test("Higher-level node becomes entry point", () => {

    const values = [0.5, 0.01];

    const index = new HNSWIndex({
        metric: "euclidean",
        M: 4,
        random: () => values.shift() ?? 0.5
    });

    index.insert("A", [0, 0]);
    index.insert("B", [1, 0]);

    assert.equal(index.nodes.get("A").level, 0);
    assert.equal(index.nodes.get("B").level, 3);

    assert.equal(index.entryPoint, "B");
    assert.equal(index.maxLevel, 3);
    assert.equal(
        index.nodes.get("B").neighbors.length,
        4
    );
});


test("Preserve inserted metadata", () => {

    const index = new HNSWIndex({
        metric: "euclidean"
    });

    index.insert(
        "A",
        [1, 0],
        { text: "Binary Search Tree", page: 30 }
    );

    assert.equal(
        index.nodes.get("A").metadata.text,
        "Binary Search Tree"
    );
});


test("Reject duplicate node IDs", () => {

    const index = new HNSWIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    assert.throws(
        () => index.insert("A", [2, 0]),
        /already exists/
    );
});


test("Reject incorrect vector dimensions", () => {

    const index = new HNSWIndex({
        metric: "euclidean"
    });

    index.insert("A", [1, 0]);

    assert.throws(
        () => index.insert("B", [1, 2, 3]),
        /dimensions/
    );
});


test("Reject zero vector for cosine metric", () => {

    const index = new HNSWIndex({
        metric: "cosine"
    });

    assert.throws(
        () => index.insert("A", [0, 0]),
        /Zero vectors/
    );
});


test("Respect maximum neighbor degree", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        M: 2,
        efConstruction: 30,
        random: () => 0.9
    });

    for (let i = 0; i < 40; i++) {

        index.insert(
            `V${i}`,
            [Math.cos(i), Math.sin(i)]
        );
    }

    for (const node of index.nodes.values()) {

        for (
            let layer = 0;
            layer <= node.level;
            layer++
        ) {

            const limit =
                layer === 0 ? 2 * index.M : index.M;

            assert.ok(
                node.neighbors[layer].size <= limit,
                `Node ${node.id} exceeds the connection limit at layer ${layer}`
            );
        }
    }
});


test("Connections do not reference missing nodes", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        random: () => 0.5
    });

    for (let i = 0; i < 20; i++) {
        index.insert(`V${i}`, [i, i * 2]);
    }

    for (const node of index.nodes.values()) {

        for (
            let layer = 0;
            layer <= node.level;
            layer++
        ) {

            for (const id of node.neighbors[layer]) {

                const neighbor = index.nodes.get(id);

                assert.ok(neighbor);

                assert.ok(
                    neighbor.level >= layer
                );
            }
        }
    }
});


// ==========================================
// GRAPH STATISTICS
// ==========================================

test("Return graph statistics", () => {

    const index = new HNSWIndex({
        metric: "euclidean",
        random: () => 0.5
    });

    index.insert("A", [1, 0]);
    index.insert("B", [0, 1]);

    const stats = index.getStats();

    assert.equal(stats.totalNodes, 2);
    assert.equal(stats.dimensions, 2);
    assert.equal(stats.maxLevel, 0);
    assert.equal(stats.layerDistribution[0], 2);
});


// ==========================================
// INVALID INPUTS
// ==========================================

test("Reject invalid vector", () => {

    const index = new HNSWIndex();

    assert.throws(
        () => index.insert("A", [1, NaN]),
        /finite numbers/
    );
});


test("Reject invalid entry point", () => {

    const index = createTestGraph();

    assert.throws(
        () => index._greedySearch(
            [1, 0],
            "UNKNOWN",
            0
        ),
        /does not exist/
    );
});

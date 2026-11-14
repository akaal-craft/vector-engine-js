
import test from "node:test";
import assert from "node:assert/strict";

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { VectorIndexManager } from "../src/core/indexManager.js";

// Test persistence with all three algorithms.
const algorithms = [
    "bruteForce",
    "kdTree",
    "hnsw"
];

for (const algorithm of algorithms) {

    test(`Save and restore vectors using ${algorithm}`, async (t) => {

        // Create temporary storage for this test.
        const directory = await mkdtemp(
            join(tmpdir(), "vector-engine-")
        );

        t.after(async () => {
            await rm(directory, {
                recursive: true,
                force: true
            });
        });

        const filePath = join(directory, "vectors.json");

        const options = {
            metric: "euclidean",
            M: 8,
            efConstruction: 100,
            efSearch: 50
        };

        // Create original index.
        const original = new VectorIndexManager({
            algorithm,
            options
        });

        // Insert vectors.
        original.insert("A", [0, 0], {
            label: "Origin"
        });

        original.insert("B", [5, 5], {
            label: "Far"
        });

        // Update one vector.
        original.update("B", [1, 1], {
            label: "Updated"
        });

        // Delete another vector.
        original.delete("A");

        // Save everything.
        await original.save(filePath);

        // Simulate a new application instance.
        const restored = new VectorIndexManager({
            algorithm,
            options
        });

        // Load previously saved data.
        const count = await restored.load(filePath);

        assert.equal(count, 1);
        assert.equal(restored.size, 1);

        // Verify actual search.
        const results = restored.search([1, 1], 1);

        assert.equal(results[0].id, "B");
        assert.equal(results[0].metadata.label, "Updated");
    });
}

import {
    euclideanDistance,
    manhattanDistance
} from "./distance.js";

// Supported metrics
const DISTANCE_METRICS = {
    euclidean: euclideanDistance,
    manhattan: manhattanDistance
};


// ==========================================
// VALIDATION
// ==========================================

function validateVector(vector, name = "vector") {
    if (!Array.isArray(vector) || vector.length === 0) {
        throw new TypeError(
            `${name} must be a non-empty array.`
        );
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


// ==========================================
// RESULT ORDERING
// ==========================================

function compareResults(a, b) {
    return (
        a.distance - b.distance ||
        a.id.localeCompare(b.id)
    );
}


// ==========================================
// BUILD BALANCED KD-TREE
// ==========================================

function buildTree(records, depth, dimensions) {
    if (records.length === 0) {
        return null;
    }

    // Rotate through dimensions.
    const axis = depth % dimensions;

    // Sort using the current splitting dimension.
    records.sort((a, b) => {
        const difference =
            a.vector[axis] - b.vector[axis];

        return difference || a.id.localeCompare(b.id);
    });

    // Select median as the current node.
    const middle = Math.floor(records.length / 2);

    const node = {
        record: records[middle],
        axis,
        left: null,
        right: null
    };

    // Recursively construct both sides.
    node.left = buildTree(
        records.slice(0, middle),
        depth + 1,
        dimensions
    );

    node.right = buildTree(
        records.slice(middle + 1),
        depth + 1,
        dimensions
    );

    return node;
}


// ==========================================
// KD-TREE INDEX
// ==========================================

export class KDTreeIndex {

    constructor({ metric = "euclidean" } = {}) {

        if (!DISTANCE_METRICS[metric]) {
            throw new Error(
                `KD-Tree does not support metric: ${metric}`
            );
        }

        this.metric = metric;

        this.distanceFunction =
            DISTANCE_METRICS[metric];

        // Original records are maintained in a Map.
        this.records = new Map();

        // Root of the KD-Tree.
        this.root = null;

        this.dimensions = null;

        this.lastSearchStats = {
            visitedNodes: 0,
            totalVectors: 0
        };
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

        if (
            this.dimensions !== null &&
            vector.length !== this.dimensions
        ) {
            throw new Error(
                "Vector dimensions do not match the index."
            );
        }

        if (this.dimensions === null) {
            this.dimensions = vector.length;
        }

        // Insert or update the record.
        this.records.set(id, {
            id,
            vector: [...vector],
            metadata: { ...metadata }
        });

        // Rebuild the balanced tree.
        this.rebuild();

        return id;
    }


    // ======================================
    // REBUILD TREE
    // ======================================

    rebuild() {

        if (this.records.size === 0) {
            this.root = null;
            this.dimensions = null;
            return;
        }

        const records = [
            ...this.records.values()
        ];

        this.root = buildTree(
            records,
            0,
            this.dimensions
        );
    }


    // ======================================
    // NEAREST NEIGHBOR SEARCH
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

        if (this.records.size === 0) {
            this.lastSearchStats = {
                visitedNodes: 0,
                totalVectors: 0
            };

            return [];
        }

        if (queryVector.length !== this.dimensions) {
            throw new Error(
                "Query vector dimensions do not match the index."
            );
        }

        const candidates = [];

        let visitedNodes = 0;

        // ----------------------------------
        // Add candidate in sorted order.
        // ----------------------------------

        function addCandidate(candidate) {

            candidates.push(candidate);

            candidates.sort(compareResults);

            if (candidates.length > k) {
                candidates.pop();
            }
        }


        // ----------------------------------
        // Recursive KD-Tree search.
        // ----------------------------------

        const searchNode = (node) => {

            if (node === null) {
                return;
            }

            visitedNodes++;

            const record = node.record;

            // Check current vector.
            const distance = this.distanceFunction(
                queryVector,
                record.vector
            );

            addCandidate({
                id: record.id,
                distance,
                metadata: { ...record.metadata }
            });


            // Difference along splitting dimension.
            const axisDifference =
                queryVector[node.axis] -
                record.vector[node.axis];


            // Visit the closer side first.
            const nearBranch =
                axisDifference <= 0
                    ? node.left
                    : node.right;

            const farBranch =
                axisDifference <= 0
                    ? node.right
                    : node.left;


            searchNode(nearBranch);


            // ----------------------------------
            // PRUNING DECISION
            // ----------------------------------

            const worstDistance =
                candidates.length < k
                    ? Infinity
                    : candidates[k - 1].distance;

            const minimumPossibleDistance =
                Math.abs(axisDifference);


            // Search the far region only when
            // it could contain a better result.

            if (
                minimumPossibleDistance <= worstDistance
            ) {
                searchNode(farBranch);
            }
        };


        // Start from the root.
        searchNode(this.root);


        // Store search statistics.
        this.lastSearchStats = {
            visitedNodes,
            totalVectors: this.records.size
        };


        return candidates;
    }


    // ======================================
    // DELETE VECTOR
    // ======================================

    delete(id) {

        const deleted = this.records.delete(id);

        if (deleted) {
            this.rebuild();
        }

        return deleted;
    }


    // ======================================
    // CLEAR INDEX
    // ======================================

    clear() {

        this.records.clear();

        this.root = null;

        this.dimensions = null;

        this.lastSearchStats = {
            visitedNodes: 0,
            totalVectors: 0
        };
    }


    // ======================================
    // VECTOR COUNT
    // ======================================

    get size() {
        return this.records.size;
    }
}
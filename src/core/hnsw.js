
import {
    euclideanDistance,
    manhattanDistance,
    cosineDistance
} from "./distance.js";

import { PriorityQueue } from "./priorityQueue.js";


// ==========================================
// DISTANCE METRICS
// ==========================================

const DISTANCE_METRICS = {
    euclidean: euclideanDistance,
    manhattan: manhattanDistance,
    cosine: cosineDistance
};


function validateVector(vector) {
    if (!Array.isArray(vector) || vector.length === 0) {
        throw new TypeError(
            "Vector must be a non-empty array."
        );
    }

    for (const value of vector) {
        if (
            typeof value !== "number" ||
            !Number.isFinite(value)
        ) {
            throw new TypeError(
                "Vector must contain finite numbers."
            );
        }
    }
}


function compareDistance(a, b) {
    return (
        a.distance - b.distance ||
        a.id.localeCompare(b.id)
    );
}


function compareWorstFirst(a, b) {
    return compareDistance(b, a);
}


// ==========================================
// HNSW INDEX
// ==========================================

export class HNSWIndex {

    constructor({
        metric = "cosine",
        M = 8,
        efConstruction = 100,
        efSearch = 50,
        random = Math.random
    } = {}) {

        if (!DISTANCE_METRICS[metric]) {
            throw new Error(
                `Unsupported metric: ${metric}`
            );
        }

        if (!Number.isInteger(M) || M < 2) {
            throw new Error(
                "M must be an integer greater than or equal to 2."
            );
        }

        if (
            !Number.isInteger(efConstruction) ||
            efConstruction < 1
        ) {
            throw new Error(
                "efConstruction must be a positive integer."
            );
        }

        if (
            !Number.isInteger(efSearch) ||
            efSearch < 1
        ) {
            throw new Error(
                "efSearch must be a positive integer."
            );
        }

        if (typeof random !== "function") {
            throw new TypeError(
                "Random generator must be a function."
            );
        }

        this.metric = metric;
        this.distanceFunction = DISTANCE_METRICS[metric];

        this.M = M;
        this.efConstruction = efConstruction;
        this.efSearch = efSearch;

        this.random = random;

        this.nodes = new Map();

        this.entryPoint = null;
        this.maxLevel = -1;
        this.dimensions = null;
    }


    // ======================================
    // RANDOM LAYER ASSIGNMENT
    // ======================================

    _randomLevel() {

        const randomValue = this.random();

        if (
            typeof randomValue !== "number" ||
            !Number.isFinite(randomValue) ||
            randomValue < 0 ||
            randomValue > 1
        ) {
            throw new Error(
                "Random generator must return a number between 0 and 1."
            );
        }

        const safeRandom = Math.max(
            Number.MIN_VALUE,
            randomValue
        );

        return Math.floor(
            -Math.log(safeRandom) / Math.log(this.M)
        );
    }


    // ======================================
    // CREATE GRAPH NODE
    // ======================================

    _createNode(
        id,
        vector,
        metadata = {},
        level = this._randomLevel()
    ) {

        if (
            typeof id !== "string" ||
            id.trim().length === 0
        ) {
            throw new TypeError(
                "Node ID must be a non-empty string."
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
            !Number.isInteger(level) ||
            level < 0
        ) {
            throw new Error(
                "Node level must be non-negative."
            );
        }

        return {
            id,
            vector: [...vector],
            metadata: { ...metadata },
            level,

            // One independent neighbor set per layer.
            neighbors: Array.from(
                { length: level + 1 },
                () => new Set()
            )
        };
    }


    // ======================================
    // QUERY VALIDATION
    // ======================================

    _validateQuery(queryVector, layer) {

        validateVector(queryVector);

        if (
            !Number.isInteger(layer) ||
            layer < 0
        ) {
            throw new Error(
                "Layer must be a non-negative integer."
            );
        }

        if (
            this.dimensions !== null &&
            queryVector.length !== this.dimensions
        ) {
            throw new Error(
                "Query dimensions do not match index."
            );
        }

        if (
            this.metric === "cosine" &&
            queryVector.every(value => value === 0)
        ) {
            throw new Error(
                "Zero query vector is invalid for cosine distance."
            );
        }
    }


    // ======================================
    // GREEDY SEARCH
    // ======================================

    _greedySearch(
        queryVector,
        entryPointId,
        layer
    ) {

        this._validateQuery(queryVector, layer);

        let current = this.nodes.get(entryPointId);

        if (!current) {
            throw new Error(
                "Entry point does not exist."
            );
        }

        if (current.level < layer) {
            throw new Error(
                "Entry point does not exist in this layer."
            );
        }

        let currentDistance = this.distanceFunction(
            queryVector,
            current.vector
        );

        while (true) {

            let improved = false;

            for (const neighborId of current.neighbors[layer]) {

                const neighbor = this.nodes.get(neighborId);

                if (!neighbor || neighbor.level < layer) {
                    continue;
                }

                const distance = this.distanceFunction(
                    queryVector,
                    neighbor.vector
                );

                const candidate = {
                    id: neighbor.id,
                    distance
                };

                const best = {
                    id: current.id,
                    distance: currentDistance
                };

                if (compareDistance(candidate, best) < 0) {
                    current = neighbor;
                    currentDistance = distance;
                    improved = true;
                }
            }

            if (!improved) {
                break;
            }
        }

        return {
            id: current.id,
            distance: currentDistance,
            node: current
        };
    }


    // ======================================
    // BEST-FIRST SEARCH IN ONE LAYER
    // ======================================

    _searchLayer(
        queryVector,
        entryPoints,
        ef,
        layer
    ) {

        this._validateQuery(queryVector, layer);

        if (!Number.isInteger(ef) || ef < 1) {
            throw new Error(
                "ef must be a positive integer."
            );
        }

        if (!Array.isArray(entryPoints)) {
            throw new TypeError(
                "Entry points must be an array."
            );
        }

        // Closest unexplored candidate first.
        const candidates = new PriorityQueue(
            compareDistance
        );

        // Worst accepted result first.
        const best = new PriorityQueue(
            compareWorstFirst
        );

        const visited = new Set();

        // Initialize entry points.
        for (const entryPointId of entryPoints) {

            if (visited.has(entryPointId)) {
                continue;
            }

            const node = this.nodes.get(entryPointId);

            if (!node) {
                throw new Error(
                    `Entry point ${entryPointId} does not exist.`
                );
            }

            if (node.level < layer) {
                throw new Error(
                    "Entry point does not exist in requested layer."
                );
            }

            visited.add(entryPointId);

            const candidate = {
                id: node.id,
                distance: this.distanceFunction(
                    queryVector,
                    node.vector
                )
            };

            candidates.push(candidate);
            best.push(candidate);

            if (best.size > ef) {
                best.pop();
            }
        }

        // Explore candidates.
        while (!candidates.isEmpty) {

            const current = candidates.pop();
            const worst = best.peek();

            if (
                best.size >= ef &&
                compareDistance(current, worst) > 0
            ) {
                break;
            }

            const currentNode = this.nodes.get(
                current.id
            );

            for (
                const neighborId of currentNode.neighbors[layer]
            ) {

                if (visited.has(neighborId)) {
                    continue;
                }

                visited.add(neighborId);

                const neighbor = this.nodes.get(neighborId);

                if (!neighbor || neighbor.level < layer) {
                    continue;
                }

                const candidate = {
                    id: neighbor.id,
                    distance: this.distanceFunction(
                        queryVector,
                        neighbor.vector
                    )
                };

                const currentWorst = best.peek();

                if (
                    best.size < ef ||
                    compareDistance(
                        candidate,
                        currentWorst
                    ) < 0
                ) {

                    candidates.push(candidate);
                    best.push(candidate);

                    if (best.size > ef) {
                        best.pop();
                    }
                }
            }
        }

       return best.toSortedArray().sort(compareDistance);
    }


    // ======================================
    // SELECT NEAREST NEIGHBORS
    // ======================================

  
_selectNeighbors(candidates, maxNeighbors) {
    return candidates
        .slice()
        .sort(compareDistance)
        .slice(0, maxNeighbors);
}

    // ======================================
    // CONNECT TWO NODES
    // ======================================

    _connectNodes(idA, idB, layer) {

        if (idA === idB) {
            return;
        }

        const nodeA = this.nodes.get(idA);
        const nodeB = this.nodes.get(idB);

        if (!nodeA || !nodeB) {
            throw new Error(
                "Cannot connect nodes that do not exist."
            );
        }

        if (
            nodeA.level < layer ||
            nodeB.level < layer
        ) {
            throw new Error(
                "Both nodes must exist in the connection layer."
            );
        }

        // Establish bidirectional connections.
        nodeA.neighbors[layer].add(idB);
        nodeB.neighbors[layer].add(idA);
    }


    // ======================================
    // PRUNE EXCESS CONNECTIONS
    // ======================================

   _pruneConnections(node, layer) {

    const maxNeighbors =
        layer === 0 ? 2 * this.M : this.M;

    const neighborIds = [
        ...node.neighbors[layer]
    ];

    if (neighborIds.length <= maxNeighbors) {
        return;
    }

    // Calculate distance from this node
    // to every connected neighbor.

    const rankedNeighbors = neighborIds
        .map(id => {

            const neighbor = this.nodes.get(id);

            if (!neighbor) {
                return null;
            }

            return {
                id,
                distance: this.distanceFunction(
                    node.vector,
                    neighbor.vector
                )
            };
        })
        .filter(Boolean)
        .sort(compareDistance);

    // Keep only the closest permitted neighbors.

    const keep = new Set(
        rankedNeighbors
            .slice(0, maxNeighbors)
            .map(item => item.id)
    );

    // Remove excess connections.
    for (const neighborId of neighborIds) {

        if (keep.has(neighborId)) {
            continue;
        }

        node.neighbors[layer].delete(neighborId);

        // Remove the reverse connection too.
        const neighbor = this.nodes.get(neighborId);

        if (neighbor) {
            neighbor.neighbors[layer].delete(node.id);
        }
    }
}


    // ======================================
    // INSERT VECTOR — CORE HNSW OPERATION
    // ======================================

    insert(id, vector, metadata = {}) {

        if (
            typeof id !== "string" ||
            id.trim().length === 0
        ) {
            throw new TypeError(
                "Node ID must be a non-empty string."
            );
        }

        if (this.nodes.has(id)) {
            throw new Error(
                `Node ID ${id} already exists.`
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
                "Vector dimensions do not match index."
            );
        }

        if (
            this.metric === "cosine" &&
            vector.every(value => value === 0)
        ) {
            throw new Error(
                "Zero vectors cannot be inserted with cosine distance."
            );
        }

        // Assign maximum layer.
        const level = this._randomLevel();

        const newNode = this._createNode(
            id,
            vector,
            metadata,
            level
        );

        // ----------------------------------
        // CASE 1: FIRST NODE
        // ----------------------------------

        if (this.nodes.size === 0) {

            this.dimensions = vector.length;

            this.nodes.set(id, newNode);

            this.entryPoint = id;

            this.maxLevel = level;

            return id;
        }


        // Store new node before constructing links.
        this.nodes.set(id, newNode);

        const previousMaxLevel = this.maxLevel;

        let currentEntryPoint = this.entryPoint;


        // ----------------------------------
        // CASE 2: DESCEND UPPER LAYERS
        // ----------------------------------

        // Use greedy search from the existing
        // top layer down to the new node's level.

        for (
            let layer = previousMaxLevel;
            layer > level;
            layer--
        ) {

            const result = this._greedySearch(
                vector,
                currentEntryPoint,
                layer
            );

            currentEntryPoint = result.id;
        }


        // ----------------------------------
        // CASE 3: CONNECT APPLICABLE LAYERS
        // ----------------------------------

        const lowestConnectableLevel = Math.min(
            level,
            previousMaxLevel
        );

        for (
            let layer = lowestConnectableLevel;
            layer >= 0;
            layer--
        ) {

            // Find promising existing neighbors.
            const candidates = this._searchLayer(
                vector,
                [currentEntryPoint],
                this.efConstruction,
                layer
            );

            // Layer zero permits 2M neighbors.
            // Upper layers permit M neighbors.
            const maxNeighbors =
                layer === 0 ? 2 * this.M : this.M;

            // Select the nearest candidates.
            const selectedNeighbors =
                this._selectNeighbors(
                    candidates,
                    maxNeighbors
                );

            // Establish graph connections.
            for (const candidate of selectedNeighbors) {

                this._connectNodes(
                    id,
                    candidate.id,
                    layer
                );
            }

            // Existing neighbors may now have
            // more connections than permitted.
            for (const candidate of selectedNeighbors) {

                const neighbor = this.nodes.get(
                    candidate.id
                );

                this._pruneConnections(
                    neighbor,
                    layer
                );
            }

            // Keep the new node within its degree limit.
            this._pruneConnections(newNode, layer);

            // Best candidate becomes the entry
            // for the next lower layer.
            if (candidates.length > 0) {
                currentEntryPoint = candidates[0].id;
            }
        }


        // ----------------------------------
        // CASE 4: UPDATE ENTRY POINT
        // ----------------------------------

        // A node with a higher maximum layer
        // becomes the new graph entry point.

        if (level > previousMaxLevel) {

            this.entryPoint = id;

            this.maxLevel = level;
        }

        return id;
    }


    // ======================================
    // NAVIGATE TO LAYER ZERO
    // ======================================

    _navigateToLayerZero(queryVector) {

        if (this.entryPoint === null) {
            return null;
        }

        let current = this.nodes.get(
            this.entryPoint
        );

        for (
            let layer = this.maxLevel;
            layer >= 1;
            layer--
        ) {

            const result = this._greedySearch(
                queryVector,
                current.id,
                layer
            );

            current = result.node;
        }

        return current.id;
    }


    // ======================================
    // PUBLIC NEAREST-NEIGHBOR SEARCH
    // ======================================

    search(
        queryVector,
        k = 5,
        { efSearch = this.efSearch } = {}
    ) {

        validateVector(queryVector);

        if (!Number.isInteger(k) || k <= 0) {
            throw new Error(
                "K must be a positive integer."
            );
        }

        if (
            !Number.isInteger(efSearch) ||
            efSearch <= 0
        ) {
            throw new Error(
                "efSearch must be a positive integer."
            );
        }

        if (this.nodes.size === 0) {
            return [];
        }

        // Validate dimensions and cosine query.
        this._validateQuery(queryVector, 0);

        // We need at least K candidates.
        const searchWidth = Math.max(k, efSearch);

        // Step 1: Navigate from the top layer
        // towards the nearest region at layer zero.
        const entryPointId =
            this._navigateToLayerZero(queryVector);

        if (entryPointId === null) {
            return [];
        }

        // Step 2: Perform broader search at layer zero.
        const candidates = this._searchLayer(
            queryVector,
            [entryPointId],
            searchWidth,
            0
        );

        // Step 3: Return nearest K results.
        return candidates
            .slice(0, k)
            .map(candidate => {

                const node = this.nodes.get(
                    candidate.id
                );

                return {
                    id: candidate.id,
                    distance: candidate.distance,
                    metadata: { ...node.metadata }
                };
            });
    }

    // ======================================
    // GRAPH STATISTICS
    // ======================================

    get size() {
        return this.nodes.size;
    }


    getStats() {

        const layerDistribution = {};

        let totalEdges = 0;

        for (const node of this.nodes.values()) {

            for (
                let layer = 0;
                layer <= node.level;
                layer++
            ) {

                layerDistribution[layer] =
                    (layerDistribution[layer] ?? 0) + 1;

                totalEdges += node.neighbors[layer].size;
            }
        }

        return {
            totalNodes: this.nodes.size,
            dimensions: this.dimensions,
            maxLevel: this.maxLevel,
            entryPoint: this.entryPoint,
            totalDirectedEdges: totalEdges,
            layerDistribution,
            metric: this.metric,
            M: this.M,
            efConstruction: this.efConstruction,
            efSearch: this.efSearch
        };
    }
}

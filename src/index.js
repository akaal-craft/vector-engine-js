
import { HNSWIndex } from "./core/hnsw.js";
import { BruteForceIndex } from "./core/bruteForce.js";

const vectors = [
    { id: "A", vector: [0, 0] },
    { id: "B", vector: [5, 0] },
    { id: "C", vector: [2, 0] },
    { id: "D", vector: [1, 0] },
    { id: "E", vector: [3, 1] }
];

const config = {
    metric: "euclidean",
    M: 8,
    efConstruction: 30,
    efSearch: 20,
    random: () => 0.5
};

const hnsw = new HNSWIndex(config);
const brute = new BruteForceIndex({
    metric: "euclidean"
});

for (const item of vectors) {
    hnsw.insert(item.id, item.vector, {
        label: item.id
    });

    brute.insert(item.id, item.vector, {
        label: item.id
    });
}

const query = [1.1, 0];
const k = 3;

console.log("\n===== HNSW SEARCH =====");

const hnswResults = hnsw.search(query, k);

hnswResults.forEach((result, index) => {
    console.log(
        `${index + 1}. ${result.id}: ${result.distance.toFixed(4)}`
    );
});

console.log("\n===== BRUTE FORCE SEARCH =====");

const bruteResults = brute.search(query, k);

bruteResults.forEach((result, index) => {
    console.log(
        `${index + 1}. ${result.id}: ${result.distance.toFixed(4)}`
    );
});

console.log("\n===== GRAPH STATISTICS =====");

console.log(
    JSON.stringify(hnsw.getStats(), null, 2)
);

console.log("\n===== SEARCH COMPARISON =====");

console.log(
    "HNSW results:",
    hnswResults.map(result => result.id)
);

console.log(
    "Brute Force results:",
    bruteResults.map(result => result.id)
);

console.log("\n================================");

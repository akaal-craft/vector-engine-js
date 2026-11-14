
import { BruteForceIndex } from "./bruteForce.js";
import { KDTreeIndex } from "./kdTree.js";
import { HNSWIndex } from "./hnsw.js";

import {
    mkdir,
    readFile,
    writeFile
} from "node:fs/promises";

import { dirname, resolve } from "node:path";

// Available algorithms
const INDEX_TYPES = {
    bruteForce: BruteForceIndex,
    kdTree: KDTreeIndex,
    hnsw: HNSWIndex
};

export class VectorIndexManager {

    #index;
    #records = new Map();
    #IndexClass;
    #options;

    constructor({
        algorithm = "bruteForce",
        options = {}
    } = {}) {

        if (!Object.hasOwn(INDEX_TYPES, algorithm)) {
            throw new Error(
                `Unsupported index algorithm: ${algorithm}`
            );
        }

        if (
            options === null ||
            typeof options !== "object" ||
            Array.isArray(options)
        ) {
            throw new TypeError(
                "Index options must be an object."
            );
        }

        this.algorithm = algorithm;
        this.#IndexClass = INDEX_TYPES[algorithm];
        this.#options = { ...options };

        this.#index = new this.#IndexClass(this.#options);
    }

    // INSERT
    insert(id, vector, metadata = {}) {

        if (this.#records.has(id)) {
            throw new Error(`Vector ID ${id} already exists.`);
        }

        const result = this.#index.insert(id, vector, metadata);

        this.#records.set(id, {
            vector: [...vector],
            metadata: { ...metadata }
        });

        return result;
    }

    // SEARCH
    search(queryVector, k = 5, options) {

        if (options === undefined) {
            return this.#index.search(queryVector, k);
        }

        return this.#index.search(queryVector, k, options);
    }

    // UPDATE
    update(id, vector, metadata) {

        const existing = this.#records.get(id);

        if (!existing) {
            throw new Error(`Vector ID ${id} does not exist.`);
        }

        const updatedRecords = new Map(this.#records);

        updatedRecords.set(id, {
            vector: [...vector],
            metadata: metadata === undefined
                ? existing.metadata
                : { ...metadata }
        });

        this.#rebuildIndex(updatedRecords);

        return id;
    }

    // DELETE
    delete(id) {

        if (!this.#records.has(id)) {
            return false;
        }

        const updatedRecords = new Map(this.#records);

        updatedRecords.delete(id);

        this.#rebuildIndex(updatedRecords);

        return true;
    }

    // DELETE AN ENTIRE DOCUMENT
    deleteDocument(documentId) {

        const updatedRecords = new Map(this.#records);

        let deletedChunks = 0;

        for (const [id, record] of updatedRecords) {

            if (record.metadata?.documentId === documentId) {
                updatedRecords.delete(id);
                deletedChunks++;
            }
        }

        if (deletedChunks === 0) {
            return 0;
        }

        // Rebuild index without the deleted document.
        this.#rebuildIndex(updatedRecords);

        return deletedChunks;
    }


    // REBUILD INDEX
    #rebuildIndex(records) {

        const newIndex = new this.#IndexClass(this.#options);

        for (const [id, record] of records) {
            newIndex.insert(
                id,
                record.vector,
                record.metadata
            );
        }

        this.#index = newIndex;

        this.#records = new Map(
            [...records].map(([id, record]) => [
                id,
                {
                    vector: [...record.vector],
                    metadata: { ...record.metadata }
                }
            ])
        );
    }

    // CHECK DOCUMENT ID
    hasDocument(documentId) {

        for (const record of this.#records.values()) {
            if (record.metadata?.documentId === documentId) {
                return true;
            }
        }

        return false;
    }

    // CHECK PDF FILE PATH
    hasSource(filePath) {

        const target = resolve(filePath).toLowerCase();

        for (const record of this.#records.values()) {

            const source = record.metadata?.source;

            if (
                typeof source === "string" &&
                resolve(source).toLowerCase() === target
            ) {
                return true;
            }
        }

        return false;
    }

    // GET DOCUMENT LIST
    getDocuments() {

        const documents = new Map();

        for (const record of this.#records.values()) {

            const metadata = record.metadata ?? {};
            const documentId = metadata.documentId;

            if (!documentId) {
                continue;
            }

            if (!documents.has(documentId)) {

                documents.set(documentId, {
                    documentId,
                    source: metadata.source ?? "Unknown",
                    totalChunks: metadata.totalChunks ?? 0,
                    chunks: 0
                });
            }

            documents.get(documentId).chunks++;
        }

        return [...documents.values()];
    }

    // SAVE
    async save(filePath = "./data/vectors.json") {

        const data = {
            version: 1,
            algorithm: this.algorithm,
            options: this.#options,

            records: [...this.#records.entries()].map(
                ([id, record]) => ({
                    id,
                    vector: record.vector,
                    metadata: record.metadata
                })
            )
        };

        await mkdir(dirname(filePath), {
            recursive: true
        });

        await writeFile(
            filePath,
            JSON.stringify(data, null, 2),
            "utf8"
        );

        return filePath;
    }

    // LOAD
    async load(filePath = "./data/vectors.json") {

        const content = await readFile(filePath, "utf8");

        const data = JSON.parse(content);

        if (
            !data ||
            data.version !== 1 ||
            data.algorithm !== this.algorithm ||
            !Array.isArray(data.records)
        ) {
            throw new Error("Invalid or incompatible vector data file.");
        }

        const restoredOptions = data.options ?? this.#options;

        if (
            restoredOptions === null ||
            typeof restoredOptions !== "object" ||
            Array.isArray(restoredOptions)
        ) {
            throw new Error("Invalid saved index options.");
        }

        const newIndex = new this.#IndexClass(restoredOptions);
        const newRecords = new Map();

        for (const record of data.records) {

            if (
                !record ||
                typeof record.id !== "string" ||
                !Array.isArray(record.vector) ||
                newRecords.has(record.id)
            ) {
                throw new Error("Invalid vector record in saved file.");
            }

            const metadata = record.metadata ?? {};

            newIndex.insert(record.id, record.vector, metadata);

            newRecords.set(record.id, {
                vector: [...record.vector],
                metadata: { ...metadata }
            });
        }

        this.#options = { ...restoredOptions };
        this.#index = newIndex;
        this.#records = newRecords;

        return this.size;
    }

    // SIZE
    get size() {
        return this.#records.size;
    }

    // STATISTICS
    getStats() {

        return {
            algorithm: this.algorithm,
            size: this.size,
            details:
                typeof this.#index.getStats === "function"
                    ? this.#index.getStats()
                    : null
        };
    }
}

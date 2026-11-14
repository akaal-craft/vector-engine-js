import test from "node:test";
import assert from "node:assert/strict";

import { PriorityQueue } from "../src/core/priorityQueue.js";


test("Min-heap returns smallest value first", () => {

    const queue = new PriorityQueue(
        (a, b) => a - b
    );

    [8, 3, 6, 1, 5].forEach(
        value => queue.push(value)
    );

    assert.equal(queue.pop(), 1);
    assert.equal(queue.pop(), 3);
    assert.equal(queue.pop(), 5);
});


test("Max-heap returns largest value first", () => {

    const queue = new PriorityQueue(
        (a, b) => b - a
    );

    [8, 3, 6, 1, 5].forEach(
        value => queue.push(value)
    );

    assert.equal(queue.pop(), 8);
    assert.equal(queue.pop(), 6);
    assert.equal(queue.pop(), 5);
});


test("Peek does not remove element", () => {

    const queue = new PriorityQueue(
        (a, b) => a - b
    );

    queue.push(5);
    queue.push(2);

    assert.equal(queue.peek(), 2);
    assert.equal(queue.size, 2);
});


test("Empty queue returns null", () => {

    const queue = new PriorityQueue(
        (a, b) => a - b
    );

    assert.equal(queue.pop(), null);
    assert.equal(queue.peek(), null);
});


test("Return sorted copy", () => {

    const queue = new PriorityQueue(
        (a, b) => a - b
    );

    [5, 1, 4].forEach(
        value => queue.push(value)
    );

    assert.deepEqual(
        queue.toSortedArray(),
        [1, 4, 5]
    );

    assert.equal(queue.size, 3);
});
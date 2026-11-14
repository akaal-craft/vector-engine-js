// ==========================================
// BINARY HEAP PRIORITY QUEUE
// ==========================================

export class PriorityQueue {

    constructor(compare) {

        if (typeof compare !== "function") {
            throw new TypeError(
                "A comparison function is required."
            );
        }

        this.heap = [];
        this.compare = compare;
    }


    // ======================================
    // SIZE
    // ======================================

    get size() {
        return this.heap.length;
    }


    get isEmpty() {
        return this.heap.length === 0;
    }


    // ======================================
    // PEEK
    // ======================================

    peek() {
        return this.heap[0] ?? null;
    }


    // ======================================
    // INSERT
    // ======================================

    push(value) {

        this.heap.push(value);

        this._bubbleUp(this.heap.length - 1);
    }


    // ======================================
    // REMOVE HIGHEST PRIORITY ELEMENT
    // ======================================

    pop() {

        if (this.isEmpty) {
            return null;
        }

        const top = this.heap[0];

        const last = this.heap.pop();

        if (!this.isEmpty) {
            this.heap[0] = last;
            this._bubbleDown(0);
        }

        return top;
    }


    // ======================================
    // BUBBLE UP
    // ======================================

    _bubbleUp(index) {

        while (index > 0) {

            const parent = Math.floor(
                (index - 1) / 2
            );

            if (
                this.compare(
                    this.heap[index],
                    this.heap[parent]
                ) >= 0
            ) {
                break;
            }

            [
                this.heap[index],
                this.heap[parent]
            ] = [
                this.heap[parent],
                this.heap[index]
            ];

            index = parent;
        }
    }


    // ======================================
    // BUBBLE DOWN
    // ======================================

    _bubbleDown(index) {

        const length = this.heap.length;

        while (true) {

            const left = 2 * index + 1;
            const right = 2 * index + 2;

            let best = index;


            if (
                left < length &&
                this.compare(
                    this.heap[left],
                    this.heap[best]
                ) < 0
            ) {
                best = left;
            }


            if (
                right < length &&
                this.compare(
                    this.heap[right],
                    this.heap[best]
                ) < 0
            ) {
                best = right;
            }


            if (best === index) {
                break;
            }


            [
                this.heap[index],
                this.heap[best]
            ] = [
                this.heap[best],
                this.heap[index]
            ];

            index = best;
        }
    }


    // ======================================
    // RETURN SORTED COPY
    // ======================================

    toSortedArray() {
        return [...this.heap].sort(this.compare);
    }
}
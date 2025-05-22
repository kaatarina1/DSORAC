class QElement {
    constructor(index, dx, dy, distance) {
        this.index = index;
        this.dx = dx;
        this.dy = dy;
        this.distance = distance;
    }
}

export class PriorityQueue {
    constructor() {
        this.items = [];
    }

    enqueue(index, dx, dy, distance) {
        let element = new QElement(index, dx, dy, distance);
        let added = false;

        for (let i = 0; i < this.items.length; i++) {
            if (element.distance < this.items[i].distance) {
                this.items.splice(i, 0, element);
                added = true;
                break;
            }
        }

        if (!added) {
            this.items.push(element);
        }
    }

    dequeue() {
        if (this.isEmpty()) {
            return null;
        }
        return this.items.shift();
    }

    front() {
        if (this.isEmpty()) {
            return null;
        }
        return this.items[0];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }
}
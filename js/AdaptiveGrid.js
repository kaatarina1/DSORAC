// Razdelitev oblaka točk na mrežo celic, 
// ki se prilagajajo gostoti točk. 
// Celice z več točkami se delijo, 
// dokler ne dosežejo največjega števila točk na celico.
// Delimo le po višini in širini (x in z), ne pa po višini (y).
// Takšno delitev uporabljamo zaradi sortiranja točk,
// potrebno za renderiranje točk z gaussovkami.
export class AdaptiveGrid {
    constructor(positions, maxPointsPerCell) {
        this.positions = positions;
        this.maxPointsPerCell = maxPointsPerCell;
        this.cells = [];

        const allIndices = new Int32Array(positions.length / 3);
        for (let i = 0; i < allIndices.length; i++) allIndices[i] = i;

        this.split({
            indices: allIndices,
            minX: -Infinity, maxX: Infinity,
            minZ: -Infinity, maxZ: Infinity,
        });
    }

    split(cell) {
        if (cell.indices.length <= this.maxPointsPerCell) {
            this.cells.push(cell);
            return;
        }

        let minX = Infinity, maxX = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (const i of cell.indices) {
            const x = this.positions[i * 3];
            const z = this.positions[i * 3 + 2];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }

        const splitOnX = (maxX - minX) >= (maxZ - minZ);
        const mid = splitOnX
            ? (minX + maxX) / 2
            : (minZ + maxZ) / 2;

        const left = [];
        const right = [];

        for (const i of cell.indices) {
            const val = splitOnX
                ? this.positions[i * 3]
                : this.positions[i * 3 + 2]; 
            (val < mid ? left : right).push(i);
        }

        if (left.length === 0 || right.length === 0) {
            const half = Math.floor(cell.indices.length / 2);
            this.split({ indices: cell.indices.slice(0, half) });
            this.split({ indices: cell.indices.slice(half) });
            return;
        }

        this.split({ indices: left });
        this.split({ indices: right });
    }
}
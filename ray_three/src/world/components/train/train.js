import { Group } from 'three';

import { createGeometries } from "./geometries.js";
import { createMaterials } from "./materials.js";
import { createMeshes } from "./meshes.js";

class Train extends Group {
    constructor() {
        super();

        this.meshes = createMeshes();

        this.add(
            this.meshes.nose,
            this.meshes.cabin,
            this.meshes.chimney,
            this.meshes.smallWheelRear,
            this.meshes.smallWheelCenter,
            this.meshes.smallWheelFront,
            this.meshes.bigWheel
        );

        this.update = (delta) => {};
    }
}

export { Train }
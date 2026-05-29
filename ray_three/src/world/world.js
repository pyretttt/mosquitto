import { createCamera } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createScene } from './components/scene.js';
import { createLights } from './components/lights.js';

import { createRenderer } from './systems/renderer.js';
import { Resizer } from './systems/resizer.js';

import { Vector3 } from 'three';


class World {
    constructor(container) {
        this.camera = createCamera();
        this.scene = createScene();
        this.renderer = createRenderer();
        container.append(this.renderer.domElement);

        const cube = createCube();
        cube.rotation.x = Math.PI / 4;
        cube.scale.set(0.5, 0.5, 0.5);
        const subCube = createCube();
        subCube.material.color.set('green');
        subCube.position.x = -3;
        subCube.rotation.x = Math.PI / 4;
        subCube.scale.set(2, 2, 2);
        cube.add(subCube);

        const light = createLights();

        this.scene.add(cube, light);

        this.resizer = new Resizer(container, this.camera, this.renderer);
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }
}

export { World };

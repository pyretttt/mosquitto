import { createCamera } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createScene } from './components/scene.js';

import { createRenderer } from './systems/renderer.js';
import { Resizer } from './systems/resizer.js';


class World {
    constructor(container) {
        this.camera = createCamera();
        this.scene = createScene();
        this.renderer = createRenderer();
        container.append(this.renderer.domElement);

        const cube = createCube();
        this.scene.add(cube);

        this.resizer = new Resizer(container, this.camera, this.renderer);
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }
}

export { World };

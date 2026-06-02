import { createCamera } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createScene } from './components/scene.js';
import { createLights } from './components/lights.js';
import { createControls } from './systems/control.js';

import { createRenderer } from './systems/renderer.js';
import { Resizer } from './systems/resizer.js';
import { Loop } from './systems/loop.js';

import { Vector3 } from 'three';


class World {
    constructor(container) {
        this.camera = createCamera();
        this.scene = createScene();
        this.renderer = createRenderer();
        this.loop = new Loop(this.camera, this.scene, this.renderer);
        container.append(this.renderer.domElement);

        this.controls = createControls(this.camera, this.renderer.domElement);

        const cube = createCube();
        cube.update = (delta) => {
            // cube.rotation.z +=  0.5238 * delta;
            // cube.rotation.x += 0.5238 * delta;
            // cube.rotation.y += 0.5238 * delta;
        };
        this.camera.update = (delta) => {
            this.camera.updateProjectionMatrix();
        };
        this.loop.updatables.push(cube, this.camera, this.controls);
        const { directionalLight, ambientLight } = createLights();

        this.scene.add(cube, directionalLight, ambientLight);

        this.controls.target.copy(cube.position);


        this.resizer = new Resizer(container, this.camera, this.renderer);
        this.resizer.onResize = () => {
            this.render();
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    start() {
        this.loop.start();
    }

    stop() {
        this.loop.end();
    }
}

export { World };

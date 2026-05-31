import { createCamera } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createScene } from './components/scene.js';
import { createLights } from './components/lights.js';

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

        const cube = createCube();
        cube.update = (delta) => {
            console.log(delta);
            cube.rotation.z +=  0.5238 * delta;
            cube.rotation.x += 0.5238 * delta;
            cube.rotation.y += 0.5238 * delta;
        };
        this.camera.update = (delta) => {
            this.camera.fov = (Math.sin(performance.now() / 1000) * 0.5 + 0.5) * Math.PI/4 + Math.PI / 4;
            this.camera.fov = this.camera.fov * 180 / Math.PI;
            this.camera.updateProjectionMatrix();
        };
        this.loop.updatables.push(cube, this.camera);
        // cube.rotation.x = Math.PI / 4;
        // cube.scale.set(0.5, 0.5, 0.5);
        // const subCube = createCube();
        // subCube.material.color.set('green');
        // subCube.position.x = -3;
        // subCube.rotation.x = Math.PI / 4;
        // subCube.scale.set(2, 2, 2);
        // cube.add(subCube);

        const light = createLights();

        this.scene.add(cube, light);

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

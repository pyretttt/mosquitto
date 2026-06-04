import { createCamera } from './components/camera.js';
import { createScene } from './components/scene.js';
import { createLights } from './components/lights.js';
import { createControls } from './systems/control.js';
import { createMeshGroup } from './components/group.js';
import { loadGirl } from './girl.js';

import { createRenderer } from './systems/renderer.js';
import { Resizer } from './systems/resizer.js';
import { Loop } from './systems/loop.js';

import { Box3, Vector3 } from 'three';


class World {
    constructor(container) {
        this.camera = createCamera();
        this.scene = createScene();
        this.renderer = createRenderer();
        this.loop = new Loop(this.camera, this.scene, this.renderer);
        container.append(this.renderer.domElement);

        this.controls = createControls(this.camera, this.renderer.domElement);
        this.camera.update = (delta) => {
            this.camera.updateProjectionMatrix();
        };
        const { directionalLight, ambientLight } = createLights();
        directionalLight.position.set(5, 10, 7);
        const meshGroup = createMeshGroup();

        this.loop.updatables.push(this.camera, this.controls);
        this.scene.add(directionalLight, ambientLight);

        this.resizer = new Resizer(container, this.camera, this.renderer);
        this.resizer.onResize = () => {
            this.render();
        }
    }

    async init() {
        const girlScene = await loadGirl();
        this.scene.add(girlScene);
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

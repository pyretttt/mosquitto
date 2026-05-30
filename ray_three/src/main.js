import {
    BoxGeometry,
    Color,
    Mesh,
    MeshBasicMaterial,
    PerspectiveCamera,
    Scene,
    WebGLRenderer,
} from 'three';

import { World } from './world/world.js';

function main() {
    // Get a reference to the container element
    const container = document.querySelector('#scene-container');

    // 1. Create an instance of the World app
    const world = new World(container);

    world.scene.background = new Color('red');

    // 2. Render the scene
    world.start();
}

main();
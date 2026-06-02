import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

function createControls(camera, canvas) {
    const orbitControls = new OrbitControls(camera, canvas);
    orbitControls.tick = () => orbitControls.update();

    return orbitControls;
}

export { createControls };
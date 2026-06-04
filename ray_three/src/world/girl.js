import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

async function loadGirl() {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync('asset/girl/scene.gltf');

    console.log('gltf!', gltf);
    return gltf.scene;
}

export { loadGirl };
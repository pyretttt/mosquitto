import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import { PerspectiveRenderUseCase } from './world/render.js';

async function main() {
    const gltf = await new GLTFLoader().loadAsync('asset/jill_vampire/scene.gltf');
    const render_use_case = new PerspectiveRenderUseCase(gltf.scene);

    render_use_case.start();
}

main().catch((err) => {
    console.error(err);
});
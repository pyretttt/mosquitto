import {
    BoxGeometry,
    Color,
    Mesh,
    MeshBasicMaterial,
    PerspectiveCamera,
    Scene,
    WebGLRenderer,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import { PerspectiveRenderUseCase } from './world/render.js';

async function main() {
    const vampire_scene = await new GLTFLoader().loadAsync("asset/jill_vampire/scene.gltf");
    const render_use_case = new PerspectiveRenderUseCase(vampire_scene);

    render_use_case.start();
}

main().catch((err) => {
    console.error(err);
});
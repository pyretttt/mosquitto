import { PerspectiveCamera, WebGLRenderer, Scene, Color, AmbientLight, DirectionalLight, Group } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RenderPipeline, RenderController } from '../world/render.js';

function createLightGroup() {
    const ambientLight = new AmbientLight(new Color('white'), 1);
    const dirLight = new DirectionalLight(new Color('white'), 2);
    dirLight.position.set(1, 1, 1);
    dirLight.lookAt(0, 0, 0);

    const group = new Group();
    group.add(ambientLight, dirLight);
    return group;
}

class PerspectiveRenderUseCase {
    constructor(scene) {
        const rootScene = new Scene();
        rootScene.add(scene);
        rootScene.background = new Color('red');

        const lightGroup = createLightGroup();
        rootScene.add(lightGroup);

        const canvas = document.querySelector('#scene-container');
        const renderer = new WebGLRenderer({
            antialias: true,
            depth: true,
        });
        canvas.append(renderer.domElement);

        const camera = new PerspectiveCamera(
            75,
            16/9,
            0.1,
            1000
        );
        camera.position.set(0, 0, 5);
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.updateTick = (delta) => {
            orbitControls.update();
        };
        renderer.physicallyCorrectLights = true;
        this.renderPipeline = new RenderPipeline(
            camera,
            rootScene,
            renderer
        );

        this.controller = new RenderController(this.renderPipeline, canvas, orbitControls);
        this.controller.updatables.push(orbitControls);
    }

    start() {
        this.controller.startRenderLoop();
    }

    stop() {
        this.controller.stopRenderLoop();
    }
}

export { PerspectiveRenderUseCase };
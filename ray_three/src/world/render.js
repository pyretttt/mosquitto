import { PerspectiveCamera, WebGLRenderer, Timer, Scene, Color } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function resize(camera, renderer, canvas) {
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
}

class RenderController {
    constructor(pipeline, canvas, orbitControls) {
        this.pipeline = pipeline;
        this.isRendering = true;
        this.timer = new Timer();
        this.canvas = canvas;
        this.orbitControls = orbitControls;

        this.updatables = [];
        resize(this.pipeline.camera, this.pipeline.renderer, canvas);
        window.addEventListener('resize', () => {
            resize(this.pipeline.camera, this.pipeline.renderer, this.canvas);
        });
    }

    startRenderLoop() {
        this.pipeline.renderer.setAnimationLoop(() => {
            if (!this.isRendering) {
                this.pipeline.renderer.setAnimationLoop(null);
                return;
            }

            this.timer.update();
            this.updatables.forEach(
                updatable => updatable.updateTick(this.timer.getDelta())
            );
            this.pipeline.render();
        });
    }

    stopRenderLoop() {
        this.isRendering = false;
    }
}

class RenderPipeline {
    constructor(camera, scene, renderer) {
        this.camera = camera;
        this.scene = scene;
        this.renderer = renderer;
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }
}

class PerspectiveRenderUseCase {
    constructor(scene) {
        const rootScene = new Scene();
        rootScene.add(scene);
        rootScene.background = new Color('red');

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
        camera.position.set(0, 0, 10);
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

export { PerspectiveRenderUseCase, RenderController, RenderPipeline };

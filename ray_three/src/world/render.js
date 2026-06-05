import { PerspectiveCamera, WebGLRenderer, Timer } from 'three';

function resize(camera, renderer, canvas) {
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
}

class RenderController {
    constructor(pipeline, canvas) {
        this.pipeline = pipeline;
        this.isRendering = true;
        this.timer = new Timer();
        this.canvas = canvas;

        this.updatables = [];
        window.addEventListener('resize', () => {
            resize(this.pipeline.camera, this.pipeline.renderer, canvas);
        });
    }

    startRenderLoop() {
        this.pipeline.renderer.setAnimationLoop(() => {
            while (this.isRendering) {
                this.timer.update();
                this.updatables.forEach(
                    updatable => updatable.update(this.timer.getDelta())
                );
                this.pipeline.render();
            }
            this.pipeline.renderer.setAnimationLoop(null);
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
        const canvas = document.querySelector('#scene-container').domElement;
        const renderer = new WebGLRenderer({
            canvas: canvas,
            antialias: true,
            depth: true,
        });
        renderer.physicallyCorrectLights = true;
        this.renderPipeline = new RenderPipeline(
            new PerspectiveCamera(
                75,
                16/9,
                0.1,
                1000
            ),
            scene,
            renderer
        );

        this.controller = new RenderController(this.renderPipeline, canvas);
    }

    start() {
        this.controller.startRenderLoop();
    }

    stop() {
        this.controller.stopRenderLoop();
    }
}

export { PerspectiveRenderUseCase, RenderController, RenderPipeline };

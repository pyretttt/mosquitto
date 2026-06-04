import { PerspectiveCamera, WebGLRenderer, Timer } from 'three';

class RenderController {
    constructor(pipeline) {
        this.pipeline = pipeline;
        this.isRendering = true;
        this.timer = new Timer();

        this.updatables = [];
        window.addEventListener('resize', () => {
            this.pipeline.camera.aspect = container.clientWidth / container.clientHeight;
            this.pipeline.camera.updateProjectionMatrix();

            this.pipeline.renderer.setSize(container.clientWidth, container.clientHeight);
            this.pipeline.renderer.setPixelRatio(window.devicePixelRatio);
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
        const renderer = new WebGLRenderer({
            canvas: document.querySelector('#scene-container').domElement,
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

        this.controller = new RenderController(this.renderPipeline);
    }

    start() {
        this.controller.startRenderLoop();
    }

    stop() {
        this.controller.stopRenderLoop();
    }
}

export { PerspectiveRenderUseCase, RenderController, RenderPipeline };

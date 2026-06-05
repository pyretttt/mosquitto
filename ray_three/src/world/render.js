import { PerspectiveCamera, WebGLRenderer, Timer, Scene, Color, AmbientLight, DirectionalLight } from 'three';
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

export { RenderController, RenderPipeline };

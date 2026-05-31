import { Timer } from "three";

class Loop {
    constructor(camera, scene, renderer) {
        this.camera = camera;
        this.scene = scene;
        this.renderer = renderer;
        this.clock = new Timer();
        this.updatables = [];
    }

    start() {
        this.renderer.setAnimationLoop(() => {
            this.clock.update();
            this.tick(this.clock.getDelta());
            this.renderer.render(this.scene, this.camera);
        });
    }

    stop() {
        this.renderer.setAnimationLoop(null);
    }

    tick(delta) {
        for(const object of this.updatables) {
            object.update(delta);
        }
    }
}

export { Loop };

import { PerspectiveRenderUseCase } from './use_cases/perspective_render.js';
import { createSkeletalAnimationDemo } from './use_cases/skeletal_animation.js';

async function main() {
    const demo = createSkeletalAnimationDemo();
    const render_use_case = new PerspectiveRenderUseCase(demo, [demo]);

    render_use_case.start();
}

main().catch((err) => {
    console.error(err);
});
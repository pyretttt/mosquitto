import { DirectionalLight, PointLight, SpotLight, RectAreaLight } from "three";

function createLights() {
    const light = new RectAreaLight('white', 10, 1, 10);

    light.position.set(0, 0, 2);
    return light;
}

export { createLights };
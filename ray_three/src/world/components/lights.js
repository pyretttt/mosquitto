import { DirectionalLight, PointLight, SpotLight, RectAreaLight, AmbientLight, HemisphereLight } from "three";

function createLights() {
    const mainLight = new DirectionalLight('white', 2);
    const ambientLight = new AmbientLight('white', 0.1);

    return { directionalLight: mainLight, ambientLight:ambientLight };
}

export { createLights };
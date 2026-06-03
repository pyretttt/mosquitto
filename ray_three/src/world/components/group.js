import {
    SphereGeometry,
    Group,
    MathUtils,
    Mesh,
    MeshStandardMaterial,
    Sphere,
} from 'three';

function createMeshGroup() {
    const group = new Group();

    const sphere = new SphereGeometry(0.25, 128, 128);
    const material = new MeshStandardMaterial({ color: 'indigo' });
    const protoSphere = new Mesh(sphere, material);

    group.add(protoSphere);

    for (let i = 0; i < 1; i += 0.01) {
        const sphere = protoSphere.clone();

        const x = Math.cos(2 * Math.PI * i);
        const y = Math.sin(2 * Math.PI * i);
        const z = i;
        sphere.position.set(x, y, z);
        sphere.scale.multiplyScalar(0.01 + i);

        group.add(sphere);
    }

    group.scale.multiplyScalar(2);
    group.update = (delta) => {
        group.rotation.z += delta * MathUtils.degToRad(30);
    };

    return group;
}

export { createMeshGroup };
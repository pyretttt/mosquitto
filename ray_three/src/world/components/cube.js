import { BoxGeometry, CircleGeometry, Mesh, MeshBasicMaterial, MeshStandardMaterial, TextureLoader } from 'three';

function createMaterial() {
  const textureLoader = new TextureLoader();
  const texture = textureLoader.load('asset/uv_test.jpg');

  const material = new MeshStandardMaterial({ map: texture });
  material.normalMap = texture;

  return material;
}

function createCube() {
  // create a geometry
  const geometry = new BoxGeometry(2, 2, 2);

  // create a default (white) Basic material
  const material = createMaterial();

  // create a Mesh containing the geometry and material
  const cube = new Mesh(geometry, material);

  // cube.rotation.set(-0.5, -0.1, 0.8);

  return cube;
}

export { createCube };
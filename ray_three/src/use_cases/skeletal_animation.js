import {
    BufferAttribute,
    BufferGeometry,
    Color,
    DoubleSide,
    DynamicDrawUsage,
    Float32BufferAttribute,
    Group,
    LineBasicMaterial,
    LineSegments,
    Matrix3,
    Matrix4,
    Mesh,
    MeshStandardMaterial,
    Quaternion,
    ShaderMaterial,
    SphereGeometry,
    Vector3,
    Vector4,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const MAX_GPU_JOINTS = 64;

function smoothstep(edge0, edge1, value) {
    const x = Math.min(Math.max((value - edge0) / (edge1 - edge0), 0), 1);
    return x * x * (3 - 2 * x);
}

function cloneTransform(transform) {
    return {
        translation: transform.translation.clone(),
        rotation: transform.rotation.clone(),
        scale: transform.scale.clone(),
    };
}

function makeTransform(translation, rotation = new Quaternion(), scale = new Vector3(1, 1, 1)) {
    return {
        translation: translation.clone(),
        rotation: rotation.clone(),
        scale: scale.clone(),
    };
}

function composeTransform(transform) {
    return new Matrix4().compose(transform.translation, transform.rotation, transform.scale);
}

class ScratchSkeleton {
    constructor(joints, inverseBindMatrices = null) {
        this.joints = joints.map((joint) => ({
            name: joint.name,
            parentIndex: joint.parentIndex,
            bindLocal: cloneTransform(joint.bindLocal),
        }));

        this.currentLocals = this.joints.map((joint) => cloneTransform(joint.bindLocal));
        this.bindGlobals = this.joints.map(() => new Matrix4());
        this.currentGlobals = this.joints.map(() => new Matrix4());
        this.inverseBindMatrices = this.joints.map(() => new Matrix4());
        this.skinMatrices = this.joints.map(() => new Matrix4());
        this.normalMatrices = this.joints.map(() => new Matrix3());

        this.computeGlobals(this.joints.map((joint) => joint.bindLocal), this.bindGlobals);

        if (inverseBindMatrices) {
            this.inverseBindMatrices.forEach((matrix, index) => {
                matrix.copy(inverseBindMatrices[index]);
            });
        } else {
            this.inverseBindMatrices.forEach((matrix, index) => {
                matrix.copy(this.bindGlobals[index]).invert();
            });
        }

        this.updateSkinMatrices();
    }

    resetToBindPose() {
        this.currentLocals.forEach((transform, index) => {
            const bind = this.joints[index].bindLocal;
            transform.translation.copy(bind.translation);
            transform.rotation.copy(bind.rotation);
            transform.scale.copy(bind.scale);
        });
    }

    computeGlobals(localTransforms, targetGlobals) {
        localTransforms.forEach((transform, index) => {
            const localMatrix = composeTransform(transform);
            const parentIndex = this.joints[index].parentIndex;

            if (parentIndex < 0) {
                targetGlobals[index].copy(localMatrix);
            } else {
                targetGlobals[index].multiplyMatrices(targetGlobals[parentIndex], localMatrix);
            }
        });
    }

    updateSkinMatrices() {
        this.computeGlobals(this.currentLocals, this.currentGlobals);

        this.skinMatrices.forEach((matrix, index) => {
            matrix.multiplyMatrices(this.currentGlobals[index], this.inverseBindMatrices[index]);
            this.normalMatrices[index].setFromMatrix4(matrix);
        });
    }

    getJointPosition(index, target = new Vector3(), bindPose = false) {
        const source = bindPose ? this.bindGlobals : this.currentGlobals;
        return target.setFromMatrixPosition(source[index]);
    }

    clone() {
        return new ScratchSkeleton(this.joints, this.inverseBindMatrices);
    }
}

function createToySkeleton() {
    const joints = [
        {
            name: 'root',
            parentIndex: -1,
            bindLocal: makeTransform(new Vector3(0, 0, 0)),
        },
        {
            name: 'child',
            parentIndex: 0,
            bindLocal: makeTransform(new Vector3(0, 1, 0)),
        },
    ];

    return new ScratchSkeleton(joints);
}

function createToyStripGeometry() {
    const height = 2;
    const width = 0.42;
    const segments = 24;
    const positions = [];
    const normals = [];
    const skinIndices = [];
    const skinWeights = [];
    const indices = [];

    for (let row = 0; row <= segments; row += 1) {
        const t = row / segments;
        const y = t * height;
        const childWeight = smoothstep(0.35, 1.45, y);
        const rootWeight = 1 - childWeight;

        for (let side = 0; side < 2; side += 1) {
            const x = side === 0 ? -width * 0.5 : width * 0.5;
            positions.push(x, y, 0);
            normals.push(0, 0, 1);
            skinIndices.push(0, 1, 0, 0);
            skinWeights.push(rootWeight, childWeight, 0, 0);
        }

        if (row < segments) {
            const left = row * 2;
            const right = left + 1;
            const nextLeft = left + 2;
            const nextRight = left + 3;

            indices.push(left, nextLeft, right);
            indices.push(right, nextLeft, nextRight);
        }
    }

    const geometry = new BufferGeometry();
    geometry.setIndex(indices);
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
    geometry.setAttribute('skinIndex', new Float32BufferAttribute(skinIndices, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeights, 4));
    geometry.computeBoundingSphere();

    return geometry;
}

class CpuSkinnedMesh extends Mesh {
    constructor(bindGeometry, skeleton) {
        const geometry = bindGeometry.clone();
        const material = new MeshStandardMaterial({
            color: new Color('#65a9ff'),
            metalness: 0,
            roughness: 0.45,
            side: DoubleSide,
        });

        super(geometry, material);

        this.skeleton = skeleton;
        this.bindPositions = bindGeometry.getAttribute('position').array.slice();
        this.bindNormals = bindGeometry.getAttribute('normal').array.slice();
        this.skinIndex = bindGeometry.getAttribute('skinIndex');
        this.skinWeight = bindGeometry.getAttribute('skinWeight');

        this.geometry.getAttribute('position').setUsage(DynamicDrawUsage);
        this.geometry.getAttribute('normal').setUsage(DynamicDrawUsage);
    }

    updateSkin() {
        this.skeleton.updateSkinMatrices();

        const positions = this.geometry.getAttribute('position');
        const normals = this.geometry.getAttribute('normal');
        const bindPosition = new Vector4();
        const weightedPosition = new Vector4();
        const skinnedPosition = new Vector4();
        const bindNormal = new Vector3();
        const weightedNormal = new Vector3();
        const skinnedNormal = new Vector3();

        for (let vertex = 0; vertex < positions.count; vertex += 1) {
            bindPosition.set(
                this.bindPositions[vertex * 3],
                this.bindPositions[vertex * 3 + 1],
                this.bindPositions[vertex * 3 + 2],
                1,
            );
            bindNormal.set(
                this.bindNormals[vertex * 3],
                this.bindNormals[vertex * 3 + 1],
                this.bindNormals[vertex * 3 + 2],
            );

            weightedPosition.set(0, 0, 0, 0);
            weightedNormal.set(0, 0, 0);

            for (let influence = 0; influence < 4; influence += 1) {
                const weight = this.skinWeight.getComponent(vertex, influence);
                if (weight === 0) {
                    continue;
                }

                const jointIndex = this.skinIndex.getComponent(vertex, influence);

                skinnedPosition.copy(bindPosition).applyMatrix4(this.skeleton.skinMatrices[jointIndex]);
                weightedPosition.addScaledVector(skinnedPosition, weight);

                skinnedNormal.copy(bindNormal).applyMatrix3(this.skeleton.normalMatrices[jointIndex]);
                weightedNormal.addScaledVector(skinnedNormal, weight);
            }

            weightedNormal.normalize();
            positions.setXYZ(vertex, weightedPosition.x, weightedPosition.y, weightedPosition.z);
            normals.setXYZ(vertex, weightedNormal.x, weightedNormal.y, weightedNormal.z);
        }

        positions.needsUpdate = true;
        normals.needsUpdate = true;
        this.geometry.computeBoundingSphere();
    }
}

function createGpuSkinningMaterial(jointMatrices, jointCount) {
    return new ShaderMaterial({
        uniforms: {
            jointMatrices: { value: jointMatrices },
            color: { value: new Color('#f5b86b') },
            lightDirection: { value: new Vector3(0.4, 0.8, 0.5).normalize() },
        },
        vertexShader: `
            const int JOINT_COUNT = ${jointCount};

            uniform mat4 jointMatrices[JOINT_COUNT];

            attribute vec4 skinIndex;
            attribute vec4 skinWeight;

            varying vec3 vNormal;

            mat4 getJointMatrix(float index) {
                int jointIndex = int(index);

                for (int i = 0; i < JOINT_COUNT; i += 1) {
                    if (i == jointIndex) {
                        return jointMatrices[i];
                    }
                }

                return mat4(1.0);
            }

            void main() {
                mat4 skin =
                    skinWeight.x * getJointMatrix(skinIndex.x) +
                    skinWeight.y * getJointMatrix(skinIndex.y) +
                    skinWeight.z * getJointMatrix(skinIndex.z) +
                    skinWeight.w * getJointMatrix(skinIndex.w);

                vec4 skinnedPosition = skin * vec4(position, 1.0);
                vNormal = normalize(mat3(skin) * normal);

                gl_Position = projectionMatrix * modelViewMatrix * skinnedPosition;
            }
        `,
        fragmentShader: `
            uniform vec3 color;
            uniform vec3 lightDirection;

            varying vec3 vNormal;

            void main() {
                float light = max(dot(normalize(vNormal), normalize(lightDirection)), 0.0);
                vec3 shaded = color * (0.25 + 0.75 * light);
                gl_FragColor = vec4(shaded, 1.0);
            }
        `,
        side: DoubleSide,
    });
}

class GpuSkinnedMesh extends Mesh {
    constructor(bindGeometry, skeleton) {
        if (skeleton.joints.length > MAX_GPU_JOINTS) {
            throw new Error(`GPU skinning supports ${MAX_GPU_JOINTS} joints, got ${skeleton.joints.length}.`);
        }

        const gpuMatrices = Array.from({ length: skeleton.joints.length }, () => new Matrix4());
        skeleton.skinMatrices.forEach((matrix, index) => {
            gpuMatrices[index].copy(matrix);
        });

        super(bindGeometry.clone(), createGpuSkinningMaterial(gpuMatrices, skeleton.joints.length));

        this.skeleton = skeleton;
        this.gpuMatrices = gpuMatrices;
    }

    updateSkin() {
        this.skeleton.updateSkinMatrices();

        this.skeleton.skinMatrices.forEach((matrix, index) => {
            this.gpuMatrices[index].copy(matrix);
        });

        this.material.uniformsNeedUpdate = true;
    }
}

function findKeyframeSegment(keys, time) {
    if (time <= keys[0].time) {
        return [keys[0], keys[0], 0];
    }

    for (let index = 0; index < keys.length - 1; index += 1) {
        const from = keys[index];
        const to = keys[index + 1];

        if (time >= from.time && time <= to.time) {
            const alpha = (time - from.time) / (to.time - from.time);
            return [from, to, alpha];
        }
    }

    const last = keys[keys.length - 1];
    return [last, last, 0];
}

function sampleVectorKeys(keys, time, target) {
    const [from, to, alpha] = findKeyframeSegment(keys, time);
    return target.copy(from.value).lerp(to.value, alpha);
}

function sampleQuaternionKeys(keys, time, target) {
    const [from, to, alpha] = findKeyframeSegment(keys, time);
    return target.slerpQuaternions(from.value, to.value, alpha);
}

class ScratchAnimationClip {
    constructor(duration, channels) {
        this.duration = duration;
        this.channels = channels;
    }

    sample(time, skeleton) {
        const localTime = ((time % this.duration) + this.duration) % this.duration;
        skeleton.resetToBindPose();

        this.channels.forEach((channel) => {
            const transform = skeleton.currentLocals[channel.jointIndex];

            if (channel.translationKeys) {
                sampleVectorKeys(channel.translationKeys, localTime, transform.translation);
            }

            if (channel.rotationKeys) {
                sampleQuaternionKeys(channel.rotationKeys, localTime, transform.rotation);
            }

            if (channel.scaleKeys) {
                sampleVectorKeys(channel.scaleKeys, localTime, transform.scale);
            }
        });
    }
}

function rotationKey(time, angleRadians) {
    return {
        time,
        value: new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), angleRadians),
    };
}

function createToyBendClip() {
    return new ScratchAnimationClip(4, [
        {
            jointIndex: 1,
            rotationKeys: [
                rotationKey(0, 0),
                rotationKey(1, Math.PI * 0.42),
                rotationKey(2, 0),
                rotationKey(3, -Math.PI * 0.42),
                rotationKey(4, 0),
            ],
        },
    ]);
}

class SkeletonDebugView extends Group {
    constructor(skeleton, color) {
        super();

        this.skeleton = skeleton;
        this.linePositions = new Float32Array(Math.max(skeleton.joints.length - 1, 1) * 6);
        this.lineGeometry = new BufferGeometry();
        this.lineGeometry.setAttribute('position', new BufferAttribute(this.linePositions, 3));
        this.add(new LineSegments(
            this.lineGeometry,
            new LineBasicMaterial({ color }),
        ));

        this.jointMarkers = skeleton.joints.map(() => {
            const marker = new Mesh(
                new SphereGeometry(0.035, 12, 8),
                new MeshStandardMaterial({ color }),
            );
            this.add(marker);
            return marker;
        });
    }

    updateDebug(bindPose = false) {
        this.skeleton.computeGlobals(this.skeleton.currentLocals, this.skeleton.currentGlobals);

        let lineOffset = 0;
        const parentPosition = new Vector3();
        const childPosition = new Vector3();

        this.skeleton.joints.forEach((joint, index) => {
            const jointPosition = this.skeleton.getJointPosition(index, childPosition, bindPose);
            this.jointMarkers[index].position.copy(jointPosition);

            if (joint.parentIndex < 0) {
                return;
            }

            this.skeleton.getJointPosition(joint.parentIndex, parentPosition, bindPose);

            this.linePositions[lineOffset] = parentPosition.x;
            this.linePositions[lineOffset + 1] = parentPosition.y;
            this.linePositions[lineOffset + 2] = parentPosition.z;
            this.linePositions[lineOffset + 3] = jointPosition.x;
            this.linePositions[lineOffset + 4] = jointPosition.y;
            this.linePositions[lineOffset + 5] = jointPosition.z;
            lineOffset += 6;
        });

        this.lineGeometry.getAttribute('position').needsUpdate = true;
    }
}

function createLabelPlane() {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute([
        -0.45, -0.03, 0,
        0.45, -0.03, 0,
        0.45, 0.03, 0,
        -0.45, 0.03, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    return geometry;
}

class SkeletalAnimationDemo extends Group {
    constructor() {
        super();

        this.time = 0;
        this.clip = createToyBendClip();

        const bindGeometry = createToyStripGeometry();
        this.cpuSkeleton = createToySkeleton();
        this.gpuSkeleton = this.cpuSkeleton.clone();

        this.cpuMesh = new CpuSkinnedMesh(bindGeometry, this.cpuSkeleton);
        this.cpuMesh.position.x = -0.65;
        this.add(this.cpuMesh);

        this.gpuMesh = new GpuSkinnedMesh(bindGeometry, this.gpuSkeleton);
        this.gpuMesh.position.x = 0.65;
        this.add(this.gpuMesh);

        this.cpuDebug = new SkeletonDebugView(this.cpuSkeleton, new Color('#003cff'));
        this.cpuDebug.position.copy(this.cpuMesh.position);
        this.add(this.cpuDebug);

        this.gpuDebug = new SkeletonDebugView(this.gpuSkeleton, new Color('#a85600'));
        this.gpuDebug.position.copy(this.gpuMesh.position);
        this.add(this.gpuDebug);

        this.bindPoseDebug = new SkeletonDebugView(this.cpuSkeleton, new Color('#ffffff'));
        this.bindPoseDebug.position.set(0, 0, -0.04);
        this.bindPoseDebug.updateDebug(true);
        this.add(this.bindPoseDebug);

        this.add(new Mesh(
            createLabelPlane(),
            new MeshStandardMaterial({ color: new Color('#1f1f1f'), side: DoubleSide }),
        ));
    }

    updateTick(delta) {
        this.time += delta;

        this.clip.sample(this.time, this.cpuSkeleton);
        this.clip.sample(this.time, this.gpuSkeleton);

        this.cpuMesh.updateSkin();
        this.gpuMesh.updateSkin();
        this.cpuDebug.updateDebug();
        this.gpuDebug.updateDebug();
    }
}

function parseTrackName(trackName, jointNames) {
    const propertyMatch = trackName.match(/\\.(position|quaternion|scale)$/);
    if (!propertyMatch) {
        return null;
    }

    const property = propertyMatch[1];
    const targetName = trackName.slice(0, -property.length - 1).replace(/^\\.bones\\[|\\]$/g, '');
    const jointIndex = jointNames.indexOf(targetName);

    if (jointIndex < 0) {
        return null;
    }

    return { jointIndex, property };
}

function extractAnimationChannels(animation, jointNames) {
    return animation.tracks
        .map((track) => {
            const parsed = parseTrackName(track.name, jointNames);
            if (!parsed) {
                return null;
            }

            return {
                jointIndex: parsed.jointIndex,
                property: parsed.property,
                times: Array.from(track.times),
                values: Array.from(track.values),
                interpolation: track.getInterpolation(),
            };
        })
        .filter(Boolean);
}

function extractGltfSkeletonData(gltf) {
    const skins = [];

    gltf.scene.traverse((object) => {
        if (!object.isSkinnedMesh) {
            return;
        }

        const skeleton = object.skeleton;
        const jointNames = skeleton.bones.map((bone) => bone.name);
        const joints = skeleton.bones.map((bone) => ({
            name: bone.name,
            parentIndex: skeleton.bones.indexOf(bone.parent),
            bindLocal: makeTransform(bone.position, bone.quaternion, bone.scale),
            sourceNode: bone,
        }));

        skins.push({
            name: object.name,
            joints,
            inverseBindMatrices: skeleton.boneInverses.map((matrix) => matrix.clone()),
            geometry: {
                position: object.geometry.getAttribute('position'),
                normal: object.geometry.getAttribute('normal'),
                skinIndex: object.geometry.getAttribute('skinIndex'),
                skinWeight: object.geometry.getAttribute('skinWeight'),
                index: object.geometry.index,
            },
            animations: gltf.animations.map((animation) => ({
                name: animation.name,
                duration: animation.duration,
                channels: extractAnimationChannels(animation, jointNames),
            })),
        });
    });

    return {
        scene: gltf.scene,
        skins,
        animations: gltf.animations,
    };
}

function createGeometryFromGltfSkinData(skinData) {
    const geometry = new BufferGeometry();

    geometry.setAttribute('position', skinData.geometry.position.clone());
    geometry.setAttribute('skinIndex', skinData.geometry.skinIndex.clone());
    geometry.setAttribute('skinWeight', skinData.geometry.skinWeight.clone());

    if (skinData.geometry.index) {
        geometry.setIndex(skinData.geometry.index.clone());
    }

    if (skinData.geometry.normal) {
        geometry.setAttribute('normal', skinData.geometry.normal.clone());
    } else {
        geometry.computeVertexNormals();
    }

    geometry.computeBoundingSphere();
    return geometry;
}

function createSkeletonFromGltfSkinData(skinData) {
    return new ScratchSkeleton(skinData.joints, skinData.inverseBindMatrices);
}

function createVectorKeys(times, values) {
    return times.map((time, index) => ({
        time,
        value: new Vector3(
            values[index * 3],
            values[index * 3 + 1],
            values[index * 3 + 2],
        ),
    }));
}

function createQuaternionKeys(times, values) {
    return times.map((time, index) => ({
        time,
        value: new Quaternion(
            values[index * 4],
            values[index * 4 + 1],
            values[index * 4 + 2],
            values[index * 4 + 3],
        ),
    }));
}

function createScratchClipFromGltfAnimation(animationData) {
    const channelsByJoint = new Map();

    animationData.channels.forEach((channel) => {
        if (!channelsByJoint.has(channel.jointIndex)) {
            channelsByJoint.set(channel.jointIndex, { jointIndex: channel.jointIndex });
        }

        const targetChannel = channelsByJoint.get(channel.jointIndex);

        if (channel.property === 'position') {
            targetChannel.translationKeys = createVectorKeys(channel.times, channel.values);
        }

        if (channel.property === 'quaternion') {
            targetChannel.rotationKeys = createQuaternionKeys(channel.times, channel.values);
        }

        if (channel.property === 'scale') {
            targetChannel.scaleKeys = createVectorKeys(channel.times, channel.values);
        }
    });

    return new ScratchAnimationClip(animationData.duration, Array.from(channelsByJoint.values()));
}

async function loadGltfSkeletonData(url, loader = new GLTFLoader()) {
    const gltf = await loader.loadAsync(url);
    return extractGltfSkeletonData(gltf);
}

function createSkeletalAnimationDemo() {
    return new SkeletalAnimationDemo();
}

export {
    CpuSkinnedMesh,
    GpuSkinnedMesh,
    ScratchAnimationClip,
    ScratchSkeleton,
    createSkeletalAnimationDemo,
    createGeometryFromGltfSkinData,
    createScratchClipFromGltfAnimation,
    createSkeletonFromGltfSkinData,
    createToyBendClip,
    createToySkeleton,
    createToyStripGeometry,
    extractGltfSkeletonData,
    loadGltfSkeletonData,
};

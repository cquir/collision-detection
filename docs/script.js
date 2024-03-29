import * as THREE from 'https://unpkg.com/three/build/three.module.js';
import { OrbitControls } from './OrbitControls.js'

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcbbeb5);

// resizing
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}
window.addEventListener('resize', () =>
{
    sizes.width = window.innerWidth;
    sizes.height = window.innerHeight;
    camera.aspect = sizes.width/sizes.height;
    camera.updateProjectionMatrix();
    renderer.setSize(sizes.width,sizes.height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
})

// camera
const camera = new THREE.PerspectiveCamera(25,sizes.width/sizes.height,0.1,100);
camera.position.set(0,0,20);

// renderer
const canvas = document.querySelector('canvas.webgl');
const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
});
renderer.setSize(sizes.width,sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera,renderer.domElement);
controls.update();

// light
const light = new THREE.DirectionalLight({color:'#ffffff',intensity:1});
light.position.set(0,2,0);
light.castShadow = true;
light.shadow.mapSize.width = 1024;
light.shadow.mapSize.height = 1024;
light.shadow.camera.far = 3;
light.shadow.camera.near = -3;
light.shadow.camera.top = 3;
light.shadow.camera.bottom = -3;
light.shadow.camera.right = 3;
light.shadow.camera.left = -3;
scene.add(light);

// plane
const planeGeometry = new THREE.PlaneGeometry(100,100);
const planeMaterial = new THREE.ShadowMaterial();
planeMaterial.opacity = 0.4;
const plane = new THREE.Mesh(planeGeometry,planeMaterial);
plane.receiveShadow = true;
plane.rotateX(-Math.PI/2);
scene.add(plane);

// cubes
const cubeGeometry = new THREE.BoxGeometry(1,1,1);
const cubeAmaterial = new THREE.MeshToonMaterial({color:'#525266'});
const cubeBmaterial = new THREE.MeshToonMaterial({color:'#ff6666'});
const cubeA = new THREE.Mesh(cubeGeometry,cubeAmaterial);
const cubeB = new THREE.Mesh(cubeGeometry,cubeBmaterial);
cubeA.castShadow = true;
cubeB.castShadow = true;
cubeA.receiveShadow = false;
cubeB.receiveShadow = false;

// group
const group = new THREE.Group();
group.add(cubeA);
group.add(cubeB);
scene.add(group);
group.position.set(0,1+Math.sqrt(3)/2,0);

function update(){
    // generate new example
    const relativePosition = new THREE.Vector3().random().addScalar(-0.5).multiplyScalar(1+Math.sqrt(3));
    const relativeQuaternion = new THREE.Quaternion().random();
    const groupQuaternion = new THREE.Quaternion().random();

    // update cubes' positions + rotations
    cubeB.position.copy(new THREE.Vector3()).add(relativePosition);
    cubeB.quaternion.copy(new THREE.Quaternion());
    cubeB.applyQuaternion(relativeQuaternion);
    group.quaternion.copy(groupQuaternion);

    // update collision prediction
    async function main(){
        const session = await ort.InferenceSession.create('model-lemon-sky-347.onnx');
        const data = Float32Array.from([
            relativePosition.x,
            relativePosition.y,
            relativePosition.z,
            relativeQuaternion.x,
            relativeQuaternion.y,
            relativeQuaternion.z,
            relativeQuaternion.w
        ]);
        const tensor = new ort.Tensor('float32',data,[1,7]);
        const feeds = {input: tensor};
        const results = await session.run(feeds);
        const output = results['output']['data'][0];
        const prob = Math.round(10000/(1+Math.exp(-output)))/100;
        document.getElementById("predictions").innerHTML = 'probability of collision = '+prob.toString()+' %';
    }
    main()
}

update();

document.addEventListener('keyup', event => {
    if (event.code === 'Space'){
        update();
    }
})

// animate
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}

animate()
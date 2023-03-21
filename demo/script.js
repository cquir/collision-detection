import * as THREE from 'https://unpkg.com/three/build/three.module.js';
import { OrbitControls } from './OrbitControls.js'

// **************************************************************
// NEXT: feed cubeB's quaternion (as relative quaternion)

const relX = (1+Math.sqrt(3))*(Math.random()-0.5);
const relY = (1+Math.sqrt(3))*(Math.random()-0.5);
const relZ = (1+Math.sqrt(3))*(Math.random()-0.5);

async function main(){
    const session = await ort.InferenceSession.create('model-sleek-breeze-268.onnx');
    const data = Float32Array.from([relX,relY,relZ,0,0,0,0]);
    const tensor = new ort.Tensor('float32',data,[1,7]);
    const feeds = {input: tensor};
    const results = await session.run(feeds);
    const output = results['output']['data'][0];
    const prob = 1/(1+Math.exp(-output));
    console.log(prob);
}
main()

// **************************************************************


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
camera.position.set(0,0,10);

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
//const helper = new THREE.CameraHelper(light.shadow.camera);
//scene.add(helper);

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
cubeA.position.set(0,1+Math.sqrt(3)/2,0);
cubeB.position.set(
    cubeA.position.x + relX,
    cubeA.position.y + relY,
    cubeA.position.z + relZ 
);
scene.add(cubeA);
scene.add(cubeB);

// animate
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}

animate()
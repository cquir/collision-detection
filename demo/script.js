import * as THREE from 'https://unpkg.com/three/build/three.module.js';

async function main(){
    const session = await ort.InferenceSession.create('model-sleek-breeze-268.onnx');
    const data = Float32Array.from([0,0,0,0,0,0,0]);
    const tensor = new ort.Tensor('float32',data,[1,7]);
    const feeds = {input: tensor};
    const results = await session.run(feeds);
    console.log(results['output']['data'][0])
}

main()

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcbbeb5);

const camera = new THREE.PerspectiveCamera(25,window.innerWidth/window.innerHeight,0.1,100);
camera.position.set(0,3,10);
camera.lookAt(0,3*Math.sqrt(3)/4,0);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.antialias = true;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// Cube
const cubeGeometry = new THREE.BoxGeometry(1,1,1);
const cubeMaterial = new THREE.MeshToonMaterial({color:'#ff6666'});
const cube = new THREE.Mesh(cubeGeometry,cubeMaterial);
cube.castShadow = true;
cube.receiveShadow = false;
cube.position.set(0,1,0);
scene.add(cube);

// Light
const light = new THREE.DirectionalLight({color:'#ffffff',intensity:1});
light.position.set(0,1,0);
light.castShadow = true;
light.shadow.mapSize.width = 1024;
light.shadow.mapSize.height = 1024;
light.shadow.camera.top = 2;
light.shadow.camera.bottom = -2;
light.shadow.camera.right = 2;
light.shadow.camera.left = -2;
scene.add(light);

// Plane
const planeGeometry = new THREE.PlaneGeometry(100,100);
const planeMaterial = new THREE.ShadowMaterial();
planeMaterial.opacity = 0.4;
const plane = new THREE.Mesh(planeGeometry,planeMaterial);
plane.receiveShadow = true;
plane.rotateX(-Math.PI/2);
scene.add(plane);

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene,camera);
}

animate()
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

// **************************************

// Missing: orbitControls

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
camera.position.set(0,3,10);
camera.lookAt(0,3*Math.sqrt(3)/4,0);

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

// light
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
cubeA.castShadow = true;
cubeA.receiveShadow = false;
cubeA.position.set(0,1,0);
scene.add(cubeA);
const cubeB = new THREE.Mesh(cubeGeometry,cubeBmaterial);
cubeB.castShadow = true;
cubeB.receiveShadow = false;
cubeB.position.set(1.5,1,0);
scene.add(cubeB);

// animate
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene,camera);
}

animate()
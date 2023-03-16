import { createRoot } from 'react-dom/client'
import React, { useRef } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { PresentationControls, softShadows } from '@react-three/drei'
import { Tensor, InferenceSession } from 'onnxjs'

softShadows()
let keydown = false

/*
const session = new InferenceSession()
const url = "/model-earthy-durian-312.onnx"
await session.loadModel(url)
*/

function Light(props){
	return (
	<directionalLight
		color={'#ffffff'} 
		intensity={1} 
		position={[0,1,0]} 
		castShadow
		shadow-mapSize-width={1024}
		shadow-mapSize-height={1024}
		shadow-camera-far={2}
		shadow-camera-top={2}
		shadow-camera-bottom={-2}
		shadow-camera-right={2}
		shadow-camera-left={-2}/> 
	)
}

function Plane(props){
	useThree((state) => { state.camera.lookAt(0,3*Math.sqrt(3)/4,0)})
	return (
		<mesh receiveShadow rotation={[-Math.PI/2,0,0]}>
			<planeGeometry args={[100,100]}/>
			<shadowMaterial transparent opacity={0.4}/>
		</mesh>
	)
}

function Positions(){
	let xs = [], ys = [], zs = []
	for (let i=0; i<2; i++){
		xs.push(Math.sqrt(3)*(Math.random()-0.5))
		ys.push(Math.sqrt(3)*(Math.random()-0.5))
		zs.push(Math.sqrt(3)*(Math.random()-0.5))
	}
	let min = Math.min(ys[0],ys[1])
	ys[0] = ys[0]-min+Math.sqrt(3)/2
	ys[1] = ys[1]-min+Math.sqrt(3)/2
	return [xs[0],ys[0],zs[0],xs[1],ys[1],zs[1]]
}

function Quaternion(){
	let u = Math.random()
	let v = Math.random()
	let w = Math.random()
	let a = Math.sqrt(1-u)*Math.sin(2*Math.PI*v) 
	let b = Math.sqrt(1-u)*Math.cos(2*Math.PI*v)
	let c = Math.sqrt(u)*Math.sin(2*Math.PI*w)
	let d = Math.sqrt(u)*Math.cos(2*Math.PI*w)
	return [a,b,c,d]
}

function Cubes(props){
	const cube0 = useRef()
	const cube1 = useRef()
	useFrame(() => {
		let onKeyDown = () => {
			let [x0,y0,z0,x1,y1,z1] = Positions()
			let [a0,b0,c0,d0] = Quaternion()
			let [a1,b1,c1,d1] = Quaternion()
			cube0.current.position.set(x0,y0,z0)
			cube1.current.position.set(x1,y1,z1)
			cube0.current.quaternion.set(a0,b0,c0,d0)
			cube1.current.quaternion.set(a1,b1,c1,d1)
			let text = (Math.random() > 0.5)? 'COLLISION':'NO COLLISION'
			document.getElementById('header').innerHTML = text
		}
		document.addEventListener('keydown',(e) => {
			if (e.key === " "){
				if (!keydown) {
					keydown = true
					onKeyDown()
				}
			}
		})
		document.addEventListener('keyup',(e) => {
			if (e.key === " "){
				if (keydown){
					keydown = false
				}
			}
		})
	})
	let [x0,y0,z0,x1,y1,z1] = Positions()
	let [a0,b0,c0,d0] = Quaternion()
	let [a1,b1,c1,d1] = Quaternion()
	return (
	<group>
		<mesh ref={cube0} castShadow position={[x0,y0,z0]} quaternion={[a0,b0,c0,d0]}>
			<boxGeometry />
			<meshToonMaterial color={'#ff6666'}/>
		</mesh>
		<mesh ref={cube1} castShadow position={[x1,y1,z1]} quaternion={[a1,b1,c1,d1]}>
			<boxGeometry />
			<meshToonMaterial color={'#525266'}/>
		</mesh>
	</group>
	)
}

function Header(props){
	return(
	<div style={{position:'absolute',top:'10%',left:'50%',transform: 'translate3d(-50%,-50%,0)'}}>
		<h1 id='header' style={
			{fontFamily:"'Roboto', sans-serif",
			fontSize:'50px',
			fontWeight:'900',
			textAlign:'center',
			color:'white',
			letterSpacing:'1em'}}>
			COLLISION</h1>
	</div>
	)
}

function Paragraph(props){
	return(
	<div style={{position:'absolute',top:'90%',left:'50%',transform: 'translate3d(-50%,-50%,0)'}}>
		<p style={
			{fontFamily:"'Roboto', sans-serif",
			fontSize:'25px',
			fontWeight:'300',
			textAlign:'center',
			color:'#423f3b'
			}}>
			Press the spacebar for a new example.</p>
		<p style={
			{fontFamily:"'Roboto', sans-serif",
			fontSize:'25px',
			fontWeight:'300',
			textAlign:'center',
			color:'#423f3b'}}>
			Click and drag to move around.</p>
	</div>
	)
}

createRoot(document.getElementById('root')).render(
	<div id='canvas-container' style={{width:window.innerWidth,height:window.innerHeight}}>
		<Canvas shadows camera={{fov:25,position:[0,3,10]}}>
			<color attach='background' args={['#cbbeb5']}/>
				<PresentationControls global>
					<Light/>
					<Plane/>
					<Cubes/>
				</PresentationControls>
		</Canvas>
		<Header/>
		<Paragraph/>
	</div>
)
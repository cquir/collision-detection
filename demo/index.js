import ReactDOM from 'react-dom'
import { Canvas, useFrame } from '@react-three/fiber'
import { PresentationControls, softShadows } from '@react-three/drei'
import './index.css'

softShadows()

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
	useFrame((state) => { state.camera.lookAt(0,3*Math.sqrt(3)/4,0)})
	return (
		<mesh receiveShadow rotation={[-Math.PI/2,0,0]}>
			<planeGeometry args={[100,100]}/>
			<shadowMaterial transparent opacity={0.4}/>
		</mesh>
	)
}

let Heights = () => {
	let y0 = Math.sqrt(3)*(Math.random()-0.5) 
	let y1 = Math.sqrt(3)*(Math.random()-0.5) 
	let min = Math.min(y0,y1)
	y0 = y0-min+Math.sqrt(3)/2
	y1 = y1-min+Math.sqrt(3)/2
	return [y0,y1]
}

function Cube(props){
	let x = Math.sqrt(3)*(Math.random()-0.5)
	let z = Math.sqrt(3)*(Math.random()-0.5)
	let u = Math.random()
	let v = Math.random()
	let w = Math.random()
	let a = Math.sqrt(1-u)*Math.sin(2*Math.PI*v)
	let b = Math.sqrt(1-u)*Math.cos(2*Math.PI*v)
	let c = Math.sqrt(u)*Math.sin(2*Math.PI*w)
	let d = Math.sqrt(u)*Math.cos(2*Math.PI*w)
	return (
	<mesh castShadow position={[x,props.y,z]} quaternion={[a,b,c,d]}>
		<boxGeometry />
		<meshToonMaterial color={props.color}/>
	</mesh>
	)
}

function Header(props){
	return(
	<div style={{position:'absolute',top:'10%',left:'50%',transform: 'translate3d(-50%,-50%,0)'}}>
		<h1 style={
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

// next: press spacebar for new collision + random 'Collision!' or 'No collision!' text 

function App() {
	let ys = Heights()
	return (
		<div id='canvas-container' style={{width:window.innerWidth,height:window.innerHeight}}>
			<Canvas shadows camera={{fov:25,position:[0,3,10]}}>
				<color attach='background' args={['#cbbeb5']}/>
					<PresentationControls global>
						<Light/>
						<Plane/>
						<Cube y={ys[0]} color={'#ff6666'}/>
						<Cube y={ys[1]} color={'#525266'}/>
					</PresentationControls>
			</Canvas>
			<Header/>
			<Paragraph/>
		</div>
	)
}

ReactDOM.render(<App />, document.getElementById('root'))

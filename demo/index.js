import ReactDOM from 'react-dom'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'


function Cube(props){
	let x = Math.sqrt(3)*(Math.random()-0.5)
	let y = Math.sqrt(3)*(Math.random()-0.5)
	let z = Math.sqrt(3)*(Math.random()-0.5)
	let u = Math.random()
	let v = Math.random()
	let w = Math.random()
	let a = Math.sqrt(1-u)*Math.sin(2*Math.PI*v)
	let b = Math.sqrt(1-u)*Math.cos(2*Math.PI*v)
	let c = Math.sqrt(u)*Math.sin(2*Math.PI*v)
	let d = Math.sqrt(u)*Math.cos(2*Math.PI*w)
	return (
	<mesh position={[x,y,z]} quaternion={[a,b,c,d]}>
		<boxGeometry />
		<meshToonMaterial {...props}/>
	</mesh>
	)
}

// next: press spacebar for new collision

function App() {
	return (
		<div id="canvas-container" style={{width:window.innerWidth,height:window.innerHeight}}>
			<Canvas camera={{position:[0,0,7]}}>
					<OrbitControls/>
					<directionalLight color={'#ffffff'} intensity={1} position={[0,1,0]}/> 
					<Cube color={'#525e98'}/>
					<Cube color={'#da6f46'}/>
			</Canvas>
			<div style={{position:'absolute',top:'90%',left:'50%',transform: 'translate3d(-50%,-50%,0)'}}>
				<h1 style={{fontSize:'100px'}}>TEST</h1>
			</div>
		</div>
	)
}

ReactDOM.render(<App />, document.getElementById('root'))

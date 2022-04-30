import scipy.spatial
import subprocess
import collide
import numpy
import click
import os

# generate random uniform quaternion
def uniform_quaternion():
	u = numpy.random.normal(size=3)
	u /= numpy.sqrt(numpy.sum(u**2))
	theta = numpy.random.uniform(0,2*numpy.pi)
	return [numpy.cos(theta/2),*u*numpy.sin(theta/2)]

# generate random quaternion where theta is sampled from a beta distribution
def rbetatheta():
	inside = False
	while not inside:
		scale = 6.3282388368044025; loc = -0.018771213624692785
		theta = numpy.random.beta(a=1.45655417928243,b=1.4973878857803622)*scale+loc	
		inside = (0 <= theta <= 2*numpy.pi)
	u = numpy.random.normal(size=3)
	u /= numpy.sqrt(numpy.sum(u**2))
	return [numpy.cos(theta/2),*u*numpy.sin(theta/2)]


# generate random position where distance from center is sampled from a normal distribution
def rnormd():
	d = numpy.random.normal(loc=1.3957095697470043,scale=0.10287206575906595)
	inside = False 
	while not inside:
		u = numpy.random.normal(size=3)
		u /= numpy.sqrt(numpy.sum(u**2))
		pos = u*d
		x,y,z  = pos; hl = (numpy.sqrt(3)+1)/2.
		inside = (-hl <= x <= hl) and (-hl <= y <= hl) and (-hl <= z <= hl)
	return pos

# generate example comprising of random position + quaternion 
def example(label,split):
	hl = (numpy.sqrt(3)+1)/2.
	pos = numpy.random.uniform(-hl,hl,3)
	q = uniform_quaternion()
	if 'train' in label:
		if numpy.random.rand() < split:
			pos = rnormd()
			q = rbetatheta()
	return pos,q

@click.command()
@click.option('--Ntot','Ntot',default=1e6,help='Number of examples')
@click.option('--split','split',default=0.0,help='Split between sampling distributions for training')
def gen(Ntot,split):
	if not os.path.isdir('../../data/datasets/'):
		os.mkdir('../../data/datasets/')
	numpy.random.seed(0)
	for dataset,perc in zip([f'train','validation','test'],[0.64, 0.16,0.2]):
		N = int(Ntot*perc)
		X = []; Y = []
		for i in range(int(N)):
			# generate example
			pos,q = example(dataset,split)
			
			# check if there is a collision 
			_, _, res = collide.collision_detection([0,0,0],pos,[1,0,0,0],q)
			# add data to arrays to save later
			X.append(numpy.concatenate((pos,q)))
			Y.append(int(res))
		# save data
		label = f'rel-in-split-{split}-Ntot-{Ntot:.1e}'
		if not os.path.isdir(f'../../data/datasets/{label}/'):
			os.mkdir(f'../../data/datasets/{label}/')
		numpy.savetxt(f'../../data/datasets/{label}/X{dataset}-{label}.dat',X)
		numpy.savetxt(f'../../data/datasets/{label}/Y{dataset}-{label}.dat',Y,fmt='%i')

if __name__ == "__main__":
	gen()

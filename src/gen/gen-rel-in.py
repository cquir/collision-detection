import scipy.spatial
import subprocess
import collide
import numpy
import click
import os

# generate random uniform quaternion
def uniform_quaternion():
    [u,v,w] = numpy.random.uniform(0,1,3)
    a = lambda u,v: numpy.sqrt(1-u)*numpy.sin(2*numpy.pi*v)
    b = lambda u,v: numpy.sqrt(1-u)*numpy.cos(2*numpy.pi*v)
    c = lambda u,w: numpy.sqrt(u)*numpy.sin(2*numpy.pi*w)
    d = lambda u,w: numpy.sqrt(u)*numpy.cos(2*numpy.pi*w)
    return [a(u,v),b(u,v),c(u,w),d(u,w)]

# generate example comprising of random position + quaternion
def example():
    hl = (numpy.sqrt(3)+1)/2.
    pos = numpy.random.uniform(-hl,hl,3)
    q = uniform_quaternion()
    return pos,q

@click.command()
@click.option('--Ntot','Ntot',default=1e6,help='Number of examples')
def gen(Ntot):

    if not os.path.isdir('../../data/datasets/'):
        os.mkdir('../../data/datasets/')

    numpy.random.seed(0)

    for label,perc in zip(['train-rel-in','validation-rel-in','test-rel-in'],[0.64, 0.16,0.2]):

        N = int(Ntot*perc)

        X = []; Y = []

        for i in range(int(N)):
            # generate example
            pos,q = example()
            
            # check if there is a collision 
            _, _, res = collide.collision_detection([0,0,0],pos,[1,0,0,0],q)

            # add data to arrays to save later
            X.append(numpy.concatenate((pos,q)))
            Y.append(int(res))

        # rescale inputs
        numpy.savetxt(f'../../data/datasets/X{label}-{Ntot:.1e}-mean.dat',numpy.mean(X,axis=0))
        numpy.savetxt(f'../../data/datasets/X{label}-{Ntot:.1e}-std.dat',numpy.std(X,axis=0))
        X = (X-numpy.mean(X,axis=0))/numpy.std(X,axis=0)

        # save data
        numpy.savetxt(f'../../data/datasets/X{label}-{Ntot:.1e}.dat',X)
        numpy.savetxt(f'../../data/datasets/Y{label}-{Ntot:.1e}.dat',Y,fmt='%i')

if __name__ == "__main__":
    gen()

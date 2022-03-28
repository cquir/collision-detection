import scipy.spatial
import subprocess
import collide
import numpy
import os

numpy.random.seed(0)

# convert 3 random uniform numbers between 0 and 1 (u,v,w) to a random uniform quaternion
a = lambda u,v: numpy.sqrt(1-u)*numpy.sin(2*numpy.pi*v)
b = lambda u,v: numpy.sqrt(1-u)*numpy.cos(2*numpy.pi*v)
c = lambda u,w: numpy.sqrt(u)*numpy.sin(2*numpy.pi*w)
d = lambda u,w: numpy.sqrt(u)*numpy.cos(2*numpy.pi*w)

# generate random uniform quaternion
def uniform_quaternion():
    [u,v,w] = numpy.random.uniform(0,1,3)
    return [a(u,v),b(u,v),c(u,w),d(u,w)]

# generate example comprising of random position + quaternion for each cube
def example():
    pos0 = numpy.random.uniform(0,numpy.sqrt(3),3)
    pos1 = numpy.random.uniform(0,numpy.sqrt(3),3)
    q0 = uniform_quaternion()
    q1 = uniform_quaternion()
    return pos0,pos1,q0,q1

# generate data
def data(N,label):

    Xtoy = []; X = []; Y = []

    for i in range(int(N)):
        # generate example
        pos0,pos1,q0,q1 = example()
        
        # check if there is a collision 
        cs, res = collide.collision_detection(pos0,pos1,q0,q1)

        if label == 'toy':
            Xtoy.append(cs[0][0]-pos0) # corner position w/o translation

        # add data to arrays to save later
        X.append(numpy.concatenate((pos0,q0,pos1,q1)))
        Y.append(int(res))

    # rescale inputs 
    numpy.savetxt(f'data/datasets/X{label}_mean.dat',numpy.mean(X,axis=0))
    numpy.savetxt(f'data/datasets/X{label}_std.dat',numpy.std(X,axis=0))
    X = (X-numpy.mean(X,axis=0))/numpy.std(X,axis=0)

    # save data
    if label == 'toy':
        numpy.savetxt(f'data/datasets/X{label}.dat',Xtoy)
    else:
        numpy.savetxt(f'data/datasets/X{label}.dat',X)
        numpy.savetxt(f'data/datasets/Y{label}.dat',Y,fmt='%i')

#data(100,'toy')
data(6.4*1e5,'train')
#data(1.6*1e5,'validation')
#data(2e5,'test')
#data(2*64,'tiny_train')

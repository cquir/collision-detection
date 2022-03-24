import scipy.spatial
import collide
import numpy

# functions to convert 3 random uniform between 0 and 1 (u,v,w) to a uniform quaternion
a = lambda u,v: numpy.sqrt(1-u)*numpy.sin(2*numpy.pi*v)
b = lambda u,v: numpy.sqrt(1-u)*numpy.cos(2*numpy.pi*v)
c = lambda u,w: numpy.sqrt(u)*numpy.sin(2*numpy.pi*w)
d = lambda u,w: numpy.sqrt(u)*numpy.cos(2*numpy.pi*w)

def gen_data(N,label):

    Xtoy = []; Xlow = []; Xhigh = []; Y = []

    for i in range(int(N)):
        # random position + quaternion vectors for each cube
        pos0 = numpy.random.uniform(0,numpy.sqrt(3),3)
        pos1 = numpy.random.uniform(0,numpy.sqrt(3),3)
        [u,v,w] = numpy.random.uniform(0,1,3)
        q0 = [a(u,v),b(u,v),c(u,w),d(u,w)]
        [u,v,w] = numpy.random.uniform(0,1,3)
        q1 = [a(u,v),b(u,v),c(u,w),d(u,w)]
        
        # check if there is a collision 
        cs, res = collide.collision_detection(pos0,pos1,q0,q1)

        if label == 'toy':
            Xtoy.append(cs[0][0]-pos0)

        # add results to arrays to save later
        Xlow.append(numpy.concatenate((pos0,q0,pos1,q1)))
        Xhigh.append(cs.flatten())
        Y.append(int(res))

    # rescale inputs 
    Xlow = (Xlow-numpy.mean(Xlow,axis=0))/numpy.std(Xlow,axis=0)
    Xhigh = (Xhigh-numpy.mean(Xhigh,axis=0))/numpy.std(Xhigh,axis=0)

    # save data
    if label == 'toy':
        numpy.savetxt(f'data/datasets/X{label}.dat',Xtoy)
    subdir = 'low-level-features'
    numpy.savetxt(f'data/datasets/{subdir}/X{label}.dat',Xlow)
    numpy.savetxt(f'data/datasets/{subdir}/Y{label}.dat',Y,fmt='%i')
    subdir = 'high-level-features'
    numpy.savetxt(f'data/datasets/{subdir}/X{label}.dat',Xhigh)
    numpy.savetxt(f'data/datasets/{subdir}/Y{label}.dat',Y,fmt='%i')

#gen_data(100,'toy')
gen_data(6.4*1e5,'train')
gen_data(1.6*1e5,'validation')
gen_data(2e5,'test')

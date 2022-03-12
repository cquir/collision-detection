import collide
import numpy

def gen_data(N,label):

    X = []; Y = []

    for i in range(int(N)):
        # random position + rotation vectors for each cube
        pos0vec = numpy.sqrt(3)*numpy.random.random(3)
        pos1vec = numpy.sqrt(3)*numpy.random.random(3)
        r0vec = 2*numpy.pi*numpy.random.random(3)
        r1vec = 2*numpy.pi*numpy.random.random(3)

        # check if there is a collision 
        res, cs = collide.collision_detection(pos0vec,pos1vec,r0vec,r1vec)
        
        # add results to arrays to save later
        X.append(numpy.concatenate([pos0vec,pos1vec,r0vec,r1vec]))
        Y.append(int(res))

    # save data
    numpy.savetxt('data/X%s.dat'%label,X)
    numpy.savetxt('data/Y%s.dat'%label,Y,fmt='%i')

gen_data(9e5,'train')
gen_data(1e5,'test')

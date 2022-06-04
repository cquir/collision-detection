import scipy.spatial
import numpy

def collision_detection(pos0,pos1,q0,q1):
    # rotation vector + position of center of cubes
    r0 = scipy.spatial.transform.Rotation.from_quat(q0)
    r1 = scipy.spatial.transform.Rotation.from_quat(q1)

    # corners + normal vectors of cubes (prior to rotation + translation)
    r = 1.; j = numpy.arange(8)
    c = (r/2)*numpy.array([(-1)**(j//2+1),(-1)**(j//4+1),(-1)**((j+1)//2+1)]).T
    cs = numpy.array([c,c])
    n = numpy.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    ns = numpy.array([n,n])

    # rotate then translate corners of cubes
    cs[0] = r0.apply(cs[0])+pos0
    cs[1] = r1.apply(cs[1])+pos1

    # rotate normal vectors of cubes
    ns[0] = r0.apply(ns[0])
    ns[1] = r1.apply(ns[1])

    # calculate normal vectors of the planes defined by the edges of the two cubes
    x = numpy.concatenate([ns[0],ns[0],ns[0]])
    y = numpy.tile(ns[1],3).reshape(9,3)
    ns = numpy.append(ns,numpy.cross(x,y)).reshape(-1,3)

    # determine if there is a collision along each normal vector
    collides = []; prjs_arr = []
    for n in ns:
        prjs = numpy.dot(numpy.concatenate(cs),n)
        prjs_arr.append(prjs)
        pmin = min(prjs[:8]); pmax = max(prjs[:8])
        collides.append(len(prjs[8:][(prjs[8:] >= pmin) & (prjs[8:] <= pmax)]) > 0)

    # collision if collision along all normal vectors
    return cs, prjs_arr, all(collides)

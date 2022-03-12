from scipy.spatial.transform import Rotation as R
import numpy

# n is unit vector so no need to divide by its magnitude
scalar_proj = lambda v,n: v[0]*n[0]+v[1]*n[1]+v[2]*n[2]

def collision_detection(r0vec,r1vec,pos1vec,pos2vec):
    # rotation vector + position of center of cubes
    r0 = R.from_rotvec(r0vec)
    r1 = R.from_rotvec(r1vec)
    pos = numpy.array([pos1vec,pos2vec])

    # corners + normal vectors of cubes (prior to rotation + translation)
    r = 1.; j = numpy.arange(8)
    c = (r/2)*numpy.array([(-1)**(j//2+1),(-1)**(j//4+1),(-1)**((j+1)//2+1)]).T
    cs = numpy.array([c,c])
    n = numpy.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    ns = numpy.array([n,n])

    # rotate then translate corners of cubes
    cs[0] = r0.apply(cs[0])+pos[0]
    cs[1] = r1.apply(cs[1])+pos[1]

    # rotate normal vectors of cubes
    ns[0] = r0.apply(ns[0])
    ns[1] = r1.apply(ns[1])

    # determine if there is a collision along each normal vector
    collides = []
    for i,n in enumerate(numpy.concatenate(ns)):
        prjs = numpy.array([scalar_proj(c,n) for c in numpy.concatenate(cs)])
        prjs = prjs.reshape(2,8)
        pmin = min(prjs[0]); pmax = max(prjs[0])
        collides.append(sum([pmin <= p <= pmax for p in prjs[1]]) > 0)

    # collision if collision along all normal vectors
    return all(collides), cs

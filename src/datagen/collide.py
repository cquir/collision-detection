import scipy.spatial
import numpy as np

def collision_detection(x : np.ndarray):

    pos0 = x[0:3]
    pos1 = x[3:6]
    q0 = x[6:10]
    q1 = x[10:]

    # rotation vector + position of center of cubes
    r0 = scipy.spatial.transform.Rotation.from_quat(q0)
    r1 = scipy.spatial.transform.Rotation.from_quat(q1)

    # corners + normal vectors of cubes (prior to rotation + translation)
    r = 1.; j = np.arange(8)
    c = (r/2)*np.array([(-1)**(j//2+1),(-1)**(j//4+1),(-1)**((j+1)//2+1)]).T
    cs = np.array([c,c])
    n = np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    ns = np.array([n,n])

    # rotate then translate corners of cubes
    cs[0] = r0.apply(cs[0])+pos0
    cs[1] = r1.apply(cs[1])+pos1

    # rotate normal vectors of cubes
    ns[0] = r0.apply(ns[0])
    ns[1] = r1.apply(ns[1])

    # determine if there is a collision along each normal vector
    collides = []
    for n in np.concatenate(ns):
        prjs = np.dot(np.concatenate(cs),n)
        pmin = min(prjs[:8]); pmax = max(prjs[:8])
        collides.append(len(prjs[8:][(prjs[8:] >= pmin) & (prjs[8:] <= pmax)]) > 0)

    # collision if collision along all normal vectors
    return cs, np.array([int(all(collides))], dtype=np.float32)

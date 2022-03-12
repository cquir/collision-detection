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

r0vec = 2*numpy.pi*numpy.random.random(3)
r1vec = 2*numpy.pi*numpy.random.random(3)
pos1vec = numpy.sqrt(3)*numpy.random.random(3)
pos2vec = numpy.sqrt(3)*numpy.random.random(3)

collide, cs = collision_detection(r0vec,r1vec,pos1vec,pos2vec)

# testing 
if True:
    import phymcmc.plot

    x0,y0,z0 = cs[0].T
    x1,y1,z1 = cs[1].T

    pid = -1
    gridfig = phymcmc.plot.grid_plot((1,3),wspace=0.4,rwidth=3.9)
    s = 1

    def fix_ax(ax,xlab,ylab):
        hl = numpy.sqrt(3)/2
        ax.set_xlim(-hl,3*hl)
        ax.set_ylim(-hl,3*hl)
        ax.set_xticks(numpy.arange(-1,3))
        ax.set_yticks(numpy.arange(-1,3))
        ax.set_xlabel(r'$%s$'%xlab)
        ax.set_ylabel(r'$%s$'%ylab)

    x0s = [x0,x0,y0]
    y0s = [y0,z0,z0]
    x1s = [x1,x1,y1]
    y1s = [y1,z1,z1]
    xlabs = ['x','x','y']
    ylabs = ['y','z','z']
    xidxs = [numpy.arange(4),numpy.arange(4,8),numpy.arange(4)]
    yidxs = [numpy.roll(numpy.arange(4),1),numpy.roll(numpy.arange(4,8),1),numpy.arange(4,8)]

    for x0,y0,x1,y1,xlab,ylab in zip(x0s,y0s,x1s,y1s,xlabs,ylabs):
        pid += 1; ax = gridfig.subaxes(pid)
        for x,y,col in zip([x0,x1],[y0,y1],['tab:blue','tab:orange']):
            for xidx,yidx in zip(xidxs,yidxs):
                for i,j in zip(xidx,yidx):
                    ax.plot(x[[i,j]],y[[i,j]],color=col)
        fix_ax(ax,xlab,ylab)
        if pid == 1:
            ax.set_title(r'collide = %s'%collide)

    gridfig.fig.savefig('test.pdf',bbox_inches='tight')




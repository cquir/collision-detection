import phymcmc.plot
import collide
import numpy

# random initialization
pos0vec = numpy.sqrt(3)*numpy.random.random(3)
pos1vec = numpy.sqrt(3)*numpy.random.random(3)
r0vec = 2*numpy.pi*numpy.random.random(3)
r1vec = 2*numpy.pi*numpy.random.random(3)

# check if there is a collision + get corner position of cubes
cs, res = collide.collision_detection(pos0vec,pos1vec,r0vec,r1vec)

# define graph
pid = -1
gridfig = phymcmc.plot.grid_plot((1,3),wspace=0.4,rwidth=3.9)

# fix graph
def fix_ax(ax,xlab,ylab):
    hl = numpy.sqrt(3)/2
    ax.set_xlim(-hl,3*hl)
    ax.set_ylim(-hl,3*hl)
    ax.set_xticks(numpy.arange(-1,3))
    ax.set_yticks(numpy.arange(-1,3))
    ax.set_xlabel(r'$%s$'%xlab)
    ax.set_ylabel(r'$%s$'%ylab)

# define some more stuff
x0,y0,z0 = cs[0].T
x1,y1,z1 = cs[1].T
x0s = [x0,x0,y0]
y0s = [y0,z0,z0]
x1s = [x1,x1,y1]
y1s = [y1,z1,z1]
xlabs = ['x','x','y']
ylabs = ['y','z','z']
xidxs = [numpy.arange(4),numpy.arange(4,8),numpy.arange(4)]
yidxs = [numpy.roll(numpy.arange(4),1),numpy.roll(numpy.arange(4,8),1),numpy.arange(4,8)]

# plotting
for x0,y0,x1,y1,xlab,ylab in zip(x0s,y0s,x1s,y1s,xlabs,ylabs):
    pid += 1; ax = gridfig.subaxes(pid)
    for x,y,col in zip([x0,x1],[y0,y1],['tab:blue','tab:orange']):
        for xidx,yidx in zip(xidxs,yidxs):
            for i,j in zip(xidx,yidx):
                ax.plot(x[[i,j]],y[[i,j]],color=col)
    fix_ax(ax,xlab,ylab)
    if pid == 1:
        ax.set_title(r'collide = %s'%res)

# save graph
gridfig.fig.savefig('test.pdf',bbox_inches='tight')

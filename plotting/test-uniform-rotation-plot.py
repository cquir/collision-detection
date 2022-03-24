import phymcmc.plot
import numpy

# define graph
pid = -1
gridfig = phymcmc.plot.grid_plot((1,3),wspace=0.4,rwidth=3.9)

# define some more stuff
bdir = '/home/cquir/Documents/collision-detection/data/datasets'
x,y,z = numpy.loadtxt(f'{bdir}/Xtoy.dat').T
xs = [x,x,y]
ys = [y,z,z]
xlabs = ['x','x','y']
ylabs = ['y','z','z']

# fix graph
def fix_ax(ax,xlab,ylab):
    ax.set_xlabel(r'$%s$'%xlab)
    ax.set_ylabel(r'$%s$'%ylab)

# plotting
for x,y,xlab,ylab in zip(xs,ys,xlabs,ylabs):
    pid += 1; ax = gridfig.subaxes(pid)
    ax.scatter(x,y,s=0.1)
    fix_ax(ax,xlab,ylab)

# save graph
gridfig.fig.savefig('graphs/test-uniform-rotation.pdf',bbox_inches='tight')

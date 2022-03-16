import phymcmc.plot
import numpy

# define graph
pid = -1
gridfig = phymcmc.plot.grid_plot((5,5),wspace=0.4,rwidth=3.9)

# define some stuff
lrs = numpy.tile([1e-1,1e-2,1e-3,1e-4,1e-5],5)
hs = numpy.repeat([16,32,64,128,256],5)
res = []

for lr,h in zip(lrs,hs):
    pid += 1; ax = gridfig.subaxes(pid)
    # load data
    s =  'lr_{:.1e}_h_{}'.format(lr,h)
    train_loss = numpy.loadtxt('data/train_loss_{}.dat'.format(s))
    val_loss = numpy.loadtxt('data/val_loss_{}.dat'.format(s))

    # plotting
    ax.plot(numpy.arange(len(train_loss)),train_loss)
    ax.plot(numpy.arange(len(val_loss)),val_loss)
    
    
    # store best accuracy on validation dataset
    val_accuracy = numpy.loadtxt('data/val_accuracy_{}.dat'.format(s))
    res.append(100*(1-round(max(val_accuracy),3)))

    # fix graph
    if pid < 5:
        ax.set_title('lr = {:.1e}'.format(lr))
    if pid % 5 == 0:
        ax.text(-0.5,0.5,f'h = {h}',transform=ax.transAxes)
    ax.set_ylim(0.0,0.7)

# print best accuracy on validation dataset
print(numpy.array(res).reshape(5,5))

# save graph
gridfig.fig.savefig('grid-search.pdf',bbox_inches='tight')

import phymcmc.plot
import numpy
import os

# define graph
pid = -1
gridfig = phymcmc.plot.grid_plot((10,5),wspace=0.4,rwidth=3.9)

# get min val loss for each set of hyperparameters to sort
roots = []; files_arr = []; min_val_loss = []
for root, dirs, files in os.walk('../data/'):
    if root == '../data/':
        continue
    roots.append(root)
    files_arr.append(files)
    f = list(filter(lambda f: 'val_loss' in f, files))[0]
    val_loss = numpy.loadtxt(root+'/'+f)
    val_loss = val_loss[val_loss > 0]
    min_val_loss.append(min(val_loss))

for min_val_loss, root, files in sorted(zip(min_val_loss,roots,files_arr)):
    pid += 1; ax = gridfig.subaxes(pid)
    # load plotting data
    f = list(filter(lambda f: 'train_loss' in f, files))[0]
    train_loss = numpy.loadtxt(root+'/'+f)
    train_loss = train_loss[train_loss > 0]
    f = list(filter(lambda f: 'val_loss' in f, files))[0]
    val_loss = numpy.loadtxt(root+'/'+f)
    val_loss = val_loss[val_loss > 0]

    # plotting
    ax.plot(numpy.arange(len(train_loss)),train_loss)
    ax.plot(numpy.arange(len(val_loss)),val_loss)

    #fix graph
    [_,h,_,lr,_,momentum,_,_,batch_size] = root.split('/')[-1].split('_')
    s = f'h = {h},\nlr = {lr},\nmomentum = {momentum},\nbatch size = {batch_size}'
    ax.text(0.1,0.9,s,ha='left',va='top',transform=ax.transAxes,size='small')
    if pid // 5 == len(roots) // 5:
        ax.set_xlabel('Epochs')
    if pid % 5 == 0:
        ax.set_ylabel('Loss')
    ax.set_ylim(0,1.1)

# save graph
gridfig.fig.savefig('graphs/rand-grid-search.pdf',bbox_inches='tight')

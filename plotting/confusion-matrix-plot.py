import matplotlib.cm
import phymcmc.plot
import numpy
import torch
import load
import sys
import nn
import os

# load NN with set of hyperparameters with min val loss
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
idx = numpy.argmin(min_val_loss)
root = roots[idx]; files = files_arr[idx]
f = list(filter(lambda f: 'model' in f, files))[0]
h = root.split('/')[-1].split('_')[1]
model = nn.NeuralNetwork(int(h)) 
model.load_state_dict(torch.load(root+'/'+f))

save = bool(sys.argv[1])

if save:
    # evaluate the NN's performance
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    val_data = load.Dataset('validation')
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=1000,shuffle=False)
    preds = []; labels = []
    for data,label in val_loader:
        output = model(data)
        # sigmoid layer normally included in criterion
        preds.append(torch.round(torch.sigmoid(output)).detach().numpy())
    preds = numpy.array(preds).flatten()
    labels = val_data.data[1]; p = sum(labels); n = len(labels)-p
    tp = sum([(x == y and x == 1) for x,y in zip(preds,labels)])
    fn = sum([(x != y and x == 0) for x,y in zip(preds,labels)])
    fp = sum([(x != y and x == 1) for x,y in zip(preds,labels)])
    tn = sum([(x == y and x == 0) for x,y in zip(preds,labels)])
    res = numpy.array([[tn,fp],[fn,tp],[tn/n,fp/n],[fn/p,tp/p]])
    numpy.savetxt(root+'/confusion_matrix_'+root.split('/')[-1]+'.dat',res)
else:
    res = numpy.loadtxt(root+'/confusion_matrix_'+root.split('/')[-1]+'.dat')

# define graph 
pid = -1
gridfig = phymcmc.plot.grid_plot((1,2))

# define some more stuff
cmap = matplotlib.cm.magma_r

# fix graph
def fix_ax(ax):
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual')

# plotting 
pid += 1; ax = gridfig.subaxes(pid)
ax.matshow(res[:2],cmap=cmap)
for i,val in enumerate(numpy.concatenate(res[:2])):
    col = 'w' if i % 2 == i // 2 else 'k'
    ax.text(i%2,i//2,str(int(val)),color=col,va='center',ha='center')
fix_ax(ax)
pid += 1; ax = gridfig.subaxes(pid)
ax.matshow(res[-2:],cmap=cmap)
for i,val in enumerate(numpy.concatenate(res[-2:])):
    col = 'w' if i % 2 == i // 2 else 'k'
    ax.text(i%2,i//2,str(round(100*val,1))+'\%',color=col,va='center',ha='center')
fix_ax(ax)

# save graph
gridfig.fig.savefig('graphs/confusion-matrix.pdf',bbox_inches='tight')


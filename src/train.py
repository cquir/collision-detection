import torch
import wandb
import nn

def evaluate_model(args,train_loader,val_loader):

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# initialization
	model = nn.NeuralNetwork(args['i'],args['h'],args['hidden_layers'],args['dropout'],args['pdrop']).to(device)
	wandb.watch(model,log="all")
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'])
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=args['lr'],
		steps_per_epoch=len(train_loader),
		epochs=args['epochs'],
		pct_start=args['pct_start'],
		max_momentum=args['max_momentum'])
	epoch = 1; best_score = None; counter = 0; early_stop = False

	# optionally resume from a checkpoint
	if args['resume']:
		checkpoint = torch.load(f'../data/results/{args["name"]}/model-{args["name"]}.pt')
		epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		if args['early_stopping']:
			best_score = checkpoint['best_score']
			counter = checkpoint['counter']

	# train the neural network 
	def train():
		model.train()  
		for batch_idx, (data,label) in enumerate(train_loader):
			data, label = data.to(device), label.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,label.unsqueeze(1))
			loss.backward()
			optimizer.step()
			scheduler.step()
			
	# evaluate the neural network's performance
	def test(test_loader,dataset_label,batch_size):
		model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data,label in test_loader:
				data, label = data.to(device), label.to(device)
				output = model(data)
				test_loss += criterion(output,label.unsqueeze(1)).item()
				# sigmoid layer normally included in criterion
				pred = torch.round(torch.sigmoid(output))
				correct += (pred.squeeze(1) == label).sum().item()
		test_loss *= batch_size/len(test_loader.dataset) 
		return test_loss, correct/len(test_loader.dataset)

	# main loop
	while epoch < args['epochs']+1:
		train()
		train_loss, train_accuracy = test(train_loader,'Training',args['batch_size'])
		val_loss, val_accuracy = test(val_loader,'Validation',args['test_batch_size'])
		# log results
		wandb.log({
			'Train loss':train_loss,
			'Train accuracy':train_accuracy,
			'Validation loss':val_loss,
			'Validation accuracy':val_accuracy
		})
		# save results
		if args['early_stopping']:
			if best_score is None:
				best_score = val_loss
			elif val_loss >= best_score:
				counter += 1
				if counter >= args['patience']:
					early_stop = True
			else:
				best_score = val_loss
				args['best_val_accuracy'] = val_accuracy
				torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'best_score': best_score,
					'counter': counter,
					}, f'../data/results/{args["name"]}/model-{args["name"]}-best.pt')
				counter = 0
			if early_stop:
				break
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'best_score': best_score,
			'counter': counter,
			}, f'../data/results/{args["name"]}/model-{args["name"]}.pt')
		# abort if training loss is above threshold
		if train_loss >= args['threshold']:
			break
		epoch += 1

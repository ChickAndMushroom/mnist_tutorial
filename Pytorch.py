import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10
learning_rate = 1e-2

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

#define model
class SimpleNet(nn.Module):
	def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
		super(SimpleNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Linear(in_dim, n_hidden_1),
			nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
		self.layer2 = nn.Sequential(
			nn.Linear(n_hidden_1, n_hidden_2),
			nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
		self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return x
model=SimpleNet(28*28,300,100,10)
#define loss function and optimiter
criterion =nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

# train and evaluate
for epoch in range(NUM_EPOCHS):
	running_loss=0.0
	for images, labels in tqdm(train_loader):
		#forward + backward + optimize
		images=images.view(images.size(0),-1)
		images=Variable(images)
		labels=Variable(labels)
		out=model(images)
		loss=criterion(out,labels)
		print_loss=loss.data.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		epoch+=1
		if epoch%50==0:
			print('epoch:{},loss:{:.4f}'.format(epoch,loss.data.item()))

# evaluate
# for test_dataset
model.eval()
eval_loss=0
eval_acc=0
for images, labels in tqdm(test_loader):
	images=images.view(images.size(0),-1)
	images=Variable(images,volatile=True)
	labels=Variable(labels,volatile=True)
	out=model(images)
	loss=criterion(out,labels)
	eval_loss+=loss.item()*labels.size(0)
	_,pred=torch.max(out,1)
	num_correct=(pred==labels).sum()
	eval_acc+=num_correct.item()
	
test_accuracy=eval_acc/(len(test_dataset))

# for train_dataset
eval_loss=0
eval_acc=0
for images, labels in tqdm(train_loader):
	images=images.view(images.size(0),-1)
	images=Variable(images,volatile=True)
	labels=Variable(labels,volatile=True)
	out=model(images)
	loss=criterion(out,labels)
	eval_loss+=loss.item()*labels.size(0)
	_,pred=torch.max(out,1)
	num_correct=(pred==labels).sum()
	eval_acc+=num_correct.item()
	
training_accuracy=eval_acc/(len(train_dataset))

#calculate the accuracy using traning and testing dataset
print('Training accuracy: %0.2f%%' % (training_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
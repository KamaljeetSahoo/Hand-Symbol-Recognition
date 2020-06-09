# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 03:49:02 2020

@author: Kamaljeet
"""

from Dataset import load_split_train_test
from Model import Net

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter


#========Loading Datasets=============

datadir = "alphabet_train"

train_loader, test_loader = load_split_train_test(datadir, .2)
print(train_loader.dataset.classes)

classes = train_loader.dataset.classes


#==========Displaying images from trainloader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#=============Defining Training Parameters and type of Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(out_fea = len(classes))
model = model.train()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


step = 0
loss_train = []
loss_val = []

min_loss = 100
patience = 5
training_loss_store = []
validation_loss_store = []

writer = SummaryWriter('writer')

file = open('logs_test4_epoch40_with_max_pool.txt', 'w')
print('training started.............................................')
file.write('training started.............................................\n')
start_training_time = time.time()
for epoch in range(60):  # loop over the dataset multiple times
    epoch_start = time.time()
    file.write('##############################TRAINING###############################\n')
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        step+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_train.append(loss.item())
        training_loss_store.append([epoch, loss.item()])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10))
            #log_loss_summary(logger, loss_train, step, prefix = 'train_')
            file.write('epoch = '+ str(epoch + 1) + '\t' +'step = '+ str(step) +'\t'+'train_loss = '+'\t'+str(np.mean(loss_train)) +'\n')
            writer.add_scalar('Loss/train', np.mean(loss_train), step)
            loss_train = []
            running_loss = 0.0
                
    print('Finished training for epoch ' + str(epoch) + ' time taken = ' + str(time.time() - epoch_start))
    file.write('Finished training for epoch ' + str(epoch) + ' time taken = ' + str(time.time() - epoch_start) + '\n')
    file.write('##################################evaluation##############################\n')
    print('################################evaluation###########################\n')
    with torch.no_grad():
        val_loss = 0
        model.eval()
        
        for i, data in enumerate(test_loader, 0):
            step+=1
            inputs, labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_val.append(loss.item())
            validation_loss_store.append([epoch, loss.item()])
            val_loss += loss
            
        val_loss = val_loss/float(i + 1)
        
        if val_loss < min_loss:
            min_loss = val_loss
            no_impr_epoch = 0
            
            #save the best model
            torch.save(model.state_dict(), 'weight/' + 'epoch_' + str(epoch+1) + 'loss_' + str(float(val_loss.cpu().numpy())) + '.pt')
            
            print('performance improved with validation loss ' + str(float(val_loss.cpu().numpy())))
            file.write('--------------------------------------------------------------------\n')
            file.write('performance improved with validation loss =  ' + str(float(val_loss.cpu().numpy())) + '\n')
            
            file.write('epoch = '+ str(epoch + 1) + '\t' +'step = '+ str(step) +'\t'+'val_loss = '+'\t'+str(np.mean(loss_val)) +'\n')
            file.write('--------------------------------------------------------------------\n\n')
            #log_loss_summary(logger, loss_val, step, prefix="val_")
            writer.add_scalar('Loss/val',val_loss, step)
            loss_val = []
        else:
            no_impr_epoch += 1
            print('no improvement with prev best model ' + str(no_impr_epoch) + 'th')
            file.write('no improvement with prev best model ' + str(no_impr_epoch) + 'th \n')
            
        if no_impr_epoch > patience:
            print('stop training')
            file.write('stop training')
            break
    writer.add_scalar('Memory Allocation', torch.cuda.memory_allocated(), step)
    
print('Finished Training................................................')
file.write('Finished Training................................................\n')
end_training_time = time.time()
file.write('Training time:- ' + str(end_training_time - start_training_time))
file.close()
writer.close()

if(torch.cuda.is_available()):
    torch.cuda.empty_cache()


PATH = 'weight/epoch_3loss_0.2937310039997101.pt'
model = Net(out_fea=len(classes))
model.load_state_dict(torch.load(PATH))
model.eval()

dataiter = iter(train_loader)
images, labels = dataiter.next()


# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print('Ground Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
i = outputs.detach().numpy()
#print('Scores:-', i)
for j in range(len(classes)):
    print(classes[j], ':- ', i[0, j])


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        test_start = time.time()
        images, labels = data[0], data[1]
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_end = time.time()

print('Accuracy of the network on the 3776 test images: %d %%' % (100 * correct / total))
print('Total time for completion of test_loader: ', -(test_start-test_end))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0], data[1]
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for i in range(4):#Here goes batch size
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

#file = open('result_logs.txt', 'w')
for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    #file.write('Accuracy of' + classes[i] + ':' + '\t' + str(100 * class_correct[i] / class_total[i]) +'%'+'\n')
    #writer.add_scalar('Class_accuracy', 100 * (class_correct[i] / class_total[i]), i)
#file.close()
writer.close()













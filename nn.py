import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

PATH = './data/nn'
BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

activations_dict = {}
grad_dict = {}
def backward_hook(layer_name, grad_input, grad_output): 
    if layer_name in grad_dict:
        grad_dict[layer_name]["grad_input"].append(grad_input)
        grad_dict[layer_name]["grad_output"].append(grad_output)
    else :
        grad_dict[layer_name] = {}
        grad_dict[layer_name]["grad_input"] = []
        grad_dict[layer_name]["grad_output"] = []

def forward_hook(layer_name, input, output):
    if layer_name not in activations_dict:
        activations_dict[layer_name] = {}
        activations_dict[layer_name]["input"] = []
        activations_dict[layer_name]["output"] = []
    activations_dict[layer_name]["input"].append(input)
    activations_dict[layer_name]["output"].append(output)
    
class Net(nn.Module):
    def __init__(self, dataset='CIFAR10'):
        super(Net, self).__init__()
        if 'CIFAR10' == dataset:
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        elif 'MNIST' == dataset:
            return
        else:
            print('Invalid dataset')
        self.dataset = dataset
        self.load_dataset(dataset)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def load_dataset(self, dataset='CIFAR10', root='./data', download=True):
        if dataset == 'CIFAR10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=download, transform=transform)
        
            self.testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=download, transform=transform)
            self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif dataset == 'MNIST':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.trainset = torchvision.datasets.MNIST(root=root, train=True,
                                                download=download, transform=transform)
        
            self.testset = torchvision.datasets.MNIST(root=root, train=False,
                                                download=download, transform=transform)
            self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        else:
            return
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0)
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()
        #print('load =====')
        #imshow(torchvision.utils.make_grid(images))
        #print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
        #print('=====')
                                            
    def train(self, epochs=2, criterion=nn.CrossEntropyLoss(), lr=0.001, momentum=0.9):
        self.criterion = criterion
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, start=0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward() #critical step
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        torch.save(self, PATH + '-' + self.dataset)
        print('Finished Training')
        
    def accuracy(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    
    def accuracies(self):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(BATCH_SIZE):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))
                
                
    def test(self, epochs=2, criterion=nn.CrossEntropyLoss(), lr=0.001, momentum=0.0):
        hfc2 = self.fc2.weight.register_hook(lambda grad: self.fc2.weight)
        
        self.criterion = criterion
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        with torch.no_grad():
            for data in self.testloader:
                with torch.enable_grad():
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    images.requires_grad_()
                print(images.requires_grad)
                
                self.optimizer.zero_grad()
                print(images.grad)
                print('::::::::::')
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
    def test2(self):
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            print('==========')
            print('Name: ',name)
            print(type(param))
            print(param.grad)
            
            # Don't update if this is not part of the appropriate layer.
            if not "fc2" in name:
                print('Skip')
                continue

            # Transform the parameter as required.
            #new_grad = F.relu(param)

            # Update the parameter.
            #print(state_dict[name].grad)
            if state_dict[name].grad:
                print('Hooray!')
            #    state_dict[name].grad.copy_(new_grad)
            
    def inceptionism(self, layer_name="fc2", NUM_ITER=1000):
        self.inceptionism_layer_name = layer_name
        if layer_name == "fc1":
            layer = self.fc1
        elif layer_name == "fc2":
            layer = self.fc2
        elif layer_name == "fc3":
            layer = self.fc3
        elif layer_name == "conv1":
            layer = self.conv1
        elif layer_name == "conv2":
            layer = self.conv2
        else:
            print("Invalid layer name")
            return
            
        forward_hook = self.record_forward_hook(layer_name)
        backward_hook = self.record_backward_hook(layer_name)
        self.forward_handle = layer.register_forward_hook(forward_hook)
        #self.backward_handle = layer.register_backward_hook(backward_hook)
        self.inceptionism_handle = layer.register_backward_hook(self.backward_to_forward)
        

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()
        softmax = nn.Softmax(1)
        with torch.enable_grad():   
            for i, data in enumerate(self.trainloader, start=0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images.float()
                print('start: ', i)
                beforeIm = images.cpu().detach()
                self.optimizer = get_input_optimizer(images) #self.optimizer = optim.SGD([images.requires_grad_()], lr=0.01, momentum=0.0)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                before = softmax(outputs)
                print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
                print('Predicted (before): ', ' '.join('%5s' % self.classes[predicted[j]]
                              for j in range(4)))
                for _ in range(NUM_ITER):
                    #def closure():
                    self.optimizer.zero_grad()
                    outputs = self(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward() #critical step
                    self.optimizer.step()
                afterIm = images.cpu().detach()
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                after = softmax(outputs)
                print('Predicted (after): ', ' '.join('%5s' % self.classes[predicted[j]]
                              for j in range(4)))
                imshow(torchvision.utils.make_grid(torch.cat((beforeIm, afterIm)), nrow=4))
                print('end: ', i)
        #print('=====')
        #print(grad_dict["fc2"]["grad_input"][0][0])
        #print(grad_dict["fc1"]["grad_output"][-1][0]) #[4, 120]
        #print(grad_dict["fc1"]["grad_input"][-1][0]) #[4, 120]
        #print(grad_dict["fc1"]["grad_input"][-1][1]) #[4, 16*5*5]
        print('=====')
        #print(grad_dict["fc3"]["grad_output"][-1][0])
        #print(grad_dict["fc3"]["grad_input"][-1][0])
        #print(grad_dict["fc3"]["grad_input"][-1][1])
        #for i in range(len(grad_dict["fc2"]["grad_input"])):
        #    print(grad_dict["fc2"]["grad_input"][i][1] == grad_dict["fc1"]["grad_input"][i][0])
        print('Inceptionism done.')
        
    def adversarial_noise(self, x=4):
        #forward_hook = self.record_forward_hook(layer_name)
        #backward_hook = self.record_backward_hook(layer_name)
        #self.forward_handle = layer.register_forward_hook(forward_hook)
        #self.backward_handle = layer.register_backward_hook(backward_hook)
        #self.inceptionism_handle = layer.register_backward_hook(self.backward_to_forward)
        

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()
        softmax = nn.Softmax(1) #1 = axis over which softmax operates
        print(self.classes[x])
        with torch.enable_grad():
            for i, data in enumerate(self.trainloader, start=0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                images.float()
                print('start: ', i)
                beforeIm = images.cpu().detach()
                self.optimizer = get_input_optimizer(images)  #self.optimizer = optim.SGD([images.requires_grad_()], lr=0.01, momentum=0.0)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                before = softmax(outputs)
                print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
                print('Before: ', ' '.join('%5s' % before[j][x].item()
                              for j in range(4)))
                print('Predicted (before): ', ' '.join('%5s' % self.classes[predicted[j]]
                              for j in range(4)))
                for _ in range(500):
                    #def closure():
                    self.optimizer.zero_grad()
                    outputs = self(images)
                    fake_labels = torch.tensor([x,x,x,x])
                    fake_labels = fake_labels.to(device)
                    loss = self.criterion(outputs, fake_labels) #(outputs, labels)
                    loss.backward() #critical step
                    self.optimizer.step()
                afterIm = images.cpu().detach()
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                after = softmax(outputs)
                print('Predicted (after): ', ' '.join('%5s' % self.classes[predicted[j]]
                              for j in range(4)))
                print('After: ', ' '.join('%5s' % after[j][x].item()
                              for j in range(4)))
                imshow(torchvision.utils.make_grid(torch.cat((beforeIm, afterIm)), nrow=4))
                print('end: ', i)
        #print('=====')
        #print(grad_dict["fc2"]["grad_input"][0][0])
        #print(grad_dict["fc1"]["grad_output"][-1][0]) #[4, 120]
        #print(grad_dict["fc1"]["grad_input"][-1][0]) #[4, 120]
        #print(grad_dict["fc1"]["grad_input"][-1][1]) #[4, 16*5*5]
        print('=====')
        #print(grad_dict["fc3"]["grad_output"][-1][0])
        #print(grad_dict["fc3"]["grad_input"][-1][0])
        #print(grad_dict["fc3"]["grad_input"][-1][1])
        #for i in range(len(grad_dict["fc2"]["grad_input"])):
        #    print(grad_dict["fc2"]["grad_input"][i][1] == grad_dict["fc1"]["grad_input"][i][0])
        print('Adversarial noise test done.')
        
    def backward_to_forward(self, _, grad_input, grad_output): #measurements for fc2
        #print('Begin backward hook')
        if False:
            #print('_')
            #print(type(_)) #linear
            
            print('grad_input')
            #print(len(grad_input)) #tuple length 3 of tensors
            print(grad_input[0].size()) #[4, 84] (use this?)
            print(grad_input[1].size()) #[4, 120]
            print(grad_input[2].size()) #[120, 84]
            
            print('grad_output')
            #print(len(grad_output)) #tuple length 1 of type tensor
            print(grad_output[0].size()) #[4, 84]
        forward = activations_dict[self.inceptionism_layer_name]["input"].pop()[0]
        #print(forward.shape)
        if self.inceptionism_layer_name in ["fc1", "fc2", "fc3"]:
            grad_input[1].copy_(forward)
        elif self.inceptionism_layer_name in ["conv1, conv2"]:
            grad_input[0].copy_(forward)
        #print('End backward hook')
        
    def record_backward_hook(self, name):
        def custom_backward_hook(self, module, grad_input, grad_output): #module is Linear, ignored
            backward_hook(name, grad_input, grad_output)
        return custom_backward_hook
    def record_forward_hook(self, name):
        def custom_forward_hook(self, input, output):
            forward_hook(name, input, output)
        return custom_forward_hook
            

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimgT = np.transpose(npimg, (1, 2, 0))
    #print(npimgT.shape) #32x32x3
    plt.imshow(npimgT)
    plt.show()

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.SGD([input_img.requires_grad_()], lr=0.01, momentum=0.1)
    return optimizer

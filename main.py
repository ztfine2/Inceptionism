import nn
import torch

TRAIN = True
def load_model(dataset):
    if not TRAIN:
        try:
            model = torch.load(nn.PATH + '-' + dataset)
            model.to(nn.device)
        except FileNotFoundError as e:
            print('File not found: ', e)
            model = nn.Net(dataset)
            model.to(nn.device)
            model.train()
    else:
        model = nn.Net(dataset)
        model.to(nn.device)
        model.train()
    return model
model = load_model('CIFAR10')

#model.inceptionism(layer_name="conv2", NUM_ITER=500)
model.adversarial_noise(x=4)


#model.test2()
def tensor_test():
    a = torch.arange(4.)
    b = torch.zeros(4)
    print(a, b)
    c = a.copy_(b)
    print(a, b, c)
    print(a is b, b is c, c is a)
#tensor_test()




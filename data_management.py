import torch
from torchvision import datasets, transforms, models
from PIL import Image


def load_data(path):
    print("Loading and preprocessing data from {} ...".format(path))
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms  = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])


    train_data  = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)

    
    print("Finished loading and preprocessing data.")
    
    return train_data, trainloader, validloader, testloader

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    test_transforms = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    return test_transforms(image)

    
    # TODO: Process a PIL image for use in a PyTorch model
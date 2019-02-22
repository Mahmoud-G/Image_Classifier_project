import os
import argparse
import torch
import time
import copy
from collections import OrderedDict
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch import nn, optim



def main():
    # declare the parser
    parser = argparse.ArgumentParser(description='Image Classifire')
    # working directory
    parser.add_argument('-p', '--work_path', help='Set the working directory', default='/home/workspace/aipnd-project')
    parser.add_argument('-f', '--image_folder', help='Set the image directory', default='flowers')
    parser.add_argument('-t', '--train_folder', help='Set the training directory', default='train')
    parser.add_argument('-v', '--valid_folder', help='Set the validatig directory', default='valid')
    parser.add_argument('-s', '--test_folder', help='Set the testing directory', default='test')
    # Model hyperparameters
    parser.add_argument('-m', '--model', help='Choose the Model architecture',
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet'], default='vgg16')
    parser.add_argument('-l', '--learning_r', type=float, help='Set the learning rate', default=0.003)
    parser.add_argument('-h1', '--h1', type=int, help='Set the headding layer 1', default=4000)
    parser.add_argument('-h2', '--h2', type=int, help='Set the headding layer 2', default=1000)
    parser.add_argument('-o', '--output_size', type=int, help='Set the output size', default=102)
    # training
    parser.add_argument('-ep', '--epochs', type=int, help='set the number of ephocs', default=10)
    parser.add_argument('-d', '--device', help='set the device of learning cpu or cuda', choices=['cpu', 'cuda'],
                        default='cuda')

    args = parser.parse_args()

    #     Run the method
    training(args)


def set_working_dir(path):
    os.chdir(path)


# TODO: Define your transforms for the training, validation, and testing sets
def data_transforms(args):
    data_dir = args.image_folder
    train_dir = data_dir + '/' + args.train_folder
    valid_dir = data_dir + '/' + args.valid_folder
    test_dir = data_dir + '/' + args.test_folder
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),

        'testing': transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),

        'validation': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}
    class_names = image_datasets['training'].classes

    return image_datasets, dataloaders, dataset_sizes, class_names


def initialize_model(model_name, learning_rate, hidden_layer1, hidden_layer2, output, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "vgg11":
        model = models.vgg11(pretrained=use_pretrained)

    elif model_name == "vgg13":
        model = models.vgg13(pretrained=use_pretrained)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=use_pretrained)

    elif model_name == "vgg19":
        model = models.vgg19(pretrained=use_pretrained)
        
    elif model_name == "densenet":
        model = models.densenet121(pretrained=use_pretrained)

    else:
        print("Invalid model name, exiting...")
        exit()

    for param in model.parameters():
        param.requires_grad = False

#     input_size = model.classifier[0].in_features 
    input_size = model.classifier.in_features if model_name =='densenet' else model.classifier[0].in_features

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('drpot', nn.Dropout(p=0.2)),
        ('hidden', nn.Linear(hidden_layer1, hidden_layer2)),
        ('relu2', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer2, output)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, scheduler, num_epochs, device, dataloaders, dataset_sizes):
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def checkpoint_save(model, model_name, image_datasets, optimizer):
    checkpoint = {'input_size': model.classifier.fc1.in_features,
                  'output_size': model.classifier.fc2.out_features,
                  'hidden_layers': [model.classifier.hidden.in_features, model.classifier.hidden.out_features],
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': image_datasets['training'].class_to_idx,
                  'arch': model_name
                  }
    #     convert checkpoint to cpu
    model.to('cpu')

    torch.save(checkpoint, 'checkpoint.pth')


def training(args):
    set_working_dir(r'/home/workspace/aipnd-project')
    image_datasets, dataloaders, dataset_sizes, class_names = data_transforms(args)
    model, criterion, optimizer = initialize_model(args.model, args.learning_r, args.h1, args.h2, args.output_size)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, args.epochs, args.device, dataloaders,
                           dataset_sizes)
    checkpoint_save(model_ft, args.model, image_datasets, optimizer)


if __name__ == "__main__":
    main()
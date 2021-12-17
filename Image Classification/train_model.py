#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects.double() / len(test_loader)
    
    logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(total_loss, total_acc))
    
    return total_loss

def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=50
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))        
        if loss_counter == 1:
            break
        #if epoch == 0:
        #    break 
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
               nn.Linear(num_features, 256),
               nn.ReLU(inplace=True),
               nn.Linear(256, 133))
    return model  
    
def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  
    train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=test_data_path, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    validation_set = torchvision.datasets.ImageFolder(root=validation_data_path, transform=testing_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True) 
    
    return train_loader, test_loader, validation_loader
   
    
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook.register_loss(loss_criterion)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start Model Training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook)
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Start Model Testing")
    test(model, test_loader, loss_criterion, device, hook)
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)")
    parser.add_argument("--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])

    args=parser.parse_args()
    print("!!!ARGS: ", args)
    main(args)

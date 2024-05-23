import torch
import os
import torchvision
from torchvision import datasets, models, transforms
import time
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
epochs = 20

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
#Should make this a repo
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {types[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



def train_model(model, loss_fcn, optimizer, scheduler, epochs):
    start = time.time()

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            cur_loss = 0.0
            cur_corr = 0.0

            for inputs, expected in data[phase]:
                inputs.to(device)
                expected.to(device)

                optimizer.zero_grad()

                torch.set_grad_enabled(phase == 'train')

                output = model(inputs)
                
                _, preds = torch.max(output, 1)
                loss = loss_fcn(output, expected)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                cur_loss += loss.item()
                cur_corr += torch.sum(preds == expected.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = cur_loss / image_amounts[phase]
            epoch_acc = cur_corr / image_amounts[phase]
            elapsed = time.time() - start

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s')

    elapsed = time.time() - start
    print(f'Training Complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    return model

transform = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomVerticalFlip(0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                  ]),
    'valid' : transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                  ])
}

data_dir = "data"

#puts all images in a dictionary
images = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x]) 
          for x in ['train', 'valid']}

#dictionary to make it easy to access the data for training
data = {x: torch.utils.data.DataLoader(images[x], batch_size=5,shuffle=True,
        num_workers=4) for x in ['train', 'valid']}
#amounts of each type of image
image_amounts = {'train' : len(images['train']), 'valid': len(images['valid'])}

#the types each image can be in training (a 1, 2, 3, etc.)
types = images['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load('./model.pt')
except:
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features #Used to keep input features consistent
    model.fc = torch.nn.Linear(num_ftrs,10) #Replace model head 

model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model,criterion, optimizer, lr_sched, epochs)

#Start of stuff for visualizing
# Get a batch of training data
inputs, classes = next(iter(data['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[types[x] for x in classes])
#end of stuff for visualizing

visualize_model(model, num_images=6)
torch.save(model, './model.pt')
input()

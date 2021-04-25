# What Are You Wearing Today? (WAYWT)


## HackGT Project 




## Table Of Contents


- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  * [Log in to the Microsoft Azure console and create a notebook instance](#log-in-to-the-microsoft-azure-console-and-create-a-notebook-instance)
  * [Use git to clone the repository into the notebook instance](#use-git-to-clone-the-repository-into-the-notebook-instance)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  * [Step 1 - Importing the datasets](#step-1---importing-the-datasets)
  * [Step 2 - Pre-processing data](#step-2---pre-processing-data)
  * [Step 3 - Training the CNN (using transfer learning)](#step-3---training-the-cnn-using-transfer-learning)
    + [Part A - Specify Loss Function and Optimizer](#part-a---specify-loss-function-and-optimizer)
    + [Part B - Train and Validate the Model](#part-b---train-and-validate-the-model)
    + [Part C - Test the Model](#part-c---test-the-model)
  * [Step 4 - Creation of user style vectors](#step-4---creation-of-user-style-vectors)
  * [Step 5 - Recommendation testing](#step-5---recommendation-testing)
- [Important - Deleting the notebook](#important---deleting-the-notebook)




## Introduction


The goal of this project is to develop a recommender system that will accept a few different user-supplied image of clothing as input, score them against the user's 'style vector' generated via user preferences during initialization of the app, and rank different outfits to help the user decide what to wear. All image files used to train the model for this project are from the DeepFashion dataset.


> Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.




## Setup Instructions


This project requires the following tools:


- Python - The programming language used by Flask.
- PostgreSQL - A relational database system.
- Virtualenv - A tool for creating isolated Python environments.


To get started, install Python and Postgres on your local computer if you don't have them already. A simple way for Mac OS X users to install Postgres is using [Postgres.app](https://postgresapp.com/). You can optionally use another database system instead of Postgres, like [SQLite](http://flask.pocoo.org/docs/1.0/patterns/sqlite3/).


The notebook in this repository is intended to be executed using Amazon's SageMaker platform and the following is a brief set of instructions on setting up a managed notebook instance using SageMaker.


### Log in to the Microsoft Azure console and create a notebook instance


Log in to the Azure console and go to the Azure dashboard. Click on 'Machine Learning'. It is recommended to enable GPUs for this particular project.



### Use git to clone the repository into the notebook instance


Once the instance has been started and is accessible, click on 'Open Jupyter' to get to the Jupyter notebook main page. To start, clone this repository into the notebook instance.


Click on the 'new' dropdown menu and select 'terminal'. By default, the working directory of the terminal instance is the home directory. Enter the appropriate directory and clone the repository as follows.


```
cd SageMaker
git clone https://github.com/Supearnesh/ml-waywt-rec.git
exit
```




## Machine Learning Pipeline


This was the general outline followed for this Azure project:


1. Importing the datasets
2. Pre-processing data
3. Training the CNN (using transfer learning)
4. Creation of user style vectors
5. Recommendation testing


### Step 1 - Importing the datasets


The DeepFashion dataset used in this project is open-source and freely available:


* Download the [DeepFashion dataset](https://drive.google.com/file/d/0B7EVK8r0v71pa2EyNEJ0dE9zbU0/view?usp=sharing).  Unzip the folder and place it in this project's home directory, at the location `/img`.


In the code cell below, we will write the file paths for the DeepFashion dataset in the numpy array `img_files` and check the size of the dataset.


```python
import numpy as np
from glob import glob

# !unzip img

# load filenames for clothing images
img_files = np.array(glob("img/*/*"))

# print number of images in each dataset
print('There are %d total clothing images.' % len(img_files))
```




### Step 2 - Pre-processing data


The data has already been randomly partitioned off into training, testing, and validation datasets so all we need to do is load it into a dataframe and validate that the data is split in correct proportions.


The images are then resized to 150 x 150 and centercropped to create an image tensor of size 150 x 150 x 3. They are initially 300 pixels in height and the aspect ratio is not altered. In the interest of time, this dataset will not be augmented by adding flipped/rotated images to the training set; although, that is an effective method to increase the size of the training set.


```python
import pandas as pd

df_full = pd.open_csv("data_attributes.csv")

df_train = df_full.loc[df_full['evaluation_status'] == 'train']][['img_path', 'category_values', 'attribute_values']]
df_test = df_full.loc[df_full['evaluation_status'] == 'test']][['img_path', 'category_values', 'attribute_values']]
df_val = df_full.loc[df_full['evaluation_status'] == 'val']][['img_path', 'category_values', 'attribute_values']]

print('The training set has %d records.' % len(df_train))
print('The testing set has %d records.' % len(df_test))
print('The validation set has %d records.' % len(df_val))
```


```python
import os
from PIL import Image
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### DONE: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
transform = T.Compose([T.Resize(150), T.CenterCrop(150), T.ToTensor()]) 

dataset_train = datasets.ImageFolder('img/train', transform=transform)
dataset_valid = datasets.ImageFolder('img/valid', transform=transform)
dataset_test = datasets.ImageFolder('img/test', transform=transform)

loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
loader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

loaders_transfer = {'train': loader_train, 'valid': loader_valid, 'test': loader_test}
```




### Step 3 - Training the CNN (using transfer learning)


The FashionNet model is nearly identical to the VGG-16 model architecture, with the exception of the last convolutional layer. However, instead of introducing the additional complexities of the FashionNet model, this model can be simplified by simply retaining the attributes embedding from the dataset. The data will be filtered into 1,000, potentially relevant buckets across 5 attributes of clothing, namely its pattern, material, fit, cut, and style. All layers use Rectified Linear Units (ReLUs) for the reduction in training times as documented by Nair and Hinton. It will be interesting to test the trained model to see how the the training and validation loss function perform.


> Vinod Nair and Geoffrey Hinton. [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf). In _Proceedings of ICML_, 2010.


An alternative could have been to use a pretrained VGG-19 model, which would yield an architecture similar to that described by Simonyan and Zisserman. The results attained by their model showed great promise for a similar image classification problem and it could have made sense to reuse the same architecture, and only modifying the final fully connected layer as done for the VGG-16 model in the cells below.


> Karen Simonyan and Andrew Zisserman. [Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size](https://arxiv.org/pdf/1409.1556.pdf). In _Proceedings of ICLR_, 2015.


```python
import torchvision.models as models
import torch.nn as nn
import torch

# The underlying network structure of FashionNet is identical to VGG-16
model_transfer = models.vgg19(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False

# The sixth, final convolutional layer will be adjusted to 1,000
model_transfer.classifier[6] = nn.Linear(1000, 133)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move to GPU
if use_cuda:
    model_transfer = model_transfer.cuda()

# create a complete CNN
model_transfer = Net()
print(model_transfer)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_transfer.cuda()

```


#### Part A - Specify Loss Function and Optimizer


Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
import torch.optim as optim

## select loss function
criterion_transfer = nn.CrossEntropyLoss()

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move loss function to GPU if CUDA is available
if use_cuda:
    criterion_transfer = criterion_transfer.cuda()

## select optimizer
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.001)

```


#### Part B - Train and Validate the Model


The model is to be trained and validated below, with [the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) to be saved at the filepath `'model_transfer.pt'`.


```python
n_epochs = 25

# train the model
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```


#### Part C - Test the Model


The model can be validated against test data to calculate and print the test loss and accuracy. We should ensure that the test accuracy is greater than 80%, as the implementation in the FashionNet paper yielded an accuracy of 85%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

```


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```




### Step 4 - Creation of user style vectors


This capability is the crux of a recommendation engine; it generates a feature vector for a particular user, based on images they have previously selected or liked, and subsequently compares future images to ascertain the similarity, or distance, from previous selections to recommend items that would be a good fit.


```python
## load attribute labels and their mappings

df_attributes = pd.read_csv('labels_attributes.csv')

# list of attribute names and their corresponding indices
attr_pattern = []
attr_material = []
attr_fit = []
attr_cut = []
attr_style = []

for i in range(len(df_attributes)):
    if df_attributes[['attribute_type_id']][i] == 1:
        attr_pattern.append(df_attributes[['attribute_id']][i])
    if df_attributes[['attribute_type_id']][i] == 2:
        attr_material.append(df_attributes[['attribute_id']][i])
    if df_attributes[['attribute_type_id']][i] == 3:
        attr_fit.append(df_attributes[['attribute_id']][i])
    if df_attributes[['attribute_type_id']][i] == 4:
        attr_cut.append(df_attributes[['attribute_id']][i])
    if df_attributes[['attribute_type_id']][i] == 5:
        attr_style.append(df_attributes[['attribute_id']][i])

```




### Step 5 - Recommendation testing


Test the recommender system on sample images. It would be good to understand the output and gauge its performance - regardless of which, it can tangibly be improved by:
* data augmentation of the training dataset by adding flipped/rotated images would yield a much larger training set and ultimately give better results
* further experimentation with CNN architectures could potentially lead to a more effective architecture with less overfitting
* an increase in training epochs, given more time, would both grant the training algorithms more time to converge at the local minimum and help discover patterns in training that could aid in identifying points of improvement


```python
import urllib
import matplotlib.pyplot as plt

img = Image.open(urllib.request.urlopen('https://images.footballfanatics.com/FFImage/thumb.aspx?i=/productimages/_2510000/altimages/ff_2510691alt1_full.jpg'))

plt.imshow(img)
plt.show()

transform = T.Compose([T.Resize(150), T.CenterCrop(150), T.ToTensor()])
transformed_img = transform(img)

# the images have to be loaded in to a range of [0, 1]
# then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_img = normalize(transformed_img)

# model loading
tensor_img = normalized_img.unsqueeze(0)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move image tensor to GPU if CUDA is available
if use_cuda:
    tensor_img = tensor_img.cuda()

# make prediction by passing image tensor to model
prediction = model_transfer(tensor_img)
# convert predicted probabilities to class index
tensor_prediction = torch.argmax(prediction)

# move prediction tensor to CPU if CUDA is available
if use_cuda:
    tensor_prediction = tensor_prediction.cpu()

predicted_class_index = int(np.squeeze(tensor_prediction.numpy()))

class_out = class_names[predicted_class_index] # predicted class index

# The output would then be compared against the user's style vector to rank against other potential outfits
```




## Important - Deleting the notebook


Always remember to shut down the notebook if it is no longer being used. Azure charges for the duration that a notebook is left running, so if it is left on then there could be an unexpectedly large Azure bill (especially if using a GPU-enabled instance). If allocating considerable space for the notebook (15-25GB), there might be some monthly charges associated with storage as well.

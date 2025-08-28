# <center>★ Implementation of Xception from scratch★

## Introduction

Xception, short for Extreme Inception, is a Deep Learning model developed by Francois Chollet at Google, continuing the popularity of Inception architecture, and further perfecting it.

The inception architecture utilizes inception modules, however, the Xception model replaces it with depthwise separable convolution layers, which totals 36 layers. When we compare the Xception model with the Inception V3 model, it only slightly performs better on the ImageNet dataset, however, on larger datasets consisting of 350 million images, Xception performs significantly better.

## The Journey of Deep Learning Models in Computer Vision

Utilization of deep learning architectures in computer vision began with [AlexNet](https://viso.ai/deep-learning/alexnet/) in 2012, It was the first to use Convolutional Neural Network architectures (CNNs) for [image recognition](https://viso.ai/computer-vision/image-recognition/), which won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

After AlexNet, the trend was to increase the convolutional blocks’ depth in the models, leading to researchers creating very deep models such as ZFNet, [VGGNet](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/), and [GoogLeNet](https://viso.ai/deep-learning/googlenet-explained-the-inception-model-that-won-imagenet/) (inception v1 model).

These models experimented with various techniques and combinations to improve accuracy and efficiency, with techniques such as smaller convolutional filters, deeper layers, and inception modules.

### The Inception Model

![image showing inception module](https://viso.ai/wp-content/uploads/2024/05/inception-module.png)  
*Inception Module – [source](https://conferences.computer.org/ictapub/pdfs/ITCA2020-6EIiKprXTS23UiQ2usLpR0/114100a262/114100a262.pdf)*

A standard convolution layer tries to learn filters in a 3D space, namely: width, height (spatial correlation), and channels (cross-channel correlation), thereby utilizing a single kernel to learn them.

However, the Inception module divides the task of [spatial](https://viso.ai/deep-learning/introduction-to-spatial-transformer-networks/) and cross-channel correlation using filters of different sizes (**1×1, 3×3, 5×5**) in parallel, hence, benchmarks proved that this is an efficient and better way to learn filters.

![standard inception module](https://viso.ai/wp-content/uploads/2024/05/standard-inception-module.jpg)  
*Standard Inception Module – [source](https://arxiv.org/pdf/1409.4842v1)*

Xception takes an even more aggressive approach as it entirely decouples the task of cross-channel and spatial correlation. This gave it the name Extreme Inception Model.

![diagram of xception model](https://viso.ai/wp-content/uploads/2024/05/xception-concept.jpg)  
*Concept of Xception architecture – [source](https://www.semanticscholar.org/paper/DeepHerb%3A-A-Vision-Based-System-for-Medicinal-using-Roopashree-Anitha/cfc002389353d10ba7d6bef73714f948fc92d119)*

## Xception Architecture

![image showing Xception architecture](https://viso.ai/wp-content/uploads/2024/05/xception-architecture.jpg)  
*Xception architecture – [source](https://link.springer.com/content/pdf/10.1007/s10479-022-05151-y.pdf)*

The Xception model’s core is made up of depthwise separable [convolutions](https://viso.ai/deep-learning/convolution-operations/). Therefore, before diving into individual components of Xception’s architecture, let’s take a look at depthwise separable convolution.

### Depthwise Separable Convolution

Standard convolution learns filters in 3D space, with each kernel learning width, height, and channels.

Whereas, a depthwise separable convolution divides the process into two distinctive processes using depth-wise convolution and pointwise convolution:

- **Depthwise Convolution:** Here, a single filter is applied to each input channel separately. For example, if an image has three color channels (red, green, and blue), a separate filter is applied to each color channel.
- **Pointwise Convolution:** After the depthwise convolution, a pointwise convolution is applied. This is a 1×1 filter that combines the output of the depthwise convolution into a single feature map.

![diagram of depthwise convolution](https://viso.ai/wp-content/uploads/2024/05/depthwise-convolution.jpg)  
*(a) Standard CNN. (b) Depthwise Separable – [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC7759122/)*

Xception utilizes a slightly modified version of this. In the original depthwise separable convolution, we first perform depthwise convolution and then pointwise convolution. The Xception model performs pointwise convolution first (1×1), and then the depthwise convolution using various **nxn** filters.

## The Three Parts of Xception Architecture

We divide the entire Xception architecture into three main parts: the entry flow, the middle flow, and the exit flow, with skip connections around the 36 layers.

- **Entry Flow**  
  - The input image is 299×299 pixels with 3 channels (RGB).  
  - A 3×3 convolution layer is used with 32 filters and a stride of 2×2. This reduces the image size and [extracts](https://viso.ai/deep-learning/feature-extraction-in-python/) low-level features. To introduce non-linearity, the ReLU activation function is applied.  
  - It is followed by another 3×3 convolution layer with 64 filters and ReLU.  
  - After the initial low-level feature extraction, the modified depthwise separable convolution layer is applied, along with the 1×1 convolution layer. Max pooling (3×3 with stride=2) reduces the size of the feature map.

- **Middle Flow**  
  - This block is repeated eight times.  
  - Each repetition consists of:  
    - Depthwise separable convolution with 728 filters and a 3×3 kernel.  
    - ReLU activation.  
  - By repeating it eight times, the middle flow progressively extracts higher-level features from the image.

- **Exit Flow**  
  - Separable convolution with 728, 1024, 1536, and 2048 filters, all with 3×3 kernels, further extracts complex features.  
  - Global Average Pooling summarizes the entire feature map into a single vector.  
  - Finally, at the end, a fully connected layer with logistic regression classifies the images.

### Regularization Techniques

Deep learning models aim to generalize (the model’s ability to adapt properly to new, previously unseen data), whereas [overfitting](https://viso.ai/computer-vision/what-is-overfitting/) stops the model from generalizing.

Overfitting is when a [model](https://viso.ai/computer-vision/typical-workflow-for-building-a-machine-learning-model/) learns noise from the training data or overly learns the training data. Regularization techniques help to prevent overfitting in machine learning models. The Xception model uses weight decay and dropout regularization techniques.

#### Weight Decay

Weight decay, also called L2 regularization, works by adding penalties to the larger weights. This helps to keep the size of weights small (when the weights are small, each feature contributes less to the overall decision of the model, which makes the model less sensitive to fluctuations in input data).

Without weight decay, the weight could grow exponentially, leading to overfitting.

#### Dropout

![image showing dropout](https://viso.ai/wp-content/uploads/2024/05/dropout.png)  
*Visualization of dropout operation: (a) full network; (b) network after dropout – [source](https://www.mdpi.com/1996-1073/13/20/5496)*

This regularization technique works by randomly ignoring certain neurons in training, during forward and backward passes. The dropout rate controls the probability that a certain [neuron](https://viso.ai/deep-learning/neuron-activation/) will be dropped. As a result, for each training batch, a different subset of neurons is activated, leading to more robust learning.

### Residual Connections

The Xception model has several skip connections throughout its architecture.

When training a very Deep Neural Network, the [gradients](https://viso.ai/computer-vision/gradient-descent/) used during training to update weights become small and even sometimes vanish. This is a major problem all deep learning models face. To overcome this, researchers came up with residual connections in their paper in 2016 on the [ResNet](https://viso.ai/deep-learning/resnet-residual-neural-network/) model.

Residual connections, also called skip connections, work by providing a connection between the earlier layers in the network with deeper or final layers in the network. These connections help the flow of gradients without vanishing, as they bypass the intermediate layers.

When using residual learning, the layers learn to approximate the difference (or residual) between the input and the output, as a result, the original function \( H(x) \) becomes \( H(x) = F(x) + x \).

**Benefits of Residual Connections:**

- **Deeper Networks:** Enables training of much deeper networks  
- **Improved Gradient Flow:** By providing a direct path for gradients to flow back to earlier layers, we solve the vanishing gradient problem.  
- **Better Performance**

Today, ResNet is a standard component in deep learning architectures.

## Performance and Benchmarks (From the paper)

The original paper on the Xception model used two different datasets: ImageNet and JFT. ImageNet is a popular dataset, which consists of 15 million labeled images with 20,000 categories. Testing used a subset of [ImageNet](https://viso.ai/deep-learning/imagenet/) containing around 1.2 million training images and 1,000 categories.

JFT is a large dataset that consists of over 350 million high-resolution [images annotated](https://viso.ai/computer-vision/image-annotation/) with labels of 17,000 classes.

We compare the Xception model with Inception v3 due to a similar parameter count. This ensures that any performance difference between the two models is a result of architecture efficiency and not its size.

The result obtained for ImageNet showed a marginal difference between the two models, however with a larger dataset like JFT, the Xception model shows a 4.3% relative improvement. Moreover, the Xception model outperforms the ResNet-152 and [VGG-16](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/) models.

## Performance on CIFAR10  

### Preprocessing

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# Preparing the dataset
# Define preprocessing for training and validation
transform_train = transforms.Compose([
    transforms.Resize((320, 320)),  # Resize to allow random cropping
    transforms.RandomResizedCrop(299),  # Random crop to 299x299
    transforms.RandomHorizontalFlip(),  # Augmentation for training
    transforms.ToTensor(),  # Convert to tensor (scales to [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize directly to 299x299
    transforms.CenterCrop(299),  # Center crop for consistency
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Training settings 

* Training set: 45k images
* Testing set: 10k images
* validation set: 5k images
* Batchsize: 128
* Optimiser: Adam
* Loss: Cross Entropy Loss
* Gradient scaler for faster training

### Ressources

* Kaggle notebook accelerated with 2 GPU T4

 In[1]: 
 ```python
import torch

num_gpus = torch.cuda.device_count()
print(f'Number of GPUs available: {num_gpus}')
```
Out[1]: Number of GPUs available: 2

### Inception V3 vs Xception 

![in](/images/train_invsx.png)

| Models/Performances| Parameters| Accuracy | Running time (s)|
|----------|----------|----------|----------|
| Inception V3 | 24,371,444  | 85.44 % | 1902.98 |
| Xception   | 20,825,402  | 84.87 %  | 3116.90 |

From our experiments, Inception V3 outperformed Xception in overall accuracy. However Xception tend to be more efficient. 

Xception was build from scratch [here](/notebooks/xception-implementation.ipynb) and Inception was loaded using the pytorch code below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Load Inception V3 model
# Set pretrained=True to use ImageNet pretrained weights, or False for random initialization
model = models.inception_v3(pretrained=False, init_weights=True)

num_classes = 10  # Adjust to your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

```
![comparison](/images/train_invsx.png)

### Xception with various activation

![Activations](/images/train_activations.png)

## References

1. [Computer vision ](https://viso.ai/computer-vision/)
2. [Deep Learning](https://viso.ai/deep-learning/)
3. [Kaggle notebook](https://www.kaggle.com/code/yasserh/xception-implementation)
4. [Xception paper](https://arxiv.org/pdf/1610.02357)







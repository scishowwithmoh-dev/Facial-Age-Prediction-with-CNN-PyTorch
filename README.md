# Facial-Age-Prediction-with-CNN-PyTorch
Convolutional Neural Network in PyTorch that uses regression to determine the age of an input's face by extracting features. Used UTKFace dataset.

### The Data Pipeline
As mentioned earlier, the dataset used is UTKFace, a dataset that contains cropped images of people's faces across all ages. We used the Kaggle API in order to retrieve the dataset and subsequently unzip it.

We defined two transformations: Transformations for training data, and transformations for validation data. Only the latter contained data augmentation methods. We found out the mean and std required for normalization via a method described in the datautils.py module.

We defined the dataset class using the directory containing the images. The getitem method was defined by combining the directory path with the name of the indexed file in order to retrieve the image, and a simple split statement in order to get the age label. (In every file, the age label is at the beginning of the filename, before an underscore.)

We defined a train dataset, then split it into training and validation, and then used a dataloader to batch them.

### The Model Architecture
For modularity, we defined a convolutional block class that takes the input channels, output channels, kernels, stride, and dropout frequency (added later). This convolutional block consists of a standard `nn.Conv2d()`, but also contained batch normalization, ReLU, and a max pool. It also included a neuron dropout that could be adjusted with the dropout frequency parameter.

We then combined 3 convolutional layers and a feed forward network consisting of 3 layers in the full CNN. The feed forward NN had batch normalization and dropout to function against the issue of exploding weights, particularly 

### Training Process
Pretty standard training process. Because we are using regression however, I used `nn.HuberLoss(delta=5.0)` as the loss function, particularly due to its resistance against exploding weights. I also assigned a MAE loss for evaluation. I defined a training epoch and a full training process. For evaluation, I placed a MAE loss function to evaluate every batch's performance, and average that out over the entire epoch.

## Improving the model
Initially, the model performed quite horribly after the first training (predicted a 12 year old was 47), and so I set out to look for possible issues and fixes for said issues. The first thing that came to my mind was exploding gradients. Regression problems, such as this one, are quite prone to exploding gradients compared to classification, especially in large architectures like a CNN. To combat this, I placed batch normalization everywhere I thought it would function well. In the convolutional block, and in the feed forward layer. After testing, this substantially improved the network's performance.

Another likely issue that I tried to remedy was feature loss. Essentially, in the flatten layer, there was a big jump from 128x30x30 to 512, and that (presumably) resulted in a lot of the finer features being lost in trying to compress 115200 into 512. To remedy this, I inserted an intermediate layer of size 1024 in between 512 and 115200. Not 100% sure if this noticeably improved it as I batched this in with the first one, however it likely did.

Another issue that I tried to remedy was the neural network relying on features incidental or weakly associated with age like setting, or glasses/clothes, instead of focusing on direct facial and bodily features, and generalizing. The general fix for this is dropout methods. I implemented them and there was a slight increase in performance.

Interestingly, overfitting was actually not an issue at all. Looking at the metrics per epoch, you can see that the training loss plateaued at about the same time that the validation loss did. This suggests that the data was complex enough that my 20 epochs didn't reach the necessary threshold for overfitting to occur (which is generally observed when training loss goes up and validation goes down).

## What I learned
What I learned is that regression is quite a different beast than the standard classification, especially with image processing like in neural networks. There are a lot more issues to account for, primarily exploding gradients, feature loss, and calibrating the loss function.

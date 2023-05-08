# Feed-Forward-Neural-Network-for-Image-Classification
Neural network on identifying simple test data and the MNIST data set.

</p>
Xinqi Chen @23/04/2023 

## Table of Content
- [Feed Forward Neural Network for Image Classification](#feed-forward-neural-network-for-image-classification)
  - [Abstract](#abstract)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [Algorithm Implementation and Development](#algorithm-implementation-and-development)
  - [Computational Results](#computational-results)
  - [Summary and Conclusions](#summary-and-conclusions)
  - [Acknowledgement](#acknowledgement)
  
## Abstract
This project implements a feed-forward neural network for image classification using the MNIST dataset. The goal is to build a model that can classify handwritten digits with high accuracy. The performance of the model is compared against other classifiers such as SVM and decision trees. The results are visualized using a confusion matrix.

## Overview
The MNIST dataset is a set of 70,000 images of handwritten digits, each of size 28 x 28 pixels. The goal is to build a classifier that can accurately identify the digit in each image. We start by performing a principal component analysis (PCA) to reduce the dimensionality of the dataset. We then build a feed-forward neural network with two hidden layers using the Keras library. The model is trained using the Adam optimizer and categorical cross-entropy loss function. We evaluate the model's performance on a test set and compare it against other classifiers such as SVM and decision trees.

## Theoretical Background
A feed-forward neural network is a type of artificial neural network that is commonly used for supervised learning tasks such as classification and regression. It consists of an input layer, one or more hidden layers, and an output layer. The network processes data by passing it through the layers, with each layer transforming the input in a non-linear way. The weights and biases of the network are learned through an optimization process using backpropagation.

PCA is a method for reducing the dimensionality of a dataset by projecting it onto a lower-dimensional subspace while preserving the most important information. It works by finding the principal components of the data, which are the directions that explain the most variance in the data.

## Algorithm Implementation
The following steps were taken to implement the feed-forward neural network for image classification:

First, load the MNIST dataset using the fetch_openml function from scikit-learn. Then PCA on the data was perform to reduce its dimensionality.
```ruby
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
```

The data was split into training and test sets, and was used to build a feed-forward neural network using the Keras library.
```ruby
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(1,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ]
)
```

The model was trained on the training data using the Adam optimizer and categorical cross-entropy loss function. After that, I evaluate the model's performance on the test data.
```ruby
# Build the neural network model
model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=10)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
```

Finally, the results was visualized using a confusion matrix.

Before applying the model to the MNIST data set, a smaller 1-D data set was used for examine the performence of the model. 

The code for this project is written in Python and uses libraries such as NumPy, scikit-learn, Keras, and Matplotlib. The code is available in the Jupyter Notebook file feed_forward_neural_network.ipynb.

## Computational Results

The Neural network model testing on 1-D array with 30 data performed better with larger test set. With the first 20 data as the training set, the least square error is
```
MSE on training set: 4.7277607630248895
MSE on test set: 13.147443723605564
```

With the first 10 data as training set, the least square error is
```
MSE on training set: 3.1167217045738305
MSE on test set: 8.819499679515138
```

Comparing the model fit to line fit and polynomial fit, it is clear that model fit has poorer MSE on training set, but behaves significantly better on test set. The use of a different set of training data leads to different results. This illustrates the importance of randomization and cross-validation in machine learning, as different choices of training and test data can affect the accuracy of the model.


### MNIST data

The first 20 PCS modes of the digital images are
```
Explained variance ratio: [0.09746116 0.07155445 0.06149531 0.05403385 0.04888934 0.04305227
 0.03278262 0.02889642 0.02758364 0.0234214  0.02106689 0.02037553
 0.01707064 0.01694019 0.01583382 0.01486335 0.01319356 0.01279006
 0.01187212 0.01152878]
```
 
The confusion matrix shows the result of model fitting is shown as
![CM]()
 
The model fit has a accuracy of 
```
Test accuracy: 0.9726428571428571
```
Comparing to LSTM, SVM (support vector machines) and decision tree classifiers, the results from neural network shows higher accuracy. When the digit pair with highest accuracy from in my past research reaches from SVM was 0.98, the average accuracy of neural network is 0.97. Additionally, decision trees can be effective classifiers, but they may struggle with high-dimensional datasets like the MNIST dataset. Additionally, decision trees can suffer from overfitting if the tree is too deep or if there are too many features. 

In general, feed-forward neural networks have shown to be very effective in image classification tasks, especially when dealing with high-dimensional datasets like the MNIST dataset.

## Summary and Conclusions
In this project, we explored the use of feed-forward neural networks for classification tasks on two datasets: a synthetic dataset and the MNIST handwritten digit dataset. We also compared the performance of the feed-forward neural network against other classifiers such as SVM, decision trees, and LSTM.

For the synthetic dataset, we trained a three-layer feed-forward neural network to predict a target variable based on a set of input variables. We split the dataset into training and testing sets and used mean squared error to evaluate the performance of the model. We also visualized the predictions and the model's decision boundary.

For the MNIST dataset, we used principal component analysis to reduce the dimensionality of the input images and then trained a feed-forward neural network to classify the digits. We compared the accuracy of the neural network against other classifiers using a confusion matrix.

Overall, our results demonstrate that feed-forward neural networks can be a powerful tool for classification tasks, especially when dealing with high-dimensional datasets such as images. With proper tuning of hyperparameters and architecture design, neural networks can outperform traditional classifiers on a variety of datasets.

## Acknowledgement
- [ChatGPT](https://platform.openai.com/)
- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html#user-guide)

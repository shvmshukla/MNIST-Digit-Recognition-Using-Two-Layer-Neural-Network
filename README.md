# MNIST-Digit-Recognition-Using-Two-Layer-Neural-Network

# How Do Neural Networks Work and why ReLu is selected for approximating a non linear boundary?
https://medium.com/machine-intelligence-report/how-do-neural-networks-work-57d1ab5337ce

# Backpropagation in detail
https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

# What is the purpose of multiple neurons in a hidden layer?
To explain using the sample neural network you have provided:

1. Purpose of the multiple inputs: Each input represents a feature of the input dataset.
2. Purpose of the hidden layer: Each neuron learns a different set of weights to represent different functions over the input data.
3. Purpose of the output layer: Each neuron represents a given class of the output (label/predicted variable).

If you used only a single neuron and no hidden layer, this network would only be able to learn linear decision boundaries. To learn non-linear decision boundaries when classifying the output, multiple neurons are required. By learning different functions approximating the output dataset, the hidden layers are able to reduce the dimensionality of the data as well as identify mode complex representations of the input data. If they all learned the same weights, they would be redundant and not useful.

The way they will learn different "weights" and hence different functions when fed the same data, is that when backpropagation is used to train the network, the errors represented by the output are different for each neuron. These errors are worked backwards to the hidden layer and then to the input layer to determine the most optimum value of weights that would minimize these errors.

This is why when implementing backpropagation algorithm, one of the most important steps is to randomly initialize the weights before starting the learning. If this is not done, then you would observe a large no. of neurons learning the exact same weights and give sub-optimal results.

https://datascience.stackexchange.com/questions/14028/what-is-the-purpose-of-multiple-neurons-in-a-hidden-layer

# Various types of activation functions and when to use them
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6



# What is the diffirence between SGD and back propogation?
Backpropagation is an efficient method of computing gradients in directed graphs of computations, such as neural networks. This is not a learning method, but rather a nice computational trick which is often used in learning methods. This is actually a simple implementation of chain rule of derivatives, which simply gives you the ability to compute all required partial derivatives in linear time in terms of the graph size (while naive gradient computations would scale exponentially with depth).

SGD is one of many optimization methods, namely first order optimizer, meaning, that it is based on analysis of the gradient of the objective. Consequently, in terms of neural networks it is often applied together with backprop to make efficient updates. You could also apply SGD to gradients obtained in a different way (from sampling, numerical approximators etc.). Symmetrically you can use other optimization techniques with backprop as well, everything that can use gradient/jacobian.

This common misconception comes from the fact, that for simplicity people sometimes say "trained with backprop", what actually means (if they do not specify optimizer) "trained with SGD using backprop as a gradient computing technique". Also, in old textbooks you can find things like "delta rule" and other a bit confusing terms, which describe exactly the same thing (as neural network community was for a long time a bit independent from general optimization community).

Thus you have two layers of abstraction:

gradient computation - where backprop comes to play
optimization level - where techniques like SGD, Adam, Rprop, BFGS etc. come into play, which (if they are first order or higher) use gradient computed above
https://stackoverflow.com/questions/37953585/what-is-the-diffirence-between-sgd-and-back-propogation






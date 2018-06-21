# MNIST-Digit-Recognition-Using-Two-Layer-Neural-Network

# How Do Neural Networks Work and why ReLu is selected for approximating a non linear boundary?
https://medium.com/machine-intelligence-report/how-do-neural-networks-work-57d1ab5337ce


# Backpropagation in detail
https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

# What is the purpose of multiple neurons in a hidden layer?
1. Purpose of the multiple inputs: Each input represents a feature of the input dataset.
2. Purpose of the hidden layer: Each neuron learns a different set of weights to represent different functions over the input data.
3. Purpose of the output layer: Each neuron represents a given class of the output (label/predicted variable).

If you used only a single neuron and no hidden layer, this network would only be able to learn linear decision boundaries. To learn non-linear decision boundaries when classifying the output, multiple neurons are required. By learning different functions approximating the output dataset, the hidden layers are able to reduce the dimensionality of the data as well as identify mode complex representations of the input data. If they all learned the same weights, they would be redundant and not useful.

The way they will learn different "weights" and hence different functions when fed the same data, is that when backpropagation is used to train the network, the errors represented by the output are different for each neuron. These errors are worked backwards to the hidden layer and then to the input layer to determine the most optimum value of weights that would minimize these errors.

This is why when implementing backpropagation algorithm, one of the most important steps is to randomly initialize the weights before starting the learning. If this is not done, then you would observe a large no. of neurons learning the exact same weights and give sub-optimal results.

https://datascience.stackexchange.com/questions/14028/what-is-the-purpose-of-multiple-neurons-in-a-hidden-layer

# Various types of activation functions and when to use them
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

# How does the ReLu solve the vanishing gradient problem?
To explain using the sample neural network you have provided:
The vanishing gradient problem affects saturating neurons or units only. For example the saturating sigmoid activation function as given below.

S(t)=1/1+e^−t

You can easily prove that

S(t)t→∞=1.0

and

S(t)t→−∞=0.0

Just by inspection. This is a saturation effect that becomes problematic as explained below, consider the first order derivative of the sigmoid function.

S′(t)=S(t)(1.0−S(t))

Since S(t) ranges from 0.0 to 1.0, its lower and upper asymptotes respectively, then clearly the derivative will be near zero for S(t)≈0.0 or S(t)≈1.0. The vanishing gradient problem comes about when the error signal passes backwards and starts approaching zero as it propagates backwards especially through neurons near saturation. If the network is deep enough the error signal from the output layer can be completely attenuated on it’s way back towards the input layer.

The attenuation comes about because the derivative S′(t) will always be near zero especially for saturating neurons, you do notice that if you use chain rule, which is backpropagation in neural net terms, you will somehow be multiplying this almost zero derivative with the error signal before throwing it backwards at every level or stage. Now keep doing that as you throw the error signal backwards and what you get is an error signal becoming weaker hence vanishing. Even the hyperbolic tangent has this saturating effect and hence the vanishing gradient problem also affects it.

Those units active in the linear region of the sigmoid won’t attenuate the error signal that much but this is generally problematic for very deep nets.

That’s why ReLUs are favorable not only because they solve the vanishing gradient problem but also because they result in highly sparse neural nets. Sparsity means efficient and reliable performance. The rectifier is as given below.

f(x)={0,for x<0
      x,for x≥0

Clearly it ranges from 0 to positive infinity thus it is non saturating and the derivative is given by

f′(x)={0,for x<0
       1,for x≥0

It’s always 1 hence no attenuation of an error signal propagating backwards. This makes ReLUs favorable for the deeper trainable feature detectors as in ConvNets, you can have very deep neural nets with ReLUs without the vanishing gradient problem.

EDIT:

You do notice that the negative region has a zero derivative, right? This can be a problem as the neuron is off within this region and cannot learn and gradients cannot be backpropagated through an off neuron. There are remedies to this by adding a leakage factor which results in a non-zero derivative and thus resulting in a modified neuron known as the leaky ReLU.

https://www.quora.com/How-does-the-ReLu-solve-the-vanishing-gradient-problem

# Pros of using ReLU as an activation function

The rectifier activation function allows a network to easily obtain sparse representations. For example, after uniform initialization of the weights, around 50% of hidden units continuous output values are real zeros, and this fraction can easily increase with sparsity-inducing regularization.

So rectifier activation function introduces sparsity effect on the network. Here are some advantages of sparsity from the same paper;

Information disentangling- One of the claimed objectives of deep learning algorithms (Bengio,2009) is to disentangle the factors explaining the variations in the data. A dense representation is highly entangled because almost any change in the input modifies most of the entries in the representation vector. Instead, if a representation is both sparse and robust to small input changes, the set of non-zero features is almost always roughly conserved by small changes of the input.

Efficient variable-size representation- Different inputs may contain different amounts of information and would be more conveniently represented using a variable-size data-structure, which is common in computer representations of information. Varying the number of active neurons allows a model to control the effective dimensionality of the representation for a given input and the required precision.

Linear separability- Sparse representations are also more likely to be linearly separable, or more easily separable with less non-linear machinery, simply because the information is represented in a high-dimensional space. Besides, this can reflect the original data format. In text-related applications for instance, the original raw data is already very sparse.

Distributed but sparse- Dense distributed representations are the richest representations, being potentially exponentially more efficient than purely local ones (Bengio, 2009). Sparse representations’ efficiency is still exponentially greater, with the power of the exponent being the number of non-zero features. They may represent a good trade-off with respect to the above criteria.
https://stats.stackexchange.com/questions/176794/how-does-rectilinear-activation-function-solve-the-vanishing-gradient-problem-in
For more details read the following paper:
https://www.utc.fr/~bordesan/dokuwiki/_media/en/glorot10nipsworkshop.pdf

# What is the difference between SGD and back propogation?
Backpropagation is an efficient method of computing gradients in directed graphs of computations, such as neural networks. This is not a learning method, but rather a nice computational trick which is often used in learning methods. This is actually a simple implementation of chain rule of derivatives, which simply gives you the ability to compute all required partial derivatives in linear time in terms of the graph size (while naive gradient computations would scale exponentially with depth).

SGD is one of many optimization methods, namely first order optimizer, meaning, that it is based on analysis of the gradient of the objective. Consequently, in terms of neural networks it is often applied together with backprop to make efficient updates. You could also apply SGD to gradients obtained in a different way (from sampling, numerical approximators etc.). Symmetrically you can use other optimization techniques with backprop as well, everything that can use gradient/jacobian.

This common misconception comes from the fact, that for simplicity people sometimes say "trained with backprop", what actually means (if they do not specify optimizer) "trained with SGD using backprop as a gradient computing technique". Also, in old textbooks you can find things like "delta rule" and other a bit confusing terms, which describe exactly the same thing (as neural network community was for a long time a bit independent from general optimization community).

Thus you have two layers of abstraction:

gradient computation - where backprop comes to play
optimization level - where techniques like SGD, Adam, Rprop, BFGS etc. come into play, which (if they are first order or higher) use gradient computed above
https://stackoverflow.com/questions/37953585/what-is-the-diffirence-between-sgd-and-back-propogation






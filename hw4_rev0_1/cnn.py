import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k), where k = 1 / in_features #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k), where                             #
        # k = 1 / (channels_in * kernel_size^2)                                    #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        poolsize = 2

        k_c1 = 1 / (self.C * self.filter_size * self.filter_size)
        self.params['W1'] = np.random.uniform(-np.sqrt(k_c1),np.sqrt(k_c1),(self.num_filters_1,self.C,self.filter_size,self.filter_size))

        F_conv1, H_conv1, W_conv1 = self.num_filters_1, self.H - self.filter_size + 1, self.W - self.filter_size + 1

        k_c2 = 1 / (F_conv1 * self.filter_size * self.filter_size)
        self.params['W2'] = np.random.uniform(-np.sqrt(k_c2),np.sqrt(k_c2),(self.num_filters_2,F_conv1,self.filter_size,self.filter_size))

        F_conv2, H_conv2, W_conv2 = self.num_filters_2, H_conv1//poolsize - self.filter_size + 1, W_conv1//poolsize - self.filter_size + 1
        
        in_features_l1 = F_conv2 * (H_conv2//poolsize) * (W_conv2//poolsize)
        k_l1 = 1 / in_features_l1
        self.params['W3'] = np.random.uniform(-np.sqrt(k_l1), np.sqrt(k_l1), (in_features_l1,hidden_dim))
        self.params['b3'] = np.random.uniform(-np.sqrt(k_l1), np.sqrt(k_l1), hidden_dim)

        in_features_l2 = hidden_dim
        k_l2 = 1 / in_features_l2        
        self.params['W4'] = np.random.uniform(-np.sqrt(k_l2), np.sqrt(k_l2), (in_features_l2,num_classes))
        self.params['b4'] = np.random.uniform(-np.sqrt(k_l2), np.sqrt(k_l2), num_classes)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")

        # conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax
        out_conv1, cache_conv1 = conv_forward(X,W1)
        out_relu1, cache_relu1 = relu_forward(out_conv1)
        out_poolmax1, cache_poolmax1 = max_pool_forward(out_relu1,pool_param)
        out_conv2, cache_conv2 = conv_forward(out_poolmax1,W2)
        out_relu2, cache_relu2 = relu_forward(out_conv2)
        out_poolmax2, cache_poolmax2 = max_pool_forward(out_relu2,pool_param)

        out_fc1, cache_fc1 = fc_forward(out_poolmax2.reshape(out_poolmax2.shape[0],-1),W3,b3)
        out_relu3, cache_relu3 = relu_forward(out_fc1)
        out_fc2, cache_fc2 = fc_forward(out_relu3,W4,b4)

        scores = out_fc2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # conv1 - relu1 - 2x2 max pool1 - conv2 - relu2 - 2x2 max pool2 - fc1 - relu3 - fc2 - softmax
        loss, grad_softmax = softmax_loss(out_fc2,y)
        gradx_fc2, gradw_fc2, gradb_fc2 = fc_backward(grad_softmax,cache_fc2)
        gradx_relu3 = relu_backward(gradx_fc2,cache_relu3)
        gradx_fc1, gradw_fc1, gradb_fc1 = fc_backward(gradx_relu3,cache_fc1)


        gradx_maxpool2 = max_pool_backward(gradx_fc1.reshape(out_poolmax2.shape),cache_poolmax2)
        gradx_relu2 = relu_backward(gradx_maxpool2,cache_relu2)
        gradx_conv2, gradw_conv2 = conv_backward(gradx_relu2,cache_conv2)
        gradx_maxpool1 = max_pool_backward(gradx_conv2,cache_poolmax1)
        gradx_relu1 = relu_backward(gradx_maxpool1,cache_relu1)
        gradx_conv1, gradw_conv1 = conv_backward(gradx_relu1,cache_conv1)

        grads["W1"] = gradw_conv1
        grads["W2"] = gradw_conv2
        grads["W3"] = gradw_fc1
        grads["b3"] = gradb_fc1
        grads["W4"] = gradw_fc2
        grads["b4"] = gradb_fc2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

from builtins import range
import numpy as np
import math


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn_layers.py!")


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)

    """
    out = None

    # Extract shapes and constants
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1

    # Construct output
    out = np.zeros((N, F, H_prime, W_prime))

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: For vectorization, you should loop through H_prime and W_prime.   #
    # for j in range(0, H_prime):                                             #
    #     for i in range(0, W_prime):                                         #
    #         ...                                                             #
    #         out[:,:,j,i] = ...                                              #
    # You may optionally want to implement a naive loop version for sanity    #
    # checking and then implement a vectorized version later.                 #
    # Hint: You can also check the max_pool_forward implementation for        #
    # implementation and vectorization tricks.                                #
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")

    for h_prime in range(H_prime):
        for w_prime in range(W_prime):
            
            out[:,:,h_prime,w_prime] = x[:,:, h_prime:h_prime+HH, w_prime:w_prime+WW].reshape(N,-1) @ w.reshape(F,-1).T
  
    # out2 = np.zeros((N, F, H_prime, W_prime))
    # for h in range(H_prime):
    #     for w1 in range(W_prime):
    #         for n in range(N):
    #             for f in range(F):
    #               for c in range(C):
    #                 out2[n,f,h,w1] += np.vdot(x[n,c,h:h+HH,w1:w1+WW], w[f,c,:,:])
    # print(out == out2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - dout: Upstream derivatives of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: A tuple of (x, w) as in conv_forward
      where x: Input data of shape (N, C, H, W)
      where w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    # Construct output
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    # w_pad : F x C x (2H - HH) x (2W - WW)
    pad_len = ((H_prime-1), (H_prime-1))
    w_pad = np.pad(w, ((0, 0), (0, 0), (pad_len[0], pad_len[0]), (pad_len[1], pad_len[1])), 'constant', constant_values=0)
    assert w_pad.shape[-2:] == (2*H - HH, 2*W - WW)

    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    # Hint: For vectorization, you can loop through HH and WW for dL/dw.      #
    # for j in range(0, HH):                                                  #
    #     for i in range(0, WW):                                              #
    #         ...                                                             #
    # Hint: For vectorization, you can loop through H and W for dL/dx.        #
    # for j in range(0, H):                                                   #
    #     for i in range(0, W):                                               #
    #         ...                                                             #
    # You may optionally want to implement a naive loop version for sanity    #
    # checking and then implement a vectorized version later.                 #
    #                                                                         #
    # Hint: You can also check the max_pool_backward implementation for       #
    # implementation and vectorization tricks.                                #
    # Hint: The full convolution is equal to padding the 'image' then         #
    # performing a valid convolution. The padded w is provided to you.        #
    # Hint: You can flip a numpy array by slicing.                            #
    # E.g. x[:,:,::-1,::-1] will flip the H and W dimensions of x.            #
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    
    # x:NCHW, w:FCHHWW, dout:NFH'W' 
    dout_FN = np.swapaxes(dout,0,1)
    x_CN = np.swapaxes(x,0,1)
    for hh in range(HH):
        for ww in range(WW):
            
            dw[:,:,hh,ww] = (x_CN[:,:,hh:hh+H_prime,ww:ww+W_prime].reshape(C,-1) @ dout_FN.reshape(F,-1).T).T
    
    w_pad_CF = np.swapaxes(w_pad,0,1)
    for h in range(H):
        for w in range(W):
            dx[:,:,h,w] = (w_pad_CF[:,:,h:h+H_prime,w:w+W_prime].reshape(C,-1) @ dout[:,:,::-1,::-1].reshape(N,-1).T).T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    (N, C, H, W) = x.shape

    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    S = (N, C, math.floor(H_out), math.floor(W_out))
    out = np.zeros(S)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                out[n, c, x1, y] = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])

    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            out[:, :, x1, y] = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    dx = np.zeros(x.shape)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                max_element = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])
    #                temp = np.zeros(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
    #                temp = (x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
    #                dx[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += dout[n, c, x1, y] * temp
    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            max_element = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
            max_element = max_element[:,:,np.newaxis]
            max_element = np.repeat(max_element, p_H, axis=2)
            max_element = max_element[:,:,:,np.newaxis]
            max_element = np.repeat(max_element, p_W, axis=3)
            temp = np.zeros(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
            temp = (x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
            tmp_dout = dout[:,:,x1,y]
            tmp_dout = tmp_dout[:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_H, axis=2)
            tmp_dout = tmp_dout[:,:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_W, axis=3)
            dx[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += tmp_dout * temp
    return dx

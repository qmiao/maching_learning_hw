import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from matplotlib import pyplot as plt
from datetime import datetime
import sys

def relu(x):
    """ ReLU
    Args:
        x: an np array of any shape
    Returns:
        An np array with same shape as x, and negative entries replaced by 0
    """
    return np.maximum(x, 0)

def grad_relu(x):
    """ the gradient of ReLU
    Args:
        x: an np array of any shape
    Returns:
        An np array with same shape as x, representing the gradient of ReLU.
        (The gradient of ReLU at 0 is defined to be 0 here.)
    """
    return np.array([(1 if i > 0 else 0) for i in x.ravel()]).reshape(x.shape)

def no_act(x):
    return x

def im2col(X, filter_shape, stride):
    """
    3D im2col, I referenced http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    for efficient implementation of im2col.
    Args:
        X: (h, w, d)-shaped np array
        filter_shape: a tuple (f_h, f_w, f_d)
        stride: a positive integer step size
    Returns:
        An (f_h*f_w*f_d, (w-f_w)/stride+1)-shaped np array with each column 
        representing each repective field of X based on the filter_shape and stride.
    """
    X_reshaped = np.transpose(X, (2, 0, 1))
    d, h, w = X_reshaped.shape
    f_h, f_w, f_d = filter_shape
    
    col_extent = w - f_w + 1
    row_extent = h - f_h + 1
    
    # Get Starting block indices
    start_idx = np.arange(f_h)[:, None] * w + np.arange(f_w)
    
    # Generate Depth indeces
    didx = h * w * np.arange(d)

    start_idx=(didx[:, None] + start_idx.ravel()).ravel('F')
    
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * w + np.arange(col_extent)
    
    indices = start_idx[:, None] + offset_idx[::stride, ::stride].ravel()

    # Get all actual indices & index into input array for final output
    out = np.take(X_reshaped, indices)
    return out

class CNNLayer:
    def __init__(self, n, filter_shape, stride, activation):
        """
        Initializes a conv layer object
        Args:
            n: the number of filters
            filter_shape: a tuple (f_h, f_w, f_d), the height, width, and depth of each filter
            activation: the type of activation to use (relu, no_act)
        """
        self.n = n
        self.filter_shape = filter_shape
        self.f_h, self.f_w, self.f_d = filter_shape
        self.stride = stride
        self.activation = activation
        
        # initializes the filters and bias terms to random numbers
        std_dev = sqrt(2.0/(self.f_h * self.f_w * self.f_d))
        # W is the list of n filters each with shape (f_h, f_w, f_d)
        self.W = np.random.randn(self.n, self.f_h, self.f_w, self.f_d) * std_dev
        # b is the list of n bias terms
        self.b = np.zeros((self.n, 1)) + std_dev
        
    def forward_step(self, X, pad):
        """
        Performs a forward convolution step using the conv layer.
        Args:
            X: the input data with shape (N, in_h, in_w, in_d)
                N: the number of input data examples
                in_h, in_w, in_d: the height, width, and depth of the input data (for each example)
            pad: the padding to be applied
        Returns:
            The 4D output tensor after applying the filters and the activation
        """
        
        self.X = X
        self.in_pad = pad
        self.N, self.in_h, self.in_w, self.in_d = X.shape

        # X_padded will be X after padding. It's initialized to empty right now and will be updated later
        self.X_padded = np.empty((self.N, self.in_h+pad*2, self.in_w+pad*2, self.in_d))
        
        # (out_h, out_w, out_d): height, width, depth of each 3D output tensor for each example
        self.out_h = int((self.in_h - self.f_h + 2 * pad) / self.stride) + 1
        self.out_w = int((self.in_w - self.f_w + 2 * pad) / self.stride) + 1
        self.out_d = self.n
        
        # Out will be the list of 3D output tensors for the N examples, each tensor has shape (out_h, out_w, out_d)
        self.Out = np.empty((self.N, self.out_h, self.out_w, self.out_d))
        
        # for the n filters stored in W, stretch out each (f_h, f_w, f_d)-shaped filter into a row of length (f_h*f_w*f_d),
        # and store the stretched-out n filters into an (n, f_h*f_w*f_d)-shaped np array W_row
        W_row = self.W.reshape(self.n, -1)
        
        # for each example, calculate its output tensor and update "Out"
        for example_index in range(self.N):
            
            # add the pad to the exampleas
            # X[example_index] has shape (in_h, in_w, in_d)
            # after adding pad, X_padded[example_index] will have shape (in_h+2*pad, in_w+2*pad, in_d)
            self.X_padded[example_index] = np.pad(X[example_index], pad_width=
                ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
            
            # stretch out the (in_h+2*pad, in_w+2*pad, in_d)-shaped input example into a 2D np array X_col with
            # each column representing each receptive field in the input example (from left to right, top to bottom)
            # i.e. X_col will have shape (f_h*f_w*f_d, ((in_w+2*pad-f_w)/stride+1)**2)
            X_col = im2col(self.X_padded[example_index], self.filter_shape, self.stride)

            # the dot product of W_row and X_col, adding the bias terms and calculates the activation values
            out = self.activation(np.dot(W_row, X_col) + self.b)

            # reshape out and update the "Out" tensor
            self.Out[example_index] = out.T.reshape((self.out_h, self.out_w, self.out_d))
            
        return self.Out # shape = (N, out_h, out_w, out_d)

    def backward_step(self, out_delta):
        """
        Performs a backward step duing the backpropagation based on out_delta, the "delta"
        at the next layer, and return the "delta" at this layer.
        Args:
            out_delta: the "delta" values at the next layer, should have shape (N, out_h, out_w, out_d)
        Returns:
            The "delta" values at this layer, should have shape (N, in_h+2*pad, in_w+2*pad, in_d)
        """
        
        self.Out_delta = out_delta # out_delta will be used again in the update function later

        # To convolve filters with out_delta, reconstruct W to be an (f_d, f_h, f_w, n)-shaped np array as follows
        W = np.fliplr(self.W.reshape(self.n, -1, self.f_d)).T.reshape((self.f_d, self.f_h, self.f_w, self.n))
        # then stretch out W to be an 2D np array and store in W_row
        W_row = W.reshape(self.f_d, -1)
        
        # calculates the pad that needs to be added to out_delta
        out_pad = int(((self.in_w+2*self.in_pad - 1) * self.stride - self.out_w + self.f_w) / 2)

        # In_delta will store the return values, i.e. the "delta" values at this layer
        self.In_delta = np.empty(self.X_padded.shape)
        for example_index in range(self.N):
            
            # Pad out_delta s.t. we can convolve the flipped filter with it
            out_delta_padded = np.pad(out_delta[example_index], pad_width=
                ((out_pad, out_pad), (out_pad, out_pad), (0, 0)), mode='constant', constant_values=0)

            # stretch out out_delta to a 2D np array with im2col
            out_delta_col = im2col(out_delta_padded, (self.f_h, self.f_w, self.n), self.stride)
            
            # update In_delta for this example
            self.In_delta[example_index] = np.dot(W_row, out_delta_col).T.reshape(self.X_padded[example_index].shape)
            
            # if we applied ReLU for activation, then need to multiply by the g'(in) term
            if self.activation is relu:

                grad_relu_in = np.array([(1 if i > 0 else 0) for i in self.X_padded[example_index].ravel()]).reshape(self.X_padded[example_index].shape)

                # self.In_delta[example_index] = np.multiply(grad_relu(self.X_padded[example_index]), self.In_delta[example_index])
                self.In_delta[example_index] = np.multiply(grad_relu_in, self.In_delta[example_index])
        return self.In_delta
    
    def update(self, alpha):
        """
        Perform a step of gradient descent based on the delta computed in backward_step
        Args:
            alpha: the learning rate
        Returns:
            The gradient wrt the filters, also updates filters and bias terms        
        """
        # Convolve output volume with input volume
        Out_delta = self.Out_delta
        In_activate = self.X_padded
        
        # Gradient_W and Gradient_b will hold the gradient values wrt the filters and bias terms for each example
        Gradient_W = list()
        Gradient_b = list()
        
        # for each example, calculates the gradient wrt the filters and bias terms
        for example_index in range(self.N):
            
            # Convert Out_delta into Out_delta_row with each row representing a depth of the Out_delta volume
            Out_delta_row = Out_delta[example_index].reshape(-1, self.out_d).T
            
            # Convert In_activate into In_activate_col with each column representing a "receptive field"
            depth_volume = In_activate[example_index][:,:,0].reshape(self.in_h+2*self.in_pad, self.in_w+2*self.in_pad, 1)
            In_activate_col = im2col(depth_volume, (self.out_h, self.out_w, 1), self.stride)
            
            for d in range(1, self.in_d):
                depth_volume = In_activate[example_index][:,:,d].reshape(self.in_h+2*self.in_pad, self.in_w+2*self.in_pad, 1)
                In_activate_col = np.append(In_activate_col, im2col(depth_volume, (self.out_h, self.out_w, 1), self.stride), axis=1)
            
            # Convolve Out with In
            # the dot product has shape (out_d, in_d*((in_w+2*in_pad-out_w)/stride+1)**2)
            gradient_W_temp = np.dot(Out_delta_row, In_activate_col).reshape(self.n, self.f_d, -1)
            gradient_W = np.empty_like(self.W)
            for filter_index in range(self.n):
                gradient_W[filter_index] = gradient_W_temp[filter_index].T.reshape(self.f_h, self.f_w, self.f_d)
                     
            gradient_b = np.dot(Out_delta_row, np.ones((self.out_h*self.out_w, 1)))
            
            # add the calculated gradients for this example to the Gradient_W and Gradient_b lists
            Gradient_W.append(gradient_W)
            Gradient_b.append(gradient_b)

        # update the filters and bias
        self.W = np.add(self.W, alpha * np.mean(Gradient_W, axis=0))
        self.b = np.add(self.b, alpha * np.mean(Gradient_b, axis=0))
        
        # return the gradient wrt the filters
        return(np.sum(Gradient_W, axis=0))
    
    def set_filters(self, filters, biases):
        """
        Set the filters and bias terms to the given values. (for testing)
        """
        self.W = filters
        self.b = biases[:, None]
        
    def print(self):
        """
        Print the current filters and bias terms
        """
        for f in range(self.n):
            print("Filter " + str(f) + ":")
            for d in range(self.f_d):
                print("Depth " + str(d) + ":")
                print(self.W[f][:,:,d])

class CNN:
    def __init__(self, h, s, filter_shape):
        """
        Initialize a convolutional neural network
        Args:
            h: integer, number of CNNLayers to use 
            s: integer, number of filters to use in each layer
            filter_shape = the height and width of each filter (f_h, f_w)
        """
        self.h = h
        self.s = s
        self.f_h, self.f_w = filter_shape
    
    def fit(self, X, y, alpha, t):
        """
        Train the network using back propagation using multiclass hinge loss,
        Also prints the loss and accuracy at each iteration.
        After all iterations, will plot cost vs. iterations
        Args:
            X: the training data (shape = (N, in_h, in_w, in_d))
            y: the set of labels for the training data (shape = (N, ))
            alpha: the learning rate
            t: the number of iterations
        """

        N, in_h, in_w, in_d = X.shape
        K = len(np.unique(y)) # number of classes
        
        print("Number of examples: " + str(N))
        
        # the Y matrix: each entry (i,j) is either 0 or 1, indicating whether
        # the i'th example belongs to class j
        Y = np.zeros((N, K))
        for i in range(N):
            Y[i][y[i]] = 1
        
        # Since my implementation only works for stride=1 for now, I am enforcing
        # stride=1 here. With stride=1, there is no need to pad the input, therefore,
        # I am also enforcing pad=0 here for forwarding.

        stride = 1
        in_pad = 0
        
        # a is the activation values for each layer, where a[0] is the input volume
        # each a[i] is an (N, h, w, d)-shaped np array, where h, w, d may differ for each layer
        a = [None] * (self.h+2)
        

        # cnnlayer[0] should never be accessed
        # cnnlayer[1] ... cnnlayer[h] are the CNNLayers
        # cnnlayer[-1] is the final FC layer
        self.cnnlayer = [None] * (self.h+2)
        
        # the two lists are for plotting cost vs. iterations
        num_iterations = list()
        cost_functions = list()

        accuracy = list()
        
        
        for iteration in range(t):
            
            a[0] = X
            f_d = in_d
            
            # Forward for each CNN Layer:
            for l in range(1, self.h+1):
                
                filter_shape = (self.f_h, self.f_w, f_d)
                self.cnnlayer[l] = CNNLayer(self.s, filter_shape, stride, relu)
                a[l] = self.cnnlayer[l].forward_step(a[l-1], in_pad)
                
                # determines the depth of the filter for the next cnnlayer
                f_d = a[l].shape[3]
            
            # FC Layer:
            # convert the FC layer to a CONV layer with shape (N, 1, 1, K)
            # by using K filters
            self.cnnlayer[-1] = CNNLayer(K, a[-2][0].shape, stride, relu)
            a[-1] = self.cnnlayer[-1].forward_step(a[-2], in_pad)            
            
            # calculate the loss by the activation value of the last FC layer
            loss = np.empty(N)
            
            # the delta value for each layer
            # d[0] should never be accessed
            # d[-1] should be for the FC layer
            # each d[i] should have same shape as each a[i]
            d = [None] * (self.h+2)
            
            # compute the delta value for the FC layer
            d[-1] = np.empty_like(a[-1])
            for example_index in range(N):
                # hinge loss gradient definition:
                # for the correct class, it's -1*(number of y_hat that greater than y_hat_star)
                # for the other classes, it's 0 if y_hat < y_hat_star, it's 1 if y_hat >= y_hat_star
                # where y_hat are the raw output of the FC layer
                # y_hat_star is the raw output of the FC layer for the correct class, minus 1
                
                # find k_star, which is the index of the correct class
                y_temp = Y[example_index]
                y_hat = a[-1][example_index].ravel()
                # now that y and y_hat are both a (K, )-shaped np array
                k_star = np.argmax(y_temp)
                
                # get the index of the largest "a" value that's not the correct class:
                k_largest = np.argmax(np.delete(y_hat, k_star))
                
                y_hat_star = y_hat[k_star]
                
                loss_vec = np.maximum(y_hat - y_hat_star + 1, 0)
                loss[example_index] = np.sum(loss_vec) - loss_vec[k_star]
                
                # gradient is the dL/da value at the FC layer
                gradient = np.array([(-1 if i == k_star else 1 if i == k_largest else 0) for i in y_hat])
                
                # to calculate the delta values at the FC layer, still need to multiply by g'(in_k) values
                relu_gradient = np.array([(1 if i > 0 else 0) for i in y_hat])
                delta = np.multiply(relu_gradient, gradient)
                
                d[-1][example_index] = delta.reshape((1, 1, K))           
            
            # update the two lists for plotting cost vs. iterations
            pred = np.argmax(a[-1], axis=3).ravel()
            
            loss_value = np.mean(loss)
            accuracy_value = np.mean(pred == y)
            
            format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f')
            print (format_str % (datetime.now(), iteration, loss_value, accuracy_value))
            
            
            num_iterations.append(iteration)
            cost_functions.append(loss_value)
            accuracy.append(accuracy_value)
            
            # Compute the delta value for the previous CNN Layers
            for l in range(self.h, 0, -1):
                d[l] = self.cnnlayer[l+1].backward_step(d[l+1])
                self.cnnlayer[l+1].update(alpha)
                
        # plot costs vs. iterations after fitting
        plt.plot(num_iterations, cost_functions)
        plt.xlabel("iterations")
        plt.ylabel("cost function")
        plt.title("Cost vs. Iterations (alpha=" + str(alpha) + ", h=" + str(self.h) + 
                                        ", s=" + str(self.s) + ", t=" + str(t) + ", " + 
                                        str(N) + " examples)")
        plt.show()
    
    def predict(self, T):
        """
        Return the predicted labels of test examples.
        """
        # same as forward in fit but with specified filters (self.W)
        in_pad = 0
        a = [None] * (self.h+2)
        a[0] = T
        
        # self.cnnlayer contains the layers
        for l in range(1, self.h+2):
            a[l] = self.cnnlayer[l].forward_step(a[l-1], in_pad)
        
        return np.argmax(a[-1], axis=3).ravel()
        
def normalize(X):
    """
    Normalizes input data X
    Args:
        X: an input numpy matrix of shape (m, n)
    Returns:
        a matrix after doing mean-variance normalization to each column of X 
    """
    m, n = X.shape
    
    # since I will be normalizing each column of X by iterating through the rows of
    # X transpose, therefore, I initiate normalized_X to be of shape (n, m).
    # After processing, I will return the transpose of normalized_X
    normalized_X = np.empty([n, m])
    
    index = 0
    for column in X.T:
        column = (column - np.mean(column))
        if np.std(column) != 0:
            column = column / np.std(column)

        # after calculating the normalized value for this column, append this 
        # normalized column to normalized_X
        normalized_X[index] = column
        index += 1

    return normalized_X.T


def normalize_test(train_X, test_X):
    """
    Normalizes test data by the same mean and variance of the train data
    Args:
        train_X: a numpy matrix of size (m1, n)
        test_X: a numpy matrix of size (m2, n)
    Returns:
        normalized test_X by normalizing each column around the mean and
        variance of each corresponding column of train_X
        for predicing and testing accuracy)
    """
    m, n = test_X.shape
    
    normazlied_text_X = np.empty([n, m])
    
    index = 0
    for column1, column2 in zip(train_X.T, test_X.T):
        column2 = (column2 - np.mean(column1))
        if np.std(column1) != 0:
            column2 = column2 / np.std(column1)
        normazlied_text_X[index] = column2
        index += 1
    return normazlied_text_X.T


# Train on the MNIST data set
df = pd.read_csv('MNIST_train.csv')
X = np.array(df.iloc[:100, 1:])
y = np.array(df['label'][:100])

X_normalized = normalize(X)
m, n = X_normalized.shape
X_reshaped = X_normalized.reshape(m, int(sqrt(n)), int(sqrt(n)), 1)

# select train and test data:
X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=1/3, random_state=0)

cnn = CNN(2, 32, (5, 5))
cnn.fit(X_train, y_train, 0.01, 1000)
pred = cnn.predict(X_test)

test_accuracy = np.mean(y_test == pred)
print("The test accuracy is " + str(test_accuracy))
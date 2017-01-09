import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  print N

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  for i in xrange(0, N):
    (D, C) = W.shape
    #W -= np.max(W)
    z = np.transpose(W).dot(X[i,:])
    sum_exp = 0.0
    for s in z:
        sum_exp += math.exp(s)
    normalized_z = np.exp(z)/sum_exp
    loss += normalized_z
    for c in xrange(0, C):
        temp = (y[c] - loss[c] )* X[i, :]
        #if c == y[i]:
        #    dW[:,i] -= X[i,:]
  num_train = len(X[1])
  loss /= num_train
  dW /= num_train
  su = 0
  for i in loss:
      su += i
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return su, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  W -= np.max(W)
  f = X.dot(W)
  loss = np.exp(f)/np.sum(np.exp(f))
  loss = 0.5 * reg * np.sum(W * W)
  print loss
  N = X.shape[0]
  dW = 1/N * (np.transpose(X).dot(y - loss))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


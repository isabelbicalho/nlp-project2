# Computer Science Deparment - Universidade Federal de Minas Gerais
# 
# Natural Language Processing (2018/2)
# Professor: Adriano Veloso
#
# @author Isabel Amaro
#
# References: Coursera: deeplearning.ai - Sequence Models


import numpy as np

from utils import sigmoid
from utils import softmax


class LSTM(object):

    def __init__(self):
        pass

    def forward(self, x, a0, parameters):
        """
        Implement forward propagation of teh current neural network using LSTM-cell

        Arguments:
        x          -- Input data for every time-step. Has shape (n_x, m, T_x).
        a0         -- Initial hidden state.           Has shape (n_a, m).
        parameters -- Python dictionary containing:
            W_f -- Weight matrix of the forget gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_f -- Bias of the forget gate.                               Numpy array of shape (n_a, 1).
            W_u -- Weight matrix of the update gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_u -- Bias of the update gate.                               Numpy array of shape (n_a, 1).
            W_c -- Weight matrix of the first "tanh".                     Numpy array of shape (n_a, n_a + n_x).
            b_c -- Bias of the first "tanh".                              Numpy array of shape (n_a, 1).
            W_o -- Weight matrix of the output gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_o -- Bias of the output gate.                               Numpy array of shape (n_a, 1).
            W_y -- Weight matrix relating the hidden-state to the output. Numpy array of shape (n_y, n_a).
            b_y -- Bias relating the hidden-state to the output.          Numpy array os shape (b_y, 1).

        Returns:
        a      -- Hidden states for every time-step. Numpy array of shape (n_a, m, T_x).
        y      -- Predictions for every time-step.   Numpy array of shape (n_y, m, T_x).
        caches -- Tuple of values needed for the backward pass.
        """

        # Initialize caches
        caches = []

        # Retrieve dimensions from shapes of X and parameters['W_y']
        n_x, m, T_x = x.shape
        n_y, n_a    = parameters["W_y"]
        
        # Initialize "a", "c" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_a, m, T_x))

        # Initialize a_next and c_next
        a_next = a0
        c_next = np.zeros((a0.shape))

        # loop over all time-steps
        for t in range(T_x):
            # Update next hidden state, next memory state, compute prediction, get the cache
            a_next, c_next, y_t, cache = self.cell_forward(x[:, :, t], a_next, c_next, parameters)
            # Save the value of the new "next" hidden state in a
            a[:, :, t] = a_next
            # Save the value of the prediction in y
            y[:, :, t] = y_t
            # Save the value of the next cell state
            c[:, :, t] = c_next
            # Append the cache into caches
            caches.append(cache)

        # Store the values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches

    def cell_forward (self, x_t, a_prev, c_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell

        Arguments:
        x_t        -- Input data at timestamp "t".     Numpy array of shape (n_x, m).
        a_prev     -- Hidden state at timestamp "t".   Numpy array of shape (n_a, m).
        c_prev     -- Memory state at timestamp "t-1". Numpy array of shape (n_a, m).
        parameters -- Python dictionary containing:
            W_f -- Weight matrix of the forget gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_f -- Bias of the forget gate.                               Numpy array of shape (n_a, 1).
            W_u -- Weight matrix of the update gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_u -- Bias of the update gate.                               Numpy array of shape (n_a, 1).
            W_c -- Weight matrix of the first "tanh".                     Numpy array of shape (n_a, n_a + n_x).
            b_c -- Bias of the first "tanh".                              Numpy array of shape (n_a, 1).
            W_o -- Weight matrix of the output gate.                      Numpy array of shape (n_a, n_a + n_x).
            b_o -- Bias of the output gate.                               Numpy array of shape (n_a, 1).
            W_y -- Weight matrix relating the hidden-state to the output. Numpy array of shape (n_y, n_a).
            b_y -- Bias relating the hidden-state to the output.          Numpy array os shape (b_y, 1).

        Returns:
        a_next   -- Next hidden state.           Numpy of shape (n_a, m).
        c_next   -- Next memory state.           Numpy of shape (n_a, m).
        y_t_pred -- Prediction at timestamp "t". Numpy array of shape (n_y, m).
        cache    -- Dictionary of values needed for backward pass.
        """

        # Retrieve parameters
        W_f = parameters["W_f"]
        b_f = parameters["b_f"]
        W_u = parameters["W_u"]
        b_u = parameters["b_u"]
        W_c = parameters["W_c"]
        b_c = parameters["b_c"]
        W_o = parameters["W_o"]
        b_o = parameters["b_o"]
        W_y = parameters["W_y"]
        b_y = parameters["b_y"]

        # Retrieve dimensions from shapes of x_t and W_y
        n_x, m   = x_t.shape
        n_y, n_a = W_y.shape

        # Equations:
        # forget_gate     = sigmoid(W_f.[a_prev, x_t] + b_f)
        # update_gate     = sigmoid(W_u.[a_prev, x_t] + b_u)
        # candidate_value = tanh(W_c.[a_prev, x_t] + c_prev)
        # new_cell        = forget_gate * cell_prev + update_gate * candidate_value
        # output_gate     = sigmoid(W_o.[a_prev, x_t] + b_o)
        # a               = output_gate * tanh(new_cell)

        # Concatenate a_prev and x_t
        concat = np.zeros((n_a + n_x, m))
        concat[:n_a, :] = a_prev
        concat[n_a:, :] = x_t

        # Compute the gate values
        forget_gate     = sigmoid(np.dot(W_f, concat) + b_f)
        update_gate     = sigmoid(np.dot(W_u, concat) + b_u)
        candidate_value = np.tanh(np.dot(W_c, concat) + b_c)
        c_next          = forget_gate * c_prev + update_gate * candidate_value
        output_gate     = sigmoid(np.dot(W_o, concat) + b_o)
        a_next          = output_gate * np.tanh(c_next)

        # Compute the prediction of the LSTM cell
        y_t_pred = softmax(np.dot(W_y, a_next) + b_y)

        # Store values needed for backward propagation in cache
        cache = {
            "a_next":          a_next,
            "c_next":          c_next,
            "a_prev":          a_prev,
            "c_prev":          c_prev,
            "forget_gate":     forget_gate,
            "update_gate":     update_gate,
            "candidate_value": candidate_value,
            "output_gate":     output_gate,
            "x_t":             x_t,
            "parameters":      parameters
        }

        return a_next, c_next, y_t_pred, cache

    def cell_backward(self, da_next, dc_next, cache):
        """
        Implement the backward pass for the LSTM-cell

        Arguments:
        da_next -- Gradients of next hidden state. Has shape (n_a, m).
        dc_next -- Gradients of next cell state.   Has shape (n_a, m).
        cache   -- Cache storing information from the forward pass.

        Returns:
        gradients -- Python dictionary containing:
            dx_t    -- Gradient of input data at time-step "t".              Has shape (n_x, m).
            da_prev -- Gradient w.r.t. the previous hidden state.            Numpy array of shape (n_a, m).
            dc_prev -- Gradient w.r.t. the previous memory state.            Has shape (n_a, m, T_x).
            dW_f    -- Gradient w.r.t. the weight matrix of the forget gate. Numpy array of shape (n_a, n_a + n_x).
            dW_u    -- Gradient w.r.t. the weight matrix of the update gate. Numpy array of shape (n_a, n_a + n_x).
            dW_c    -- Gradient w.r.t. the weight matrix of the memory gate. Numpy array of shape (n_a, n_a + n_x).
            dW_o    -- Gradient w.r.t. the weight matrix of the output gate. Numpy array of shape (n_a, n_a + n_x).
            db_f    -- Gradient w.r.t. biases of the forget gate.            Numpy array of shape (n_a, 1).
            db_u    -- Gradient w.r.t. biases of the update gate.            Numpy array of shape (n_a, 1).
            db_c    -- Gradient w.r.t. biases of the memory gate.            Numpy array of shape (n_a, 1).
            db_o    -- Gradient w.r.t. biases of the output gate.            Numpy array of shape (n_a, 1).
        """

        # Retrieve information from "cache"
        a_next =          cache["a_next"]
        c_next =          cache["c_next"]
        a_prev =          cache["a_prev"]
        c_prev =          cache["c_prev"]
        forget_gate =     cache["forget_gate"]
        update_gate =     cache["update_gate"]
        candidate_value = cache["candidate_value"]
        output_gate =     cache["output_gate"]
        x_t =             cache["x_t"]
        parameters =      cache["parameters"]

        # Retireve dimensions from x_t's shape and a_next's shape
        n_x, m = x_t.shape
        n_a, m = a_next.shape

        # Compute the gates related derivatives
        doutput_gate     = 
        dcandidate_value = 
        dupdate_state = 
        dforget_state = 

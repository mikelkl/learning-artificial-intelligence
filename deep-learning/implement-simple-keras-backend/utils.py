from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from six.moves import zip


class MyDense(Layer):
    def __init__(self, output_dim, activation=None,
                 kernel_initializer='he',
                 bias_initializer='zeros',
                 **kwargs):
        my_activations = {"relu": self.relu, "softmax": self.softmax}
        my_initializers = {"he": self.initialize_parameters_he,
                           'zeros': self.initialize_parameters_zeros}

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MyDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = my_activations.get(activation)
        self.kernel_initializer = my_initializers.get(kernel_initializer)
        self.bias_initializer = my_initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.my_add_weight(shape=(input_dim, self.output_dim),
                                         initializer=self.kernel_initializer)

        self.bias = self.my_add_weight(shape=(1, self.output_dim),
                                       initializer=self.bias_initializer)
        # self.built = True
        super(MyDense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        Z = np.dot(x, self.kernel) + self.bias
        A = self.activation(Z)
        return A

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def relu(self, Z):
        return Z * (Z > 0)

    def softmax(self, Z):
        """Compute softmax values for each sets of scores in Z."""
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def initialize_parameters_he(self, shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[0])

    def initialize_parameters_zeros(self, shape):
        # WX, not XW
        return np.zeros((shape[0], shape[1]))

    def my_add_weight(self, shape, initializer):
        return initializer(shape)

class MyRMSprop(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,
                 **kwargs):
        super(MyRMSprop, self).__init__(**kwargs)
        self.lr = K.variable(lr, name='lr')
        self.rho = K.variable(rho, name='rho')
        self.decay = K.variable(decay, name='decay')
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(MyRMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
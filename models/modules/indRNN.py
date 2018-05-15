from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell


class IndRNNCell(LayerRNNCell):  # 继承 LayerRNNCell

    def __init__(self,
                 num_units,
                 recurrent_min_abs=0,
                 recurrent_max_abs=None,
                 recurrent_kernel_initializer=None,
                 input_kernel_initializer=None,
                 activation=None,
                 reuse=None,
                 name=None):
        super(IndRNNCell, self).__init__(_reuse=reuse, name=name)

        self.input_spec = base_layer.InputSpec(ndim=2)

        # initialization
        self._num_units = num_units
        self._recurrent_min_abs = recurrent_min_abs

        self._recurrent_max_abs = recurrent_max_abs
        self._recurrent_recurrent_kernel_initializer = recurrent_kernel_initializer
        self._input_kernel_initializer = input_kernel_initializer
        self._activation = activation or nn_ops.relu


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        '''construct the IndRNN Cell'''
        if inputs_shape[1].value is None:
            raise ValueError("Expected input shape[1] is known")

        input_depth = inputs_shape[1]
        if self._input_kernel_initializer is None:
            self._input_kernel_initializer = init_ops.random_normal_initializer(mean=0,
                                                                                stddev=1e-3)
        # matrix W
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, self._num_units],
            initializer=self._input_kernel_initializer
        )

        if self._recurrent_recurrent_kernel_initializer is None:
            self._recurrent_recurrent_kernel_initializer = init_ops.constant_initializer(1.)

        # matrix U
        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape=[self._num_units],
            initializer=self._recurrent_recurrent_kernel_initializer
        )

        # Clip the U to min - max
        if self._recurrent_min_abs:
            abs_kernel = math_ops.abs(self._recurrent_kernel)
            min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
            self._recurrent_kernel = math_ops.multiply(
                math_ops.sign(self._recurrent_kernel),
                min_abs_kernel
            )
        if self._recurrent_max_abs:
            self._recurrent_kernel = clip_ops.clip_by_value(
                self._recurrent_kernel,
                -self._recurrent_max_abs,
                self._recurrent_max_abs
            )

        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype)
        )
        # built finished
        self.built = True


    def call(self, inputs, state):
        '''output = new state = activation(W * x + U (*) h_t-1 + b)'''

        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        # (*)
        state_update = math_ops.multiply(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, state_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output









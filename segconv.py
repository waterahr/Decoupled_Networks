from keras.layers.core import Layer
from keras.layers import Conv2D
from keras.engine import InputSpec
from keras import initializers
from keras.utils import conv_utils
from keras.regularizers import l2
import six
from keras.utils import plot_model
from keras import backend as K
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
import numpy as np

class SegConv(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same",  alpha=1, beta=0.5,
                 kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4), **kwargs):
        
        #self.nInputPlane = in_channels
        self.nOutputPlane = filters
        self.kernelSize = kernel_size
        self.alpha = alpha
        self.beta = beta
        self.strides = strides
        self.padding = padding
        self.kernelWeights = None
        self.rho = 1
        self.kernelInitializer = kernel_initializer
        self.kernelRegularizer = kernel_regularizer
        self.SphereConv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides,               
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)
        #self.LBCNN.weight.requires_grad=False        
        super(SegConv, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #print("****", input_shape)
        self.nInputPlane = input_shape[-1]
        #init weight
        initial_weights = K.random_normal((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane), 0, 1)
        self.kernelWeights = K.variable(initial_weights, name='{}_kernel_eights'.format(self.name))
        #self.kernelWeights = K.ones((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane))
        
        self.trainable_weights = [self.kernelWeights, self.rho]
        
        #print(self.kernelWeights)
        #print(self.kernelWeights.shape)
        
    def call(self, inputs, mask=None):
        #print(inputs * inputs)
        one_kernel = K.ones((self.kernelSize[0], self.kernelSize[0], self.nInputPlane, self.nOutputPlane))
        inputs_norm =  K.conv2d(inputs * inputs, one_kernel, strides = self.strides, padding = self.padding)
        inputs_norm = K.sqrt(inputs_norm)
        #print(inputs_norm)
        conv =  K.conv2d(inputs, self.kernelWeights, strides = self.strides, padding = self.padding)
        #print("+++", conv / ( inputs_norm * K.sqrt(K.sum(self.kernelWeights*self.kernelWeights))))
        #print(K.sqrt(K.sum(self.kernelWeights*self.kernelWeights)))
        g = conv / ( inputs_norm * K.sqrt(K.sum(self.kernelWeights*self.kernelWeights)))
        h_true = self.beta * inputs_norm + (self.alpha - self.beta) * self.rho
        h_false = self.alpha * inputs_norm
        flag = K.cast(K.greater(inputs_norm, self.rho), dtype='float32')
        h = flag * h_true + (1 - flag) * h_false
        return h * g
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        h_axis, w_axis = 1, 2
        stride_h, stride_w = self.strides
        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernelSize
        
        out_height = conv_utils.conv_output_length(height,
                                                      kernel_h, self.padding,
                                                      stride_h)
        out_width = conv_utils.conv_output_length(width,
                                                      kernel_w, self.padding,
                                                      stride_w)
        output_shape = (batch_size, out_height, out_width, self.nOutputPlane)
        #print(output_shape)
        return output_shape

    def get_config(self):
        config = {"in_channels" : self.nInputPlane,    
                  "out_channels" : self.nOutputPlane, 
                  "kernel_size" : self.kernelSize, 
                  "strides" : self.strides, 
                  "alpha" : self.alpha,
                  "beta" : self.beta,
                  "padding" : self.padding, 
                  "kernel_initializer" : self.kernelInitializer, 
                  "kernel_regularizer" : self.kernelRegularizer}
        base_config = super(SphereConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
if __name__ == "__main__":
    input_t = Input(shape=(3,3,1))
    conv_t = SegConv(filters=1, kernel_size=(2, 2), strides=(1, 1), padding="valid")(input_t)
    #SphereConv(in_channels=3, out_channels=64, kernel_size=(7, 7)).get_config()
    model1_t = Model(input_t, conv_t)
    a = np.asarray([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]).reshape(1, 3, 3, 1)
    print(a.shape)
    res1_t = model1_t.predict(a)
    print(res1_t.reshape(2,2))
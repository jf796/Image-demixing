import numpy as np
import tensorflow as tf
from PIL import Image


# from scipy.misc import imsave
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Layer, Input, Add, Conv2D, Multiply, Subtract
from keras.models import Model

import keras.initializers as initializers

# import cv2 as cv
import imageio
imsave = imageio.imsave

iteration_number = 400
patchsize = 50
distance = 10
img_width = patchsize
img_height = patchsize
size_v = img_width * img_height
Adam = optimizers.Adam(learning_rate=1e-4)


####################################################################
##    creat_image
####################################################################
def creat_image(image_tensor, dim1, dim2, overlap1, overlap2, n_ch):
    m1 = np.shape(image_tensor)[1]  # dimension of the patches
    n1 = np.shape(image_tensor)[2]

    num = int(float((dim1 - m1) / overlap1)) + 1
    nun = int(float((dim2 - n1) / overlap2)) + 1

    image = np.zeros((dim1, dim2, n_ch), dtype=float)
    count = np.zeros((dim1, dim2), dtype=float)
    ct = 0

    for i in [overlap1 * i for i in range(0, num)]:
        for j in [overlap2 * i for i in range(0, nun)]:
            image[i:i + m1, j:j + n1, :] = image[i:i + m1, j:j + n1, :] + image_tensor[ct, :, :, :]
            count[i:i + m1, j:j + n1] = count[i:i + m1, j:j + n1] + 1
            ct = ct + 1
    image = np.divide(image[:, :, 0], count + 0.0000001)

    return image


def creat_rgbimage(image_tensor, dim1, dim2, overlap1, overlap2, n_ch):
    m1 = np.shape(image_tensor)[1]  # dimension of the patches
    n1 = np.shape(image_tensor)[2]
    num = int(float((dim1 - m1) / overlap1)) + 1
    nun = int(float((dim2 - n1) / overlap2)) + 1
    image = np.zeros((dim1, dim2, n_ch), dtype=float)
    rgb = np.zeros((dim1, dim2, n_ch), dtype=float)
    count = np.zeros((dim1, dim2), dtype=float)
    ct = 0

    for i in [overlap1 * i for i in range(0, num)]:
        for j in [overlap2 * i for i in range(0, nun)]:
            image[i:i + m1, j:j + n1, :] = image[i:i + m1, j:j + n1, :] + image_tensor[ct, :, :, :]
            count[i:i + m1, j:j + n1] = count[i:i + m1, j:j + n1] + 1
            ct = ct + 1
    rgb[:, :, 0] = np.divide(image[:, :, 0], count + 0.0000001)
    rgb[:, :, 1] = np.divide(image[:, :, 1], count + 0.0000001)
    rgb[:, :, 2] = np.divide(image[:, :, 2], count + 0.0000001)
    return rgb


####################################################################
##    read data
####################################################################


image_data = Image.open(
    r"./aligned_data/woman_praying_aligned.tiff"
)  # open the woman praying image (reference photo for layer 1)
rgb1 = np.asarray(image_data, dtype="float32")  # convert that image to a float32 NumPy RGB array
image_data = Image.open(
    r"./aligned_data/manuscript.tiff"
)  # open the manuscript (reference photo for the hidden layer 2)
rgb2 = np.asarray(image_data, dtype="float32")  # convert to float32 NumPy RGB array
image_data = Image.open(
    r"./aligned_data/aligned_xray.bmp"
)  # open the xray image (the mixture of both layers we want to separate)
xray_data = np.asarray(image_data, dtype="float32")  # X-ray as a float32 array (loads as H x W x 3 RGB)
xray = np.empty((xray_data.shape[0],xray_data.shape[1],1),dtype="float32")
xray[:,:,0]=xray_data


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


g1 = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")
gray1_data = rgb2gray(rgb1)
g1[:, :, 0] = gray1_data
g2 = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")
gray2_data = rgb2gray(rgb2)
g2[:, :, 0] = gray2_data

xray = xray / 255
g1 = g1 / 255
g2 = g2 / 255
m = xray_data.shape[0]
n = xray_data.shape[1]
####################################################################
##    arrange image into tensor
####################################################################

image = xray
ch = image.shape[2]
num1 = int((m - patchsize) / distance) + 1
num2 = int((n - patchsize) / distance) + 1
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)
ct = 0
for i in range(0, num1):
    for j in range(0, num2):
        cube[ct, :, :, :] = image[i * distance:i * distance + patchsize, j * distance:j * distance + patchsize, :]
        ct = ct + 1
x_cube = cube

image = g1
ch = image.shape[2]
num1 = int((m - patchsize) / distance) + 1
num2 = int((n - patchsize) / distance) + 1
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)
ct = 0
for i in range(0, num1):
    for j in range(0, num2):
        cube[ct, :, :, :] = image[i * distance:i * distance + patchsize, j * distance:j * distance + patchsize, :]
        ct = ct + 1
g1_cube = cube

image = g2
ch = image.shape[2]
num1 = int((m - patchsize) / distance) + 1
num2 = int((n - patchsize) / distance) + 1
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)
ct = 0
for i in range(0, num1):
    for j in range(0, num2):
        cube[ct, :, :, :] = image[i * distance:i * distance + patchsize, j * distance:j * distance + patchsize, :]
        ct = ct + 1
g2_cube = cube

samples = x_cube.shape[0]
input_shape = g1_cube.shape[1:]



####################################################################
##    LISTA components
####################################################################
class Proximal_Conv_Operator(Layer):
    '''
    This layper perform Proximal operator. alpha^(t+1)=Prox_{threshold} (x) = sign(x)* (|x|-threshold)_{+}
    '''

    def __init__(self, units, threshold_initializer='glorot_uniform', **kwargs):
        self.units = units
        self.threshold_initializer = initializers.get(threshold_initializer)

        super(Proximal_Conv_Operator, self).__init__(**kwargs)

    def build(self, input_shape):
        # define the shape of threshold
        filters_dim = input_shape[-1]
        assert filters_dim == self.units
        

        self.threshold = self.add_weight(name='threshold',
                                         shape=(self.units,),
                                         initializer=self.threshold_initializer,
                                         trainable=True)
        self.built = True

    def call(self, x):
        outputs = K.sign(x) * K.maximum(K.abs(x) - self.threshold, 0)
        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'threshold_initializer': initializers.serialize(self.threshold_initializer)
        }
        base_config = super(Proximal_Conv_Operator, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class We_Conv_layer(Conv2D):
    def __init__(self, **kwargs):
        super(We_Conv_layer,self).__init__(**kwargs)


class S_Conv_layer(Conv2D):
    def __init__(self, **kwargs):
        super(S_Conv_layer, self).__init__(**kwargs)



####################################################################
##    define networks
####################################################################
kernel_dim = 5
filters = 128
number_layers = 6
def Encoder_x():

    input_ex = Input(shape=input_shape, name='InputEx')
    B = We_Conv_layer(kernel_size=kernel_dim, filters=filters, use_bias=False, padding='same')(input_ex)

    # initialise code
    code = Proximal_Conv_Operator(units=filters)(B)

    # start iterations
    for _ in range(number_layers - 1):
        C = S_Conv_layer(kernel_size=kernel_dim, filters=filters, use_bias=False, padding='same')(code)
        E = Add()([B, C])
        code = Proximal_Conv_Operator(units=filters)(E)

    E_x = Model(inputs=[input_ex], outputs=[code])

    return E_x


def Encoder_r():
    number_layers = 3
    input_er = Input(shape=input_shape, name='InputEr')
    B = We_Conv_layer(kernel_size=kernel_dim, filters=filters, use_bias=False, padding='same')(input_er)

    # initialise code
    code = Proximal_Conv_Operator(units=filters)(B)

    # start iterations
    for _ in range(number_layers - 1):
        C = S_Conv_layer(kernel_size=kernel_dim, filters=filters, use_bias=False, padding='same')(code)
        E = Add()([B, C])
        code = Proximal_Conv_Operator(units=filters)(E)

    E_r = Model(inputs=[input_er], outputs=[code])

    return E_r


def Decoder_r():
    input_dr = Input(shape=(input_shape[0],input_shape[1],filters), name='InputDr')
    x = Conv2D(kernel_size=kernel_dim, filters=input_shape[-1], use_bias=False, padding='same')(input_dr)
    D_r = Model(inputs=[input_dr], outputs=[x])
    return D_r


def Decoder_x():
    input_dx = Input(shape=(input_shape[0],input_shape[1],filters), name='InputDx')
    x = Conv2D(kernel_size=kernel_dim, filters=input_shape[-1], use_bias=False, padding='same')(input_dx)
    D_x = Model(inputs=[input_dx], outputs=[x])
    return D_x


####################################################################
##    traing the whole autoencoder networks
####################################################################

ze = np.zeros(shape=(samples, 1))


input_er1 = Input(shape=input_shape, name='input1')
input_er2 = Input(shape=input_shape, name='input2')
input_x = Input(shape=input_shape, name='input3')


Er = Encoder_r()
Ex = Encoder_x()
Dr = Decoder_r()
Dx = Decoder_x()

f1 = Er(input_er1)
f2 = Er(input_er2)
ff = Ex(input_x)
f = Add()([f1, f2])
r1 = Dr(f1)
r2 = Dr(f2)
x = Dx(f)
x1 = Dx(f1)
x2 = Dx(f2)
xx1 = Add()([x1, x2]) 
xx2 = Multiply()([x1, x2]) 
xr = Subtract()([xx1, xx2])

autoencoder = Model(inputs=[input_er1, input_er2, input_x], outputs=[r1, r2, x, xr])
autoencoder.compile(optimizer=Adam, loss=['mse', 'mse', 'mse', 'mse'], loss_weights=[1, 1, 4, 6])
autoencoder.fit([g1_cube, g2_cube, x_cube], [g1_cube, g2_cube, x_cube, x_cube],
                epochs=iteration_number,
                batch_size=32,
                shuffle=True)

Fx = Model(inputs=[input_er1, input_er2], outputs=[x1, x2])
[x1_pre_cube, x2_pre_cube] = Fx.predict([g1_cube, g2_cube])

####################################################################
##    Image recovery
####################################################################


x1_pre = creat_image(x1_pre_cube, m, n, distance, distance, 1)
x2_pre = creat_image(x2_pre_cube, m, n, distance, distance, 1)

imsave('x1_pre.jpg', x1_pre)
imsave('x2_pre.jpg', x2_pre)
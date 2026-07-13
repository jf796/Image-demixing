# ============================================================================
#  X-RAY IMAGE SEPARATION (source "demixing") via a LISTA convolutional
#  autoencoder.
#
#  Big picture: an X-ray of a painting often captures TWO overlaid layers at
#  once (e.g. the visible painting on top and a hidden manuscript/underdrawing
#  beneath). We also have two ordinary photographs ("reference" images), one
#  roughly showing each layer. This network learns sparse codes for the images
#  and a mixing model that reconstructs the X-ray as a combination of the two
#  layers, so we can then output each separated X-ray layer on its own.
# ============================================================================

import numpy as np  # NumPy: array maths, used for patch extraction / reassembly
import tensorflow as tf  # TensorFlow backend that Keras runs on
from PIL import Image  # Pillow: loads the .tiff / .bmp image files from disk

# from scipy.misc import imsave        # (old way to save images; removed - deprecated)
from keras import ops  # Keras backend-agnostic tensor ops (sign, abs, maximum...)
from keras import optimizers  # optimisers (we use Adam) for gradient descent
from keras.layers import Dense, Layer, Input, Add, Conv2D, Multiply, Subtract  # layer types used to build the network
from keras.models import Model  # the functional-API Model container class

import keras.initializers as initializers  # weight-initialisation helpers (for the learnable threshold)
import keras.constraints as constraints  # weight constraints (NonNeg keeps the thresholds >= 0)
import keras.regularizers as regularizers  # activity regularisers (optional L1 penalty on the codes)
from keras.layers import Activation  # Keras activation functions (we use linear for the output layers)

# import cv2 as cv                     # (OpenCV not used here)
import imageio  # imageio: modern library for reading/writing images
import matplotlib.pyplot as plt  # matplotlib: draw the side-by-side comparison figures (renders inline in Colab)

imsave = imageio.imsave  # alias so we can call imsave(...) to write output images to disk

# ---------------------------------------------------------------------------
#  Global hyper-parameters controlling the run
# ---------------------------------------------------------------------------
iteration_number = 5  # number of training epochs (kept small for a quick test run)
# iteration_number = 400              # (use 400 epochs for full-quality final results)
patchsize = 50  # each training example is a 50x50 pixel patch cropped from the image
distance = 40  # stride: move 40 px between neighbouring patches (so patches overlap)
img_width = patchsize  # patch width  = 50 (alias for readability)
img_height = patchsize  # patch height = 50 (alias for readability)
size_v = img_width * img_height  # total pixels per patch (2500); a convenience constant
Adam = optimizers.Adam(learning_rate=1e-4)  # Adam optimiser with a small learning rate of 0.0001
threshold_init = 0.25  # DEFAULT soft-threshold seed (used by the x-ray encoder Ex). The two reference
# encoders are now separate instances and each gets its own seed below (their photos differ hugely in contrast).
threshold_init_r1 = 0.15  # reference-1 encoder: image 1 is smooth/low-contrast, so a big threshold wipes its
# code out; a smaller seed keeps it from collapsing to near-empty.
threshold_init_r2 = 0.30  # reference-2 encoder: high-contrast manuscript tolerates (and benefits from) more shrinkage.
l1_lambda = 1e-8  # optional extra L1 sparsity penalty on the codes (0 = off; it is SUMMED over all code
# elements so it is very strong - if you enable it, start around 1e-8 and watch the occupancy diagnostic)
excl_weight = 1.0  # weight of the gradient-EXCLUSION loss (5th term): pushes the two separated X-ray
# layers x1,x2 to place their edges in DIFFERENT locations. Raise it if the layers still bleed into each
# other, lower it if the recombined-X-ray (xr) MSE suffers. Set 0.0 to disable the term entirely.


####################################################################
##    creat_image
####################################################################
def creat_image(image_tensor, dim1, dim2, overlap1, overlap2, n_ch):
    # Reassemble a full (dim1 x dim2) single-channel image from a stack of
    # overlapping patches, averaging the overlap regions back together.
    m1 = np.shape(image_tensor)[1]  # patch height (rows of each patch)
    n1 = np.shape(image_tensor)[2]  # patch width  (cols of each patch)

    num = int(float((dim1 - m1) / overlap1)) + 1  # how many patch positions fit vertically
    nun = int(float((dim2 - n1) / overlap2)) + 1  # how many patch positions fit horizontally

    image = np.zeros((dim1, dim2, n_ch), dtype=float)  # accumulator for summed pixel values
    count = np.zeros((dim1, dim2), dtype=float)  # counts how many patches covered each pixel
    ct = 0  # running index into the patch stack (image_tensor)

    for i in [overlap1 * i for i in range(0, num)]:  # top-edge row of each patch (0, 40, 80, ...)
        for j in [overlap2 * i for i in range(0, nun)]:  # left-edge col of each patch (0, 40, 80, ...)
            image[i : i + m1, j : j + n1, :] = (  # add this patch's pixels into the accumulator...
                image[i : i + m1, j : j + n1, :] + image_tensor[ct, :, :, :]
            )
            count[i : i + m1, j : j + n1] = count[i : i + m1, j : j + n1] + 1  # ...and note the pixels were covered once more
            ct = ct + 1  # move to the next patch in the stack
    image = np.divide(image[:, :, 0], count + 0.0000001)  # average = sum / count (tiny epsilon avoids /0)

    return image  # the reconstructed 2-D image


def creat_rgbimage(image_tensor, dim1, dim2, overlap1, overlap2, n_ch):
    # Same patch-reassembly idea as creat_image, but keeps all 3 colour channels
    # (currently unused by the main pipeline, but handy for RGB outputs).
    m1 = np.shape(image_tensor)[1]  # patch height
    n1 = np.shape(image_tensor)[2]  # patch width
    num = int(float((dim1 - m1) / overlap1)) + 1  # number of vertical patch positions
    nun = int(float((dim2 - n1) / overlap2)) + 1  # number of horizontal patch positions
    image = np.zeros((dim1, dim2, n_ch), dtype=float)  # summed-values accumulator
    rgb = np.zeros((dim1, dim2, n_ch), dtype=float)  # final averaged RGB output buffer
    count = np.zeros((dim1, dim2), dtype=float)  # per-pixel coverage count
    ct = 0  # patch index

    for i in [overlap1 * i for i in range(0, num)]:  # vertical patch origins
        for j in [overlap2 * i for i in range(0, nun)]:  # horizontal patch origins
            image[i : i + m1, j : j + n1, :] = (  # accumulate this patch into the buffer
                image[i : i + m1, j : j + n1, :] + image_tensor[ct, :, :, :]
            )
            count[i : i + m1, j : j + n1] = count[i : i + m1, j : j + n1] + 1  # increment coverage count
            ct = ct + 1  # next patch
    rgb[:, :, 0] = np.divide(image[:, :, 0], count + 0.0000001)  # average the Red channel
    rgb[:, :, 1] = np.divide(image[:, :, 1], count + 0.0000001)  # average the Green channel
    rgb[:, :, 2] = np.divide(image[:, :, 2], count + 0.0000001)  # average the Blue channel
    return rgb  # reconstructed 3-channel image


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
if xray_data.ndim == 3:  # the .bmp is grayscale stored in 3 identical RGB channels
    xray_data = xray_data[:, :, 0]  # keep one channel -> H x W (R==G==B, so this is lossless)
xray = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")  # allocate an H x W x 1 tensor
xray[:, :, 0] = xray_data  # place the X-ray into its single channel (adds the channel axis)


def rgb2gray(rgb):
    # Convert an RGB image to grayscale using the standard luminance weights.
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # weighted sum of R,G,B -> single luminance value


def histogram_stretch(img, low_pct=1.0, high_pct=99.0):
    # Paper's preprocessing for the mixed X-ray ("histogram stretching / normalization"):
    # a percentile-based contrast stretch. Drop the lowest/highest 1% of grayscale
    # values (outlier removal), then linearly rescale the surviving [p_low, p_high]
    # range onto the full [0, 255] dynamic range so every X-ray has a common scale.
    lo = np.percentile(img, low_pct)  # 1st-percentile grayscale value
    hi = np.percentile(img, high_pct)  # 99th-percentile grayscale value
    stretched = (img - lo) / (hi - lo + 1e-8) * 255.0  # map [lo, hi] -> [0, 255]
    return np.clip(stretched, 0, 255).astype("float32")  # clamp the clipped 1% tails to the [0, 255] ends


g1 = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")  # allocate H x W x 1 for reference-1 gray
gray1_data = rgb2gray(rgb1)  # grayscale version of the "woman praying" photo
g1[:, :, 0] = gray1_data  # store it in g1's single channel
g2 = np.empty((xray_data.shape[0], xray_data.shape[1], 1), dtype="float32")  # allocate H x W x 1 for reference-2 gray
gray2_data = rgb2gray(rgb2)  # grayscale version of the "manuscript" photo
g2[:, :, 0] = gray2_data  # store it in g2's single channel

xray = histogram_stretch(xray)  # paper preprocessing: clip the 1%/99% tails, stretch dynamic range to [0,255]
xray = xray / 255  # then normalise X-ray pixel values from [0,255] to [0,1]
g1 = histogram_stretch(g1)  # same 1%/99% clip + stretch on reference-1 (equalises its low contrast)
g1 = g1 / 255  # then normalise reference-1 grayscale to [0,1]
g2 = histogram_stretch(g2)  # same 1%/99% clip + stretch on reference-2
g2 = g2 / 255  # then normalise reference-2 grayscale to [0,1]
m = xray_data.shape[0]  # image height (number of rows), used everywhere below
n = xray_data.shape[1]  # image width  (number of cols)
####################################################################
##    arrange image into tensor
####################################################################

# --- Slice the X-ray into a stack of overlapping 50x50 patches ("cube") ---
image = xray  # work on the X-ray image
ch = image.shape[2]  # number of channels (1 for X-ray)
num1 = int((m - patchsize) / distance) + 1  # number of patch rows that fit vertically
num2 = int((n - patchsize) / distance) + 1  # number of patch cols that fit horizontally
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")  # flat buffer sized to hold all patches
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)  # reshape into (num_patches, 50, 50, ch)
ct = 0  # patch counter
for i in range(0, num1):  # loop over vertical patch positions
    for j in range(0, num2):  # loop over horizontal patch positions
        cube[ct, :, :, :] = image[  # copy the 50x50 window at (i,j) into patch slot ct
            i * distance : i * distance + patchsize,  # rows i*10 .. i*10+50
            j * distance : j * distance + patchsize,  # cols j*10 .. j*10+50
            :,
        ]
        ct = ct + 1  # advance to next patch slot
x_cube = cube  # x_cube = the full stack of X-ray patches (network input/target)

# --- Same patch extraction for reference image 1 (woman praying, gray) ---
image = g1  # work on reference-1 grayscale
ch = image.shape[2]  # channels (1)
num1 = int((m - patchsize) / distance) + 1  # vertical patch count
num2 = int((n - patchsize) / distance) + 1  # horizontal patch count
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")  # flat buffer for patches
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)  # (num_patches, 50, 50, 1)
ct = 0  # patch counter
for i in range(0, num1):  # vertical positions
    for j in range(0, num2):  # horizontal positions
        cube[ct, :, :, :] = image[  # extract the 50x50 window
            i * distance : i * distance + patchsize,
            j * distance : j * distance + patchsize,
            :,
        ]
        ct = ct + 1  # next patch
g1_cube = cube  # g1_cube = stack of reference-1 patches

# --- Same patch extraction for reference image 2 (manuscript, gray) ---
image = g2  # work on reference-2 grayscale
ch = image.shape[2]  # channels (1)
num1 = int((m - patchsize) / distance) + 1  # vertical patch count
num2 = int((n - patchsize) / distance) + 1  # horizontal patch count
tensor = np.zeros((num1 * num2 * patchsize, patchsize * ch), dtype="float32")  # flat buffer for patches
cube = tensor.reshape(num1 * num2, patchsize, patchsize, ch)  # (num_patches, 50, 50, 1)
ct = 0  # patch counter
for i in range(0, num1):  # vertical positions
    for j in range(0, num2):  # horizontal positions
        cube[ct, :, :, :] = image[  # extract the 50x50 window
            i * distance : i * distance + patchsize,
            j * distance : j * distance + patchsize,
            :,
        ]
        ct = ct + 1  # next patch
g2_cube = cube  # g2_cube = stack of reference-2 patches

samples = x_cube.shape[0]  # total number of patches (training-set size)
input_shape = g1_cube.shape[1:]  # shape of one patch = (50, 50, 1); the network's input shape


####################################################################
##    LISTA components
####################################################################
class Proximal_Conv_Operator(Layer):
    """
    This layper perform Proximal operator. alpha^(t+1)=Prox_{threshold} (x) = sign(x)* (|x|-threshold)_{+}
    """
    # This is the "soft-thresholding" / shrinkage non-linearity at the heart of
    # LISTA. It pushes small activations to exactly zero (encouraging sparsity)
    # and shrinks larger ones toward zero by a learnable per-channel threshold.

    def __init__(self, units, threshold_initializer=None, **kwargs):
        self.units = units  # number of feature channels this operator acts on
        # Start thresholds at a small POSITIVE constant. glorot_uniform used to
        # start ~half of them negative, which means no shrinkage on those channels.
        if threshold_initializer is None:
            threshold_initializer = initializers.Constant(threshold_init)
        self.threshold_initializer = initializers.get(threshold_initializer)  # how to initialise the threshold weights
        # Optional L1 penalty on this layer's output (the code) to reward sparsity.
        if l1_lambda > 0:
            kwargs.setdefault("activity_regularizer", regularizers.l1(l1_lambda))
        super(Proximal_Conv_Operator, self).__init__(**kwargs)  # run the base Layer constructor

    def build(self, input_shape):
        # Called once, on first use, to create the layer's trainable weights.
        filters_dim = input_shape[-1]  # number of channels arriving from the previous layer
        assert filters_dim == self.units  # sanity check: must match the declared unit count

        self.threshold = self.add_weight(  # create one learnable threshold value per channel
            name="threshold",
            shape=(self.units,),  # a vector of length = number of channels
            initializer=self.threshold_initializer,  # initial values (small positive constant)
            constraint=constraints.NonNeg(),  # clamp threshold >= 0 so it always shrinks, never inflates
            trainable=True,  # let back-prop update these thresholds
        )
        self.built = True  # mark the layer as built

    def call(self, x):
        # Forward pass: soft-threshold each element of x.
        outputs = ops.sign(x) * ops.maximum(ops.abs(x) - self.threshold, 0)  # sign(x)*max(|x|-threshold, 0)
        return outputs  # small values -> 0, large values shrunk toward 0

    def get_config(self):
        # Lets Keras serialise/save this custom layer to disk.
        config = {
            "units": self.units,
            "threshold_initializer": initializers.serialize(self.threshold_initializer),
        }
        base_config = super(Proximal_Conv_Operator, self).get_config()  # base Layer config

        return dict(list(base_config.items()) + list(config.items()))  # merge base + custom config

    def compute_output_shape(self, input_shape):
        return input_shape  # output shape is identical to input (element-wise op)


class We_Conv_layer(Conv2D):
    # "W_e" in LISTA: the convolution that maps the input image into the code
    # space (the fixed term added every iteration). Just a named Conv2D subclass.
    def __init__(self, **kwargs):
        super(We_Conv_layer, self).__init__(**kwargs)  # behaves exactly like a Conv2D


class S_Conv_layer(Conv2D):
    # "S" in LISTA: the convolution applied to the current code at each iteration
    # (the recurrent/refinement term). Also just a named Conv2D subclass.
    def __init__(self, **kwargs):
        super(S_Conv_layer, self).__init__(**kwargs)  # behaves exactly like a Conv2D


####################################################################
##    Gradient-exclusion loss (the 5th training term)
####################################################################
# Zhang et al., "Single Image Reflection Separation with Perceptual Losses"
# (CVPR 2018). Two correctly separated layers should NOT have edges in the
# same spatial locations. This layer measures how much the two layers' gradient
# fields OVERLAP; driving that overlap toward 0 stops one layer from leaking
# structure into the other (i.e. it disambiguates the x1<->x2 factorisation).
class GradientExclusion(Layer):
    def __init__(self, levels=3, **kwargs):
        super(GradientExclusion, self).__init__(**kwargs)
        self.levels = levels  # sum the overlap measure over this many image scales

    @staticmethod
    def _gradients(img):
        gx = img[:, :, 1:, :] - img[:, :, :-1, :]  # horizontal gradient (edges in x)
        gy = img[:, 1:, :, :] - img[:, :-1, :, :]  # vertical   gradient (edges in y)
        return gx, gy

    def call(self, inputs):
        a, b = inputs  # a = x1, b = x2 (the two separated X-ray layers)
        eps = 1e-8
        total = 0.0
        for _ in range(self.levels):
            gxa, gya = self._gradients(a)
            gxb, gyb = self._gradients(b)
            # balance factors so one globally-stronger layer doesn't dominate the product
            ax = 2.0 * ops.mean(ops.abs(gxa)) / (ops.mean(ops.abs(gxb)) + eps)
            ay = 2.0 * ops.mean(ops.abs(gya)) / (ops.mean(ops.abs(gyb)) + eps)
            # squash each gradient into (-1, 1): we care about edge PRESENCE, not magnitude
            gxa_s = 2.0 * ops.sigmoid(gxa) - 1.0
            gya_s = 2.0 * ops.sigmoid(gya) - 1.0
            gxb_s = 2.0 * ops.sigmoid(ax * gxb) - 1.0
            gyb_s = 2.0 * ops.sigmoid(ay * gyb) - 1.0
            # per-sample overlap energy: large only where BOTH layers have an edge
            px = ops.mean(ops.square(gxa_s) * ops.square(gxb_s), axis=[1, 2, 3])
            py = ops.mean(ops.square(gya_s) * ops.square(gyb_s), axis=[1, 2, 3])
            total = total + px + py  # accumulate over x/y directions and over scales
            # halve the resolution for the next (coarser) scale
            a = ops.average_pool(a, pool_size=2, strides=2, padding="valid")
            b = ops.average_pool(b, pool_size=2, strides=2, padding="valid")
        return ops.reshape(total, (-1, 1))  # one nonnegative value per sample -> (batch, 1)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], 1)  # (batch, 1)


def exclusion_loss(y_true, y_pred):
    # y_pred already IS the per-sample exclusion energy from GradientExclusion;
    # the target y_true is an ignored dummy of zeros, so we just average y_pred.
    return ops.mean(y_pred, axis=-1)


####################################################################
##    define networks
####################################################################
kernel_dim = 5  # convolution kernels are 5x5
filters = 128  # each conv layer produces 128 feature maps (the code has 128 channels)
number_layers = 6  # number of unrolled LISTA iterations in the X-ray encoder


def Encoder_x():
    # LISTA encoder for the X-ray: unrolls 6 iterations of sparse coding to turn
    # a patch into a sparse 128-channel "code".

    input_ex = Input(shape=input_shape, name="InputEx")  # input placeholder for a 50x50x1 patch
    B = We_Conv_layer(  # B = W_e * input : project the image into code space
        kernel_size=kernel_dim, filters=filters, use_bias=False, padding="same"
    )(input_ex)  # "same" padding keeps the 50x50 spatial size

    # initialise code
    code = Proximal_Conv_Operator(units=filters)(B)  # first sparse code = soft-threshold(B)

    # start iterations  (unrolled LISTA: code <- Prox(B + S*code))
    for _ in range(number_layers - 1):  # repeat the refinement 5 more times (total 6)
        C = S_Conv_layer(  # C = S * current_code : recurrent refinement term
            kernel_size=kernel_dim, filters=filters, use_bias=False, padding="same"
        )(code)
        E = Add()([B, C])  # E = B + C  (fixed input term + refinement term)
        code = Proximal_Conv_Operator(units=filters)(E)  # re-apply shrinkage to get the next sparse code

    E_x = Model(inputs=[input_ex], outputs=code)  # wrap the whole unrolled encoder as a reusable Model

    return E_x  # returns the X-ray encoder model


def Encoder_r(threshold_seed=None):
    # LISTA encoder for the reference photos. Same design as Encoder_x but with
    # fewer unrolled iterations (3). Each reference now gets its OWN instance
    # (independent weights) so image 1 and image 2 can use different thresholds.
    number_layers = 3  # local override: only 3 LISTA iterations here
    if threshold_seed is None:
        threshold_seed = threshold_init  # fall back to the global default seed
    prox_init = initializers.Constant(threshold_seed)  # this encoder's soft-threshold seed
    input_er = Input(shape=input_shape, name="InputEr")  # input placeholder for a 50x50x1 reference patch
    B = We_Conv_layer(  # B = W_e * input : project reference patch into code space
        kernel_size=kernel_dim, filters=filters, use_bias=False, padding="same"
    )(input_er)

    # initialise code
    code = Proximal_Conv_Operator(units=filters, threshold_initializer=prox_init)(B)  # first sparse code = soft-threshold(B)

    # start iterations
    for _ in range(number_layers - 1):  # 2 more refinement steps (total 3)
        C = S_Conv_layer(  # C = S * current_code
            kernel_size=kernel_dim, filters=filters, use_bias=False, padding="same"
        )(code)
        E = Add()([B, C])  # E = B + C
        code = Proximal_Conv_Operator(units=filters, threshold_initializer=prox_init)(E)  # shrink to next sparse code

    E_r = Model(inputs=[input_er], outputs=code)  # wrap as a reusable reference-encoder Model

    return E_r  # returns the reference encoder model


def Decoder_r():
    # Decoder that reconstructs a reference-image patch from its 128-channel code
    # (this is the learned "dictionary" for the reference domain).
    input_dr = Input(shape=(input_shape[0], input_shape[1], filters), name="InputDr")  # code input: 50x50x128
    x = Conv2D(  # single conv maps 128 code channels back to the 1-channel image
        kernel_size=kernel_dim, filters=input_shape[-1], use_bias=False, padding="same"
    )(input_dr)
    D_r = Model(inputs=[input_dr], outputs=x, name = "reconstruction_decoder")  # wrap as reference-decoder Model
    return D_r  # returns the reference decoder


def Decoder_x():
    # Decoder that reconstructs an X-ray patch from a 128-channel code
    # (the learned "dictionary" for the X-ray domain).
    input_dx = Input(shape=(input_shape[0], input_shape[1], filters), name="InputDx")  # code input: 50x50x128
    x = Conv2D(  # single conv maps 128 code channels back to the 1-channel X-ray
        kernel_size=kernel_dim, filters=input_shape[-1], use_bias=False, padding="same"
    )(input_dx)
    D_x = Model(inputs=[input_dx], outputs=x, name = "xray_decoder")  # wrap as X-ray-decoder Model
    return D_x  # returns the X-ray decoder


####################################################################
##    traing the whole autoencoder networks
####################################################################

ze = np.zeros(shape=(samples, 1))  # a zeros array (declared here but not actually used below)


input_er1 = Input(shape=input_shape, name="input1")  # network input for reference-1 patches (g1)
input_er2 = Input(shape=input_shape, name="input2")  # network input for reference-2 patches (g2)
input_x = Input(shape=input_shape, name="input3")  # network input for X-ray patches


Er1 = Encoder_r(threshold_init_r1)  # reference-1 encoder (its OWN weights + low seed 0.15; fixes image-1 collapse)
Er2 = Encoder_r(threshold_init_r2)  # reference-2 encoder (its own weights + higher seed 0.30)
Ex = Encoder_x()  # instantiate the X-ray encoder
Dr = Decoder_r()  # instantiate the reference decoder
Dx = Decoder_x()  # instantiate the X-ray decoder

f1 = Er1(input_er1)  # code of reference-1 (now from its own encoder Er1)
f2 = Er2(input_er2)  # code of reference-2 (now from its own encoder Er2)
ff = Ex(input_x)  # code of the X-ray (computed here; note: not used in the losses below)
f = Add()([f1, f2])  # combined code = code1 + code2 (both layers present in the X-ray)
r1 = Activation("linear", name="r1")(Dr(f1))  # reconstruct reference-1 image from its code
r2 = Activation("linear", name="r2")(Dr(f2))  # reconstruct reference-2 image from its code
x = Dx(ff)  # reconstruct the X-ray from the code of its x-ray
x1 = Dx(f1)  # X-ray *contribution* of layer 1 (decode ref-1's code into the X-ray domain)
x2 = Dx(f2)  # X-ray *contribution* of layer 2 (decode ref-2's code into the X-ray domain)
xx1 = Add()([x1, x2])  # x1 + x2
xx2 = Multiply()([x1, x2])  # x1 * x2 (element-wise)
xr = Subtract(name = "xr_x1_x2")([xx1, xx2])  # xr = x1 + x2 - x1*x2 -> "screen-blend" model of how two transparent X-ray layers superimpose
excl = GradientExclusion(levels=3, name="excl")([x1, x2])  # 5th term: penalise x1 & x2 for sharing edges (target is a dummy 0)

autoencoder = Model(inputs=[input_er1, input_er2, input_x], outputs=[r1, r2, x, xr, excl])  # full multi-input/multi-output model
autoencoder.compile(  # configure training
    optimizer=Adam,
    loss=["mse", "mse", "mse", "mse", exclusion_loss],
    loss_weights=[1, 1, 4, 10, excl_weight],
    # 5 loss terms, each weighted differently:
    #   r1 vs g1   (weight 1)           - reconstruct reference 1 from its own code
    #   r2 vs g2   (weight 1)           - reconstruct reference 2 from its own code
    #   x  vs xray (weight 4)           - reconstruct the X-ray from the combined code
    #   xr vs xray (weight 10)           - reconstruct the X-ray from the nonlinear mix of the separated layers (weighted highest)
    #   excl       (weight excl_weight) - gradient exclusion: x1 & x2 should NOT have edges in the same place (target is a dummy 0)
)
autoencoder.fit(  # train the network by gradient descent
    [g1_cube, g2_cube, x_cube],  # inputs: ref-1 patches, ref-2 patches, X-ray patches
    [g1_cube, g2_cube, x_cube, x_cube, ze],  # targets for the 5 outputs (r1->g1, r2->g2, x->xray, xr->xray, excl->0 dummy)
    epochs=iteration_number,  # number of passes over the data (20 here)
    batch_size=128,  # patches per gradient update
    shuffle=True,  # shuffle patch order each epoch
)

####################################################################
##    Report the 4 reconstruction MSEs (one per output)
####################################################################

# One clean pass over the full patch set. evaluate() returns the values in the
# same order as the model outputs [r1, r2, x, xr]:
#   mse[0] = total weighted loss, mse[1..4] = the four unweighted MSEs.
mse = autoencoder.evaluate(
    [g1_cube, g2_cube, x_cube],  # inputs
    [g1_cube, g2_cube, x_cube, x_cube, ze],  # targets (r1->g1, r2->g2, x->xray, xr->xray, excl->0)
    verbose=0,  # don't draw a progress bar
)
print(f"reconstruction of black and white image 1 (r1 vs g1) mse: {mse[1]}")
print(f"reconstruction of black and white image 2 (r2 vs g2) mse: {mse[2]}")
print(f"reconstruction of x-ray from its own code (x vs xray) mse: {mse[3]}")
print(f"recombined x-ray x1+x2-x1*x2 (xr vs xray) mse: {mse[4]}")
print(f"gradient exclusion between x1 and x2 (lower = cleaner split): {mse[5]}")

####################################################################
##    Sparsity diagnostics: did the LISTA codes actually go sparse?
####################################################################


# (1) Learned soft-threshold cutoffs. In sign(x)*max(|x|-threshold, 0) a
#     threshold <= 0 never clips anything (|x|-threshold >= |x|), so that
#     channel gets NO shrinkage and stays dense.
def gather_thresholds(encoder):
    thresholds = [
        layer.get_weights()[0]  # per-channel threshold vector of one proximal layer
        for layer in encoder.layers
        if isinstance(layer, Proximal_Conv_Operator)
    ]
    return np.concatenate(thresholds) if thresholds else np.array([])


print("\n================ SPARSITY DIAGNOSTICS ================")
print("Learned thresholds (higher = more shrinkage; <= 0 = none):")
for name, enc in [("reference-1 encoder Er1", Er1), ("reference-2 encoder Er2", Er2), ("x-ray encoder      Ex", Ex)]:
    t = gather_thresholds(enc)  # all thresholds in this encoder, flattened
    nonpos = int(np.sum(t <= 0))  # how many apply no shrinkage at all
    print(
        f"  {name}: {t.size} thresholds, {nonpos} <= 0 ({100 * nonpos / t.size:.1f}%), "
        f"min={t.min():.3f} mean={t.mean():.3f} max={t.max():.3f}"
    )

# (2) Actual code occupancy. Each code is (patches, 50, 50, 128) ~= 1.3 MB per
#     patch, so the full stack would be ~12 GB; measure on a subset to stay sane.
n_sample = min(200, samples)  # lower this if Colab runs out of RAM
code_model = Model(inputs=[input_er1, input_er2, input_x], outputs=[f1, f2, ff])  # expose the sparse codes
c1, c2, cx = code_model.predict(
    [g1_cube[:n_sample], g2_cube[:n_sample], x_cube[:n_sample]]  # only the first n_sample patches
)
print(f"Code occupancy on {n_sample} sample patches (a sparse code should be well under ~20% active):")
for name, c in [("reference-1 code f1", c1), ("reference-2 code f2", c2), ("x-ray code ff     ", cx)]:
    active = float(np.mean(np.abs(c) > 1e-6))  # fraction that are not (near) zero (nominal occupancy)
    strong = float(np.mean(np.abs(c) > 0.01))  # fraction that are meaningfully nonzero (true "working" sparsity)
    print(
        f"  {name}: {100 * active:5.2f}% active (|a|>1e-6), {100 * strong:5.2f}% strong (|a|>0.01), "
        f"mean|a|={np.mean(np.abs(c)):.4f}"
    )
print("======================================================\n")

Fx = Model(inputs=[input_er1, input_er2], outputs=[x1, x2])  # sub-model: from the two references, output the two separated X-ray layers
[x1_pre_cube, x2_pre_cube] = Fx.predict([g1_cube, g2_cube])  # run it to get the separated X-ray patch stacks

####################################################################
##    Image recovery
####################################################################


x1_pre = creat_image(x1_pre_cube, m, n, distance, distance, 1)  # stitch layer-1 patches back into a full image
x2_pre = creat_image(x2_pre_cube, m, n, distance, distance, 1)  # stitch layer-2 patches back into a full image

# Diagnose the intermittent "all-white" layer. White = values saturating to >= 1
# (they get clipped to white on save). Print the raw range of each separated layer.
for _name, _arr in [("x1_pre", x1_pre), ("x2_pre", x2_pre)]:
    print(
        f"{_name}: min={np.nanmin(_arr):.3f} max={np.nanmax(_arr):.3f} "
        f"mean={np.nanmean(_arr):.3f} frac>=1={np.mean(_arr >= 1):.1%} "
        f"has_nan={np.isnan(_arr).any()}"
    )


def to_uint8(img):
    # JPEG only accepts 8-bit integer pixels, but creat_image returns floats in ~[0,1].
    img = np.clip(img, 0, 1)  # clamp any slight over/undershoot from the network into [0,1]
    return (img * 255).round().astype(np.uint8)  # scale to 0..255 and cast to uint8


imsave("x1_pre.jpg", to_uint8(x1_pre))  # save the separated layer-1 X-ray to disk
imsave("x2_pre.jpg", to_uint8(x2_pre))  # save the separated layer-2 X-ray to disk


####################################################################
##    Visualise the pipeline (side-by-side comparisons)
####################################################################

# Reconstructions of the two reference photos: r1, r2 are the reference-decoder
# outputs of the trained model. Build a small sub-model that maps the two
# reference inputs to those reconstructions, run it, and stitch the patch
# stacks back into full images.
Rr = Model(inputs=[input_er1, input_er2], outputs=[r1, r2])  # sub-model: references -> their reconstructions
[r1_pre_cube, r2_pre_cube] = Rr.predict([g1_cube, g2_cube])  # reconstruct every patch
r1_pre = creat_image(r1_pre_cube, m, n, distance, distance, 1)  # stitch ref-1 reconstruction into a full image
r2_pre = creat_image(r2_pre_cube, m, n, distance, distance, 1)  # stitch ref-2 reconstruction into a full image

# Recombine the two separated X-ray layers with the same screen-blend the
# network was trained on: xr = x1 + x2 - x1*x2.
xr_pre = x1_pre + x2_pre - x1_pre * x2_pre  # combined X-ray rebuilt from the separated layers


def show_pair(left, right, left_title, right_title, left_vlim=(0, 1), right_vlim=(0, 1)):
    # Display two grayscale images side by side (renders inline in Colab/Jupyter).
    # *_vlim fix each panel's grayscale range; pass None to auto-scale a panel to
    # its own min/max (used for the sparse-code map, which is not in [0, 1]).
    lv = left_vlim or (None, None)  # (vmin, vmax) for the left panel
    rv = right_vlim or (None, None)  # (vmin, vmax) for the right panel
    _, axes = plt.subplots(1, 2, figsize=(12, 6))  # one row, two columns
    axes[0].imshow(np.squeeze(left), cmap="gray", vmin=lv[0], vmax=lv[1])  # left image
    axes[0].set_title(left_title)  # label the left panel
    axes[0].axis("off")  # hide the pixel-index ticks
    axes[1].imshow(np.squeeze(right), cmap="gray", vmin=rv[0], vmax=rv[1])  # right image
    axes[1].set_title(right_title)  # label the right panel
    axes[1].axis("off")
    plt.tight_layout()  # avoid overlapping titles/panels
    plt.show()  # render the figure


show_pair(g1, r1_pre, "Reference 1 (grayscale original)", "Reconstruction (r1)")  # ref-1 vs its reconstruction
show_pair(g2, r2_pre, "Reference 2 (grayscale original)", "Reconstruction (r2)")  # ref-2 vs its reconstruction
show_pair(xray, xr_pre, "Original mixed X-ray", "Recombined X-ray (x1 + x2 - x1*x2)")  # true X-ray vs recombination

# --- Original X-ray vs the autoencoder's reconstruction of it ---------------
# x = Dx(Ex(xray)): encode the X-ray into its sparse code, then decode back to a
# 1-channel image (this is exactly what xray_decoder_loss scores). predict()
# returns only the 1-channel reconstruction; the 128-channel code is computed
# per batch internally and never materialised in full.
x_ae = Model(inputs=[input_x], outputs=x)  # the X-ray autoencoder branch as a standalone model
x_recon_cube = x_ae.predict(x_cube)  # (samples, 50, 50, 1) reconstructed X-ray patches
x_recon = creat_image(x_recon_cube, m, n, distance, distance, 1)  # stitch into a full image
show_pair(xray, x_recon, "Original mixed X-ray", "Reconstructed X-ray  Dx(Ex(xray))")  # original vs its reconstruction
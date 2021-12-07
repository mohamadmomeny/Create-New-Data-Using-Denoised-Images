"""
******************************************************************************
An Autoencoder Model to Create New Data Using Denoised Images Corrupted by 
the Speckle, Gaussian, Poisson, and impulse Noise
******************************************************************************

The images, corrupted by the Speckle, Gaussian, Poisson, and impulse Noise, 
can be restored by enhancement approaches such as deep autoencoder networks. 
However, depending on the noise type and density, the pixel values in the 
restored image and the original noise-free image are not exactly equal. 
We aim to leverage the dissimilarity between restored and original pixels 
as a data augmentation strategy. First, noise of specific type and density 
is added to the image. Then, the noise is partially removed from the image 
by using the proposed autoencoder. The denoising autoencoder aims to produce 
the output from the noisy input, where the target is set as the original images. 
Finally, the restored images are used as augmented data.
"""

IMAGE_PATH = 'E:/Sel/Matlab Code/Dataset_227_227_3/Train/P/'  #The path of the original dataset
New_Data_Path='E:/Sel/Matlab Code/Dataset_227_227_3/Train/P/' #The path of the new dataset

noiseType='speckle' # Or another noise, 'gaussian', 'poisson', 's&p'
"""
One of the following strings, selecting the type of noise to add:
- 'gaussian'  Gaussian-distributed additive noise.
- 'poisson'   Poisson-distributed noise generated from the data.
- 's&p'       Replaces random pixels with either 1 or `low_val`, where
              `low_val` is 0 for unsigned images or -1 for signed
              images.
- 'speckle'   Multiplicative noise using out = image + n*image, where
              n is Gaussian noise with specified mean & variance.
"""

mean= 0.0 # Gaussian and Speckle noise
var= 0.01 # Gaussian and Speckle noise
amount= 0.05 # Impulse noise
salt_vs_pepper= 0.5 # Impulse noise

"""
mean : float, optional
    Mean of random distribution. Used in 'gaussian' and 'speckle'.
    Default : 0.
var : float, optional
    Variance of random distribution. Used in 'gaussian' and 'speckle'.
    Note: variance = (standard deviation) ** 2. Default : 0.01
amount : float, optional
    Proportion of image pixels to replace with noise on range [0, 1].
    Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
salt_vs_pepper : float, optional
    Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
    Higher values represent more salt. Default : 0.5 (equal amounts)
"""
epochs=20
batch_size=100

IMG_HEIGHT = 224  # Image height
IMG_WIDTH = 224 # Image width
IMG_CHANNELS = 3 # Image channel

"""
You can run!
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import sys
import glob
import random
from tqdm import tqdm
from PIL import Image
from keras.preprocessing import image
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage.viewer import ImageViewer
from skimage.transform import resize
import random
import cv2

__all__ = ['random_noise']

def _bernoulli(p, shape, *, random_state):
    """
    Bernoulli trials at a given probability of a given size.
    This function is meant as a lower-memory alternative to calls such as
    `np.random.choice([True, False], size=image.shape, p=[p, 1-p])`.
    While `np.random.choice` can handle many classes, for the 2-class case
    (Bernoulli trials), this function is much more efficient.
    Parameters
    ----------
    p : float
        The probability that any given trial returns `True`.
    shape : int or tuple of ints
        The shape of the ndarray to return.
    seed : `numpy.random.Generator`
        ``Generator`` instance.
    Returns
    -------
    out : ndarray[bool]
        The results of Bernoulli trials in the given `size` where success
        occurs with probability `p`.
    """
    if p == 0:
        return np.zeros(shape, dtype=bool)
    if p == 1:
        return np.ones(shape, dtype=bool)
    return random_state.random(shape) <= p
def random_noise(image, mode='s&p', seed=None, clip=True, **kwargs):
    """
    Function to add random noise of various types to a floating-point image.
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str, optional
        One of the following strings, selecting the type of noise to add:
        - 'gaussian'  Gaussian-distributed additive noise.
        - 'localvar'  Gaussian-distributed additive noise, with specified
                      local variance at each point of `image`.
        - 'poisson'   Poisson-distributed noise generated from the data.
        - 'salt'      Replaces random pixels with 1.
        - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                      -1 (for signed images).
        - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                      `low_val` is 0 for unsigned images or -1 for signed
                      images.
        - 'speckle'   Multiplicative noise using out = image + n*image, where
                      n is Gaussian noise with specified mean & variance.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is
        used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that
        instance is used.
        This will set the random seed before generating noise,
        for valid pseudo-random comparisons.
    clip : bool, optional
        If True (default), the output will be clipped after noise applied
        for modes `'speckle'`, `'poisson'`, and `'gaussian'`. This is
        needed to maintain the proper image data range. If False, clipping
        is not applied, and the output may extend beyond the range [-1, 1].
    mean : float, optional
        Mean of random distribution. Used in 'gaussian' and 'speckle'.
        Default : 0.
    var : float, optional
        Variance of random distribution. Used in 'gaussian' and 'speckle'.
        Note: variance = (standard deviation) ** 2. Default : 0.01
    local_vars : ndarray, optional
        Array of positive floats, same shape as `image`, defining the local
        variance at every image point. Used in 'localvar'.
    amount : float, optional
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
    salt_vs_pepper : float, optional
        Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
        Higher values represent more salt. Default : 0.5 (equal amounts)
    Returns
    -------
    out : ndarray
        Output floating-point image data on range [0, 1] or [-1, 1] if the
        input `image` was unsigned or signed, respectively.
    Notes
    -----
    Speckle, Poisson, Localvar, and Gaussian noise may generate noise outside
    the valid image range. The default is to clip (not alias) these values,
    but they may be preserved by setting `clip=False`. Note that in this case
    the output may contain values outside the ranges [0, 1] or [-1, 1].
    Use this option with care.
    Because of the prevalence of exclusively positive floating-point images in
    intermediate calculations, it is not possible to intuit if an input is
    signed based on dtype alone. Instead, negative values are explicitly
    searched for. Only if found does this function assume signed input.
    Unexpected results only occur in rare, poorly exposes cases (e.g. if all
    values are above 50 percent gray in a signed `image`). In this event,
    manually scaling the input to the positive domain will solve the problem.
    The Poisson distribution is only defined for positive integers. To apply
    this noise type, the number of unique values in the image is found and
    the next round power of two is used to scale up the floating-point result,
    after which it is scaled back down to the floating-point image range.
    To generate Poisson noise against a signed image, the signed image is
    temporarily converted to an unsigned image in the floating point domain,
    Poisson noise is generated, then it is returned to the original range.
    """
    mode = mode.lower()

    # Detect if a signed image was input
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    image = preprocess(image)

    rng = np.random.default_rng(seed)

    allowedtypes = {
        'gaussian': 'gaussian_values',
        'localvar': 'localvar_values',
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'mean': 0.,
        'var': 0.1,
        'amount': 0.05,
        'salt_vs_pepper': 0.5,
        'local_vars': np.zeros_like(image) + 0.01}

    allowedkwargs = {
        'gaussian_values': ['mean', 'var'],
        'localvar_values': ['local_vars'],
        'sp_values': ['amount'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode == 'gaussian':
        noise = rng.normal(kwargs['mean'], kwargs['var'] ** 0.5, image.shape)
        out = image + noise

    elif mode == 'localvar':
        # Ensure local variance input is correct
        if (kwargs['local_vars'] <= 0).any():
            raise ValueError('All values of `local_vars` must be > 0.')

        # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
        out = image + rng.normal(0, kwargs['local_vars'] ** 0.5)

    elif mode == 'poisson':
        # Determine unique values in image & calculate the next power of two
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))

        # Ensure image is exclusively positive
        if low_clip == -1.:
            old_max = image.max()
            image = (image + 1.) / (old_max + 1.)

        # Generating noise for each unique value in image.
        out = rng.poisson(image * vals) / float(vals)

        # Return image to original range if input was signed
        if low_clip == -1.:
            out = out * (old_max + 1.) - 1.

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = random_noise(image, mode='s&p', seed=rng,
                           amount=kwargs['amount'], salt_vs_pepper=1.)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = random_noise(image, mode='s&p', seed=rng,
                           amount=kwargs['amount'], salt_vs_pepper=0.)

    elif mode == 's&p':
        out = image.copy()
        p = kwargs['amount']
        q = kwargs['salt_vs_pepper']
        flipped = _bernoulli(p, image.shape, random_state=rng)
        salted = _bernoulli(q, image.shape, random_state=rng)
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = low_clip

    elif mode == 'speckle':
        noise = rng.normal(kwargs['mean'], kwargs['var'] ** 0.5, image.shape)
        out = image + image * noise

    # Clip back to original range, if necessary
    if clip:
        out = np.clip(out, low_clip, 1.0)
    out=out*255
    out=out.astype("uint8")
    return out


def preprocess(array):
    array = array.astype("float32")/255
    return array
   
IMG_Dataset = next(os.walk(IMAGE_PATH))[2]

Inputs = np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),dtype = np.uint8)
noisy_Inputs= np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),dtype = np.uint8)
for n, f in tqdm(enumerate(IMG_Dataset), total = len(IMG_Dataset)):
    Images = imread (IMAGE_PATH + f)[:,:,:IMG_CHANNELS]
    Images=resize(Images,(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    Images=Images*255
    Images=Images.astype("uint8")
    Inputs[n] = Images
    
    # imsave((IMAGE_PATH + f+'_1.tif'), Images)
    if noiseType=='gaussian':
        Images=random_noise(Images,noiseType, mean= mean, var= var)
    elif noiseType=='poisson':
        Images=random_noise(Images,noiseType,)
    elif noiseType=='s&p':
        Images=random_noise(Images,noiseType, amount= amount, salt_vs_pepper= salt_vs_pepper)
    elif noiseType=='speckle':
        Images=random_noise(Images,noiseType, mean= mean, var= var)
    
    
    imsave((New_Data_Path + f+'_noisy_.tif'), Images)
    noisy_Inputs[n] = Images


input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(IMG_CHANNELS, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

noisy_Inputs=preprocess(noisy_Inputs)
Inputs=preprocess(Inputs)
    
autoencoder.fit(
    x=noisy_Inputs,
    y=Inputs,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
)

predictions = autoencoder.predict(noisy_Inputs)
for n, f in tqdm(enumerate(IMG_Dataset), total = len(IMG_Dataset)):
    out=predictions[n]*255;
    out=out.astype("uint8")
    imsave((New_Data_Path + f+'_restored.tif'), out)

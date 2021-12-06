# An Autoencoder Model to Create New Data Using Denoised Images Corrupted by the Speckle, Gaussian, poisson, and Impulse Noise
Noisy images can be restored by enhancement algorithms such as autoencoder networks. However, depending on the noise type and density, the pixel values in the restored image and the original noise-free image are not exactly equal. We aim to leverage the dissimilarity between restored and original pixels as a data augmentation strategy. First, noise of specific type and density is added to the image. Then, the noise is partially removed from the image by using the proposed autoencoder. The denoising autoencoder aims to produce the output from the noisy input, where the target is set as the original images. Finally, the restored images are used as augmented data. 

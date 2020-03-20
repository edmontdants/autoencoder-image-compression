# Autoencoder-based image compression on Kodak dataset (University project)

## Background

The most common and promising direction of research in
the field of deep learning -based image compression has
focused on training deep neural networks as image
autoencoders. Autoencoders are self-supervised models
which output an approximation of a provided input by
transforming the input into a code vector of as small length
as possible, often referred to as a latent representation of
the input data. A typical autoencoder consists of an
encoder and decoder, which are trained by using a loss
function that tries to find an optimal balance between
distortion and compression rate.

## Dataset

The Kodak dataset aws used to conduct
training. A total of 16 out of 24 images with resolution of
768x512 pixels was taken from the Kodak dataset. Images
are color 24 bits per pixel uncompressed and have PNG
format.

## Results

The work considered such indicators as PSNR and
compression ratio. Average PSNR achieved was 30 dB.
Mean PNG (lossless compression) input/output file size
reduction by 35 percent. Compression ratios for the images
varied between 1.50 and 1.55 depending on an image.

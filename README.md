# Devanagari-CAE
Devanagari analysis with autoencoder approach

These scripts demonstrate the use of autoencoder for analysing pictures.

Here the technique is applyed to Devanagari handwritten letters and digits.

The dataset is composed of 36 consonants and 10 digits with a total of 92000 pictures of size 32 x 32 pixels.

First step is to compute a "mean" picture of each letter/digit.

Then we choose a picture from the training dataset to be a good average model for each letter/digit.

We use those models to train an autoencoder to match letters/digits to their model.

We then use the encoding part of the autoencoder to compute encoded representations of the pictures in the dataset.

After that, we can train the bottom dense layers classifying the encoded representation of the pictures.

We can then compile our model and test it on the test dataset to observe good training and good accuracy.

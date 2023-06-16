# NameGenerator

A multi-layered perceptron language model that generates name-like sequences given a training set of names. For example, feeding it a text files of 10k English names, it can generate sequences that are totally novel but similar to those names.

## Technologies

- Implemented a embedding layer in the beginning and a softmax layer at the end to represent tokens and calculate the probability for any given token to be generated next. 

- Improved the model's performance by designing multiple dilated hidden layers that, instead of being fully connected to the input layer, take inputs in batches of two tokens to learn the weights more gradually.

- Improved the backward pass quality of the model by implementing batch normalizations and non-linear activations, making sure the weights are Gaussian-distrbuted.

## Performance

- Trained for 200000 iterations on a V100 GPU, decreasing the training loss and validation loss to around 2.0.

- Can generate names that look a lot like genuine English names, such as:

> hicon
> frayx
> jakoo
> tefya
> rastin

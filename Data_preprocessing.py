from keras.layers import TextVectorization, Normalization, CenterCrop, Rescaling
import numpy as np
from matplotlib import pyplot as plt

# TextVectorization is used to Vectorize textual data
my_data = np.array([['This is my text'], ['This is the Next sentence']])
vectorizer = TextVectorization(output_mode='int')
# Calling `adapt` on an array or dataset makes the layer generate a vocabulary
# index for the data, which can then be reused when seeing new data.
vectorizer.adapt(my_data)
# get_vocabulary prints the vocabulary of words in the corpus
print(vectorizer.get_vocabulary())
# After calling adapt, the layer is able to encode any n-gram it has seen before
# in the `adapt()` data. Unknown n-grams are encoded via an "out-of-vocabulary"
# token.
vector_data = vectorizer(my_data)
print(vector_data)

# To get to a one-hot encoded bi-grams
vectorizer = TextVectorization(output_mode='binary', ngrams=2)
vectorizer.adapt(my_data)
print(vectorizer.get_vocabulary())
vector_data = vectorizer(my_data)
print(vector_data)

# Normalization of data
my_data = np.array([6, 8, 4, 3, 2, 1])
normalizer = Normalization(axis=None)
# The adapt method calculates the mean and variance as part of the input data
normalizer.adapt(my_data)
print('Mean ', normalizer.mean, 'Variance ', normalizer.variance)
normalized_data = normalizer(my_data)
print(normalized_data)
print('Mean of Normalized data ', int(np.mean(normalized_data)), 'Variance of Normalized data ', int(np.var(normalized_data)))

# Croping and Rescaling of images

# Example image data, with values in the [0, 255] range
# size parameter = (Batch_size, Image_width, Image_height, RGB)
my_image = np.random.randint(0, 256, size=(1, 200, 200, 3)).astype("float32")
print('Shape of image ', my_image.shape)
plt.subplot(1,2,1)
plt.imshow(my_image[0])
cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)
output_data = scaler(cropper(my_image))
plt.subplot(1,2,2)
plt.imshow(output_data[0])
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
plt.show()

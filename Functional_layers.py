from keras import layers
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

my_image = Image.open('d:/guru/guru.jpg')
#my_image = Image.open('d:/pythondata/num_3.png')
my_image = np.array(my_image)
plt.subplot(2,4,1)
plt.imshow(my_image)
plt.title('Original')
print('Original Image size ', my_image.shape)
input_shape = (1,640, 640,3)
temp_array = my_image.reshape((input_shape))

# Cropping2D 
x = layers.Cropping2D(cropping=((2,2)))(temp_array)
print('Cropped Size ', x.shape)
a, b, c, d = x.shape
plt.subplot(2,4,2)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array)
plt.title('Cropped')

# Rescale images to [0, 1]
x = layers.Rescaling(scale=1.0 / 255)(x)
plt.subplot(2,4,3)
a, b, c, d = x.shape
print('Rescale size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Rescaled')

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=3, kernel_size=(2, 2), activation="relu", padding='same', strides=2)(x)
plt.subplot(2,4,4)
a, b, c, d = x.shape
print('Convolution 1 size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Convoluted 1')

x = layers.MaxPooling2D(pool_size=(2, 2))(x)
plt.subplot(2,4,5)
a, b, c, d = x.shape
print('Maxpool 1 size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Maxpool 1')

x = layers.Conv2D(filters=3, kernel_size=(2, 2), activation="relu")(x)
plt.subplot(2,4,6)
a, b, c, d = x.shape
print('Convolution 2 size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Convoluted 2')

x = layers.AveragePooling2D(pool_size=(2, 2))(x)
plt.subplot(2,4,7)
a, b, c, d = x.shape
print('Average Pooling size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Avgpool')

x = layers.Conv2D(filters=1, kernel_size=(2, 2), activation="relu")(x)
plt.subplot(2,4,8)
a, b, c, d = x.shape
print('Convolution 3 size ', x.shape)
temp_array = np.array(x)
temp_array = temp_array.reshape((b,c,d))
plt.imshow(temp_array, norm='linear')
plt.title('Convoluted 3')
plt.show()


# Maximum / Minimum - used to find the element-wise Max / Min of tensors
a = [2,4,1,4,5]
a = np.array(a)
a = a.reshape(5,1)
b = [3,5,2,1,4]
b = np.array(b)
b = b.reshape(5,1)
c = [1,3,5,7,9]
c = np.array(c)
c = c.reshape(5,1)
d = [2,3,4,6,7]
d = np.array(d)
d = d.reshape(5,1)
x = layers.Maximum()([a,b,c,d])
print('The Max Tensor is ', x)

x = layers.Minimum()([a,b,c,d])
print('The Min Tensor is ', x)

# Elementwise Multiplication - not the dot product Matrix multiplication
a = [[1,2,3],[4,5,6],[7,8,9]]
b = [[2,1,1],[1,3,1],[1,1,4]]
a = np.array(a)
b = np.array(b)
x = layers.Multiply()([a,b])
print('Product of a and b is ', x)

# Dot product Matrix multiplication
a = np.arange(4).reshape(2,2)
b = np.arange(4).reshape(2,2)
print('Input Matrix ', a)
print('Matrix Multiplication \n', a * b)
print('Numpy Matrix Multiplication \n', np.matmul(a,b))
print('Numpy Tensor Multiplication ', np.tensordot(a,b,axes=1))
x = layers.Dot(axes=1)([a,b])
print('Dot Product of a and b is ', x)

# Flatten takes a tensor of size (batch_size, x, y) and converts it to (batch_size, x*y)
a = [[1,2,3,4,5,6,7,8,9,10,11,12]]
a = np.array(a)
a = a.reshape(2,2,3)
a = tf.convert_to_tensor(a)
print('Before Flatten')
print(a)
x = layers.Flatten()(a)
print('Flatten A')
print(x)

# Dropout - to prevent overfitting of data
a = [1.,2.,3.,4.,-5.,6, 0, -2., 3]
a = np.array(a).reshape(3,3)
x = Dropout(.2,input_shape=(3,3))(a, training=True)
print('Before Dropout \n', a)
print('After Dropout \n', x)

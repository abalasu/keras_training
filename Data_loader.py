from keras.utils import image_dataset_from_directory

# Input to Keras can be a Numpy Array, a Keras Dataset (for large data) or data coming from a Python generator (using yield)
# Large amount of data from files can be loaded to a Keras dataset as given below for images. Similar to this there are API's
# for loading Text data from TXT files and Structured data from CSV files

dataset = image_dataset_from_directory('D:/PythonData/num-images', batch_size=64, image_size=(28,28))

print(len(dataset))

i = 1
for data, labels in dataset:
   print(i)    
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
   i+=1
   
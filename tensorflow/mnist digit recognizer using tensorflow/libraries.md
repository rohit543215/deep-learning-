import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
(x_train,y_train),(x_test,y_test)=mnist.load_data()

fig,axes=plt.subplots(2,5,figsize=(12,6))
for i, ax in enumerate(axes.flat):
   ax.imshow(x_train[i],cmap='gray')
   ax.set_title(f"digit: {y_train[i]}")
plt.suptitle('sample mnist digits',fontsize=16)
plt.tight_layout()
plt.show()   
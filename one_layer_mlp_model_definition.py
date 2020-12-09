import tensorflow
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import numpy as np

class MyModel(Model):
	def __init__(self):
	    super(MyModel, self).__init__()
	    self.flatten = Flatten(input_shape=(28,28,1))
	    self.d1 = Dense(64,activation='relu')
	    self.d2 = Dense(10)
#self.d2 = Dense(10, activation='softmax')
	def call(self, x):
	    x = self.flatten(x)
	    x = self.d1(x)
	    return self.d2(x)
		
	def get_middle_result(self, x):
	    x = self.flatten(x)
	    middle_result = self.d1(x)
	    return middle_result
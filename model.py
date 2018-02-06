from keras import Model
from keras.layers import BatchNormalization, Conv2D, Cropping2D, Dense, Flatten, Input, Lambda, MaxPool2D
from keras.utils import plot_model
import numpy as np


class BehavioralDCNN:
    """
    Deep convolutional neural network for behavioral cloning (a type of imitation learning).
    This is an end-to-end model based on NVIDIA's research.
    See: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    """
    def __init__(self):
        self.model = self._create_model()
        
    def _create_model(self):
        # Image size was not reduced to ensure better quality video
        image_input = Input(shape=(160,320,3), dtype=np.float64, name='Input')

        # Preprocess image inputs
        x = Lambda(lambda image: image/255. - 0.5, name='Normalization')(image_input)

        # Convolutions
        x = Conv2D(24, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv1')(x)
        x = Conv2D(36, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv2')(x)
        x = Conv2D(48, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv3')(x)
        x = Conv2D(64, kernel_size=3, padding='valid', activation='relu', name='conv4')(x)
        x = Conv2D(64, kernel_size=3, padding='valid', activation='relu', name='conv5')(x)

        # Flatten convolution
        x = Flatten(name='flatten')(x)

        # Fully-connected layers
        x = Dense(300, activation='relu', name='fc1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dense(50, activation='relu', name='fc3')(x)
        x = Dense(10, activation='relu', name='fc4')(x)
        output = Dense(1, name='output')(x)

        return Model(inputs=image_input, outputs=output)
    
    def draw_model(self, filename):
        plot_model(self.model, to_file=filename)
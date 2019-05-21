from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import backend as K
import os

from glove_loader import GloveModel

class DCGan(object):
    model_name = 'dc-gan'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width  = 7
        self.img_height = 7
        self.img_channels = 1
        self.random_input_dim = 100
        self.text_input_dim = 100
        self.config = None
        self.glove_source_dir_path = './very_large_data'
        self.glove_model = GloveModel()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGan.model_name + '-config.npy')


    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGan.model_name + '-' + model_type + '-weights.h5')


    def create_model(self):
        init_img_width = self.img_width
        init_img_height = self.img_height

        random_input = Input(shape=(self.random_input_dim,))
        text_input1 = Input(shape=(self.text_input_dim,))
        random_dense = Dense(1024)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(128 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height, 128),
                                  input_shape=(128 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, text_input1], generator_output)

        self.generator.compile(loss='mean_squared_error', optimizer='SGD')

        print('generator:', self.generator.summary())

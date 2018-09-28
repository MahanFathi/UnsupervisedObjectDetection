import os
import random
import json
import numpy as np
import tensorflow as tf
from config import Config
from data_util import read_images
from agent import Agent
from scipy.misc import imsave
from tensorflow.python.keras.preprocessing import image


class Settings(object):
    seeding = {
        'personal': 100,
        'for-the-home': 100,
        'electronic-devices': 100,
        'vehicles': 100,
    }
    validation_json = os.path.join(Config.dataset_path, 'validation.json')
    validation_dir = os.path.join(Config.dataset_path, 'validation')


class ValidationSet(object):
    def __init__(self, config):
        self.config = config
        self.agent = Agent(self.config)
        self.settings = Settings

    def make_validation_set(self):
        if not os.path.isdir(self.settings.validation_dir):
            os.mkdir(self.settings.validation_dir)

        the_dictionary = {}
        for cat, count in self.settings.seeding.items():
            cat_path = os.path.join(self.config.dataset_path, cat)
            file_names = random.sample(os.listdir(cat_path), count)
            counter = 0

            while counter < count:
                batch_file_names = file_names[counter:counter + self.config.batch_size]

                # load images and concat
                image_batch = read_images(cat, batch_file_names)
                counter += self.config.batch_size

                # pass through net
                bounding_boxes, class_indices = self.agent.get_bounding_box(image_batch)

                # fill the dictionary and dump occluded image
                for i, name in enumerate(batch_file_names):
                    image_name = cat + '_' + name
                    the_dictionary[image_name] = {
                        'class': str(class_indices[i]),
                        'bounding_box': bounding_boxes[i],
                        'path_to_original': os.path.join(cat_path, name)
                    }
                    # generate and apply mask
                    x1 = np.zeros([self.config.vgg16.input_size[0],
                                   self.config.vgg16.input_size[1],
                                   3])
                    x2 = np.zeros([self.config.vgg16.input_size[0],
                                   self.config.vgg16.input_size[1],
                                   3])
                    x1[:, bounding_boxes[i][0][0]:bounding_boxes[i][1][0], :] = 1.0
                    x2[bounding_boxes[i][0][1]:bounding_boxes[i][1][1], :, :] = 1.0
                    mask = 1.0 * np.logical_and(x1, x2)
                    masked_image = np.multiply(image_batch[i], mask)
                    imsave(os.path.join(self.settings.validation_dir, image_name), masked_image)

        # dump dictionary as json
        with open(self.settings.validation_json, 'w') as validation_dict:
            json.dump(the_dictionary, validation_dict)


def make_predictions(val_net='vgg', cropped=True, name='vgg_cropped'):
    with open(Settings.validation_json) as val_dict:
        validation_dict = json.load(val_dict)

    global model, net
    if val_net is 'vgg':
        net = tf.keras.applications.vgg16
        model = net.VGG16()
    elif val_net is 'res':
        net = tf.keras.applications.resnet50
        model = net.ResNet50()

    for file_name, file_dict in validation_dict.items():  # lazy iteration
        file_path = os.path.join(Settings.validation_dir, file_name) if cropped else file_dict['path_to_original']
        img = image.load_img(file_path, target_size=Config.vgg16.input_size)
        img = image.img_to_array(img)
        prediction = model.predict(np.expand_dims(net.preprocess_input(img), 0)).flatten()
        top_class = np.argmax(prediction)
        validation_dict[file_name][name] = str(top_class)
    tf.keras.backend.clear_session()

    with open(Settings.validation_json, 'w') as val_dict_dump:
        json.dump(validation_dict, val_dict_dump)


def fill_json():
    make_predictions(val_net='vgg', cropped=True, name='vgg_cropped')
    make_predictions(val_net='res', cropped=True, name='res_cropped')
    make_predictions(val_net='res', cropped=False, name='res')


# I don't know how to clear gpu memory so I run these separately
if __name__ == '__main__':
    ValidationSet(Config).make_validation_set()
# if __name__ == '__main__':
#     fill_json()

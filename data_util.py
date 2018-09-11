import os
import random
from lapjv import lapjv
import numpy as np
from tensorflow.python.keras.preprocessing import image
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from config import Config


def sample_data(batch_size, folders=None):
    size_per_cat = batch_size // len(Config.dataset_allowed_folders)
    size = size_per_cat * len(Config.dataset_allowed_folders)
    input_size = Config.vgg16.input_size if Config.resize else Config.default_size
    image_batch = np.empty([size] + list(input_size))
    allowed_folders = Config.dataset_allowed_folders if folders is None else folders
    index = 0
    for cat in allowed_folders:
        cat_path = os.path.join(Config.dataset_path, cat)
        for i, img in enumerate(random.sample(os.listdir(cat_path), size_per_cat)):
            img_path = os.path.join(cat_path, img)
            loaded_image = image.load_img(img_path, target_size=input_size)
            image_batch[index + i] = image.img_to_array(loaded_image)
        index += size_per_cat
    return image_batch


class T_SNE(object):
    def __init__(self):
        self.perplexity = Config.tsne_perplexity
        self.iters = Config.tsne_iter
        self.output_dir = Config.tsne_output_dir
        self.grid_size = Config.grid_size
        self.tile_res = Config.vgg16.input_size[0]
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def generate_tsne(self, embeddings):
        tsne = TSNE(perplexity=self.perplexity, n_components=2, init='random', n_iter=self.iters)
        tsne_embedding = tsne.fit_transform(embeddings.squeeze()[:self.grid_size ** 2, :])
        tsne_embedding -= tsne_embedding.min(axis=0)
        tsne_embedding /= tsne_embedding.max(axis=0)
        return tsne_embedding

    def save_grid(self, images, tsne_embedding, out_name):
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, self.grid_size), np.linspace(0, 1, self.grid_size))).reshape(-1, 2)
        cost_matrix = cdist(grid, tsne_embedding, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)
        grid_jv = grid[col_asses]
        out = np.ones((self.grid_size * self.tile_res, self.grid_size * self.tile_res, 3))

        for pos, img in zip(grid_jv, images[0:self.grid_size ** 2]):
            h_range = int(np.floor(pos[0] * (self.grid_size - 1) * self.tile_res))
            w_range = int(np.floor(pos[1] * (self.grid_size - 1) * self.tile_res))
            out[h_range:h_range + self.tile_res, w_range:w_range + self.tile_res] = image.img_to_array(img)

        im = image.array_to_img(out)
        im.save(self.output_dir + out_name, quality=100)


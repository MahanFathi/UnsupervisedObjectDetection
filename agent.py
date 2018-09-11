import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from data_util import T_SNE, sample_data
from net import Net


class Agent(object):
    """
        recipes for doing the dirty work
    """

    def __init__(self, config):
        self.config = config
        self.net = Net(config)

    def visualize(self, guided_backprops):
        guided_backprops = guided_backprops[-self.config.max_per_row_image_plot:]
        for i, image in enumerate(guided_backprops):
            plt.subplot(1, min([self.config.max_per_row_image_plot, len(guided_backprops)]), i + 1)
            plt.imshow(image.squeeze().astype('float32'))
            plt.axis('off')
        plt.show()

    def get_bounding_box(self, image):
        preprocessed_image = self._preprocess(image)
        kmax_neuron_indices, top_class = self._get_kmax_neurons(preprocessed_image)
        guided_backprops = self._get_guided_backprops(preprocessed_image, kmax_neuron_indices)
        masks = self._get_image_masks(guided_backprops)
        topk_neurons_relative_indices = self._get_topk_neurons(preprocessed_image, masks, top_class)
        bounding_box = self._get_bounding_box(masks, topk_neurons_relative_indices)
        return bounding_box

    def make_tsne_pic_for_folder(self, folder='personal'):
        fc_features = None
        images = None
        images_number = self.config.grid_size ** 2
        batch_count = images_number // self.config.batch_size + 1
        for i in range(batch_count):
            image_batch = sample_data(self.config.batch_size, [folder])
            images = image_batch if images is None else np.concatenate([images, image_batch])
            fc_features_batch = self.net.get_fc_features(image_batch)
            fc_features = fc_features_batch if fc_features is None else np.concatenate([fc_features, fc_features_batch])

        tsne = T_SNE()
        tsne_embedding = tsne.generate_tsne(fc_features)
        tsne.save_grid(images, tsne_embedding, folder+'.jpg')

    def _preprocess(self, image):
        return self.net.vgg16.preprocess_input(image)

    def _get_kmax_neurons(self, image):
        class_scores = self.net.get_class_scores(image)
        n = len(self.config.top_n_classes_weights)
        top_class = np.argmax(class_scores)
        top_classes = np.argsort(class_scores)[-n:]
        top_scores = class_scores[top_classes]
        activations = self.net.get_activations(image)

        normalizer = 1.0 / (np.sum(self.config.top_n_classes_weights) * np.sum(top_scores))

        impact_gradient = 0
        for i, class_index in enumerate(top_classes):
            impact_gradient_per_class = self.net.get_impact_gradient_per_class(activations=activations,
                                                                               class_index=class_index)
            impact_gradient += impact_gradient_per_class * top_scores[i] * \
                               self.config.top_n_classes_weights[i] * normalizer

        rank_score = activations * impact_gradient
        kmax_neurons = np.argsort(rank_score.ravel())[-self.config.kmax:]
        return kmax_neurons, top_class

    def _get_guided_backprops(self, image, neuron_indices):
        guided_backprops = [self.net.get_guided_backprop(image, neuron_index) for neuron_index in neuron_indices]
        return guided_backprops

    def _get_image_masks(self, guided_backprops):
        projected_gbps = [np.max(gb, axis=-1).squeeze() for gb in guided_backprops]
        raw_masks = [pgbp > np.percentile(pgbp, self.config.cut_off_percentile) for pgbp in projected_gbps]
        # erosion and dilation
        masks = [binary_dilation(binary_erosion(raw_mask)).astype(projected_gbps[0].dtype) for raw_mask in raw_masks]
        return masks

    def _get_topk_neurons(self, image, masks, top_class):
        reshaped_image = image.reshape(np.roll(self.config.vgg16.input_size, 1))
        images = np.stack([np.reshape(reshaped_image * mask, self.config.vgg16.input_size) for mask in masks])
        losses = self.net.get_batch_loss(images, top_class)
        return list(np.argsort(losses)[:self.config.k])
        # return kmax_neuron_indices[np.argsort(losses)[:self.config.k]]

    def _get_bounding_box(self, masks, mask_indices):
        # sorry, worst piece of code
        final_masks = np.array(masks)[mask_indices]
        the_mask = final_masks[0] * False
        for mask in final_masks:
            the_mask = np.logical_or(the_mask, mask)
        y_min = self.config.vgg16.input_size[0]
        x_min = self.config.vgg16.input_size[1]
        y_max = x_max = 0
        for i in range(self.config.vgg16.input_size[0]):
            for j in range(self.config.vgg16.input_size[1]):
                if the_mask[i, j]:
                    y_min = min(y_min, i)
                    x_min = min(x_min, j)
                    y_max = max(y_max, i)
                    x_max = max(x_max, j)
        return [[y_min, x_min], [y_max, x_max]], the_mask

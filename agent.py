import numpy as np
from vis_util import get_full_plot
from scipy.ndimage import binary_erosion, binary_dilation
from data_util import T_SNE, sample_data
from net import Net


def do_batch_plot(guided_backprops, indices, images, bounding_boxes, files_dict):
    folders_and_names = []
    for key, values in files_dict.items():
        for val in values:
            folders_and_names.append([key, val])
    for gbps, index, image, bb, fn in zip(guided_backprops, indices, images, bounding_boxes, folders_and_names):
        top_grads = [gbps[i] for i in index]
        get_full_plot(top_grads, image, bb, *fn)


class Agent(object):
    """
        recipes for doing the dirty work
    """

    def __init__(self, config):
        self.config = config
        self.net = Net(config)

    def get_bounding_box(self, image=None, files_dict=None):
        if image is None and files_dict is None:
            image, files_dict = sample_data(self.config.batch_size)
        preprocessed_image = self._preprocess(image.copy())
        kmax_neuron_indices, top_classes_indices = self.net.get_top_kmax_neurons(preprocessed_image)
        guided_backprops = self._get_guided_backprops(preprocessed_image, kmax_neuron_indices)
        masks = self._get_images_masks(guided_backprops)
        top_k_neurons_relative_indices = self._get_top_k_neurons(preprocessed_image, masks, top_classes_indices)
        bounding_boxes = self._get_all_bounding_boxes(masks, top_k_neurons_relative_indices)
        if self.config.do_plotting:
            do_batch_plot(guided_backprops, top_k_neurons_relative_indices,
                          image, bounding_boxes, files_dict)
        return bounding_boxes, top_classes_indices

    def make_tsne_pic_for_directory(self, folder='personal'):
        fc_features = None
        images = None
        images_number = self.config.grid_size ** 2
        batch_count = images_number // self.config.batch_size + 1
        for i in range(batch_count):
            image_batch, _ = sample_data(self.config.batch_size, [folder])
            images = image_batch if images is None else np.concatenate([images, image_batch])
            fc_features_batch = self.net.get_fc_features(image_batch)
            fc_features = fc_features_batch if fc_features is None else np.concatenate([fc_features, fc_features_batch])

        tsne = T_SNE()
        tsne_embedding = tsne.generate_tsne(fc_features)
        tsne.save_grid(images, tsne_embedding, folder + '.jpg')

    def _preprocess(self, image):
        return self.net.vgg16.preprocess_input(image)

    def _get_guided_backprops(self, images, neuron_indices):
        # lazy programming, this part should as well be vectorized
        guided_backprops = []
        for image, neuron_index in zip(images, neuron_indices):
            gbp = [self.net.get_guided_backprop(np.expand_dims(image, axis=0), ni)
                   for ni in neuron_index]
            guided_backprops.append(gbp)
        return guided_backprops

    def _get_images_masks(self, guided_backprops):
        masks = []
        for gbps in guided_backprops:
            projected_gbps = [np.max(gb, axis=-1).squeeze() for gb in gbps]
            raw_masks = [pgbp > np.percentile(pgbp, self.config.cut_off_percentile) for pgbp in projected_gbps]
            # erosion and dilation
            masks_per_image = [binary_erosion(binary_dilation(raw_mask)).astype(projected_gbps[0].dtype) for raw_mask in
                               raw_masks]
            masks.append(masks_per_image)
        return masks

    def _get_top_k_neurons(self, images, masks, top_class):
        top_k_neurons_relative_indices = []
        for i, image in enumerate(images):
            reshaped_image = image.reshape(np.roll(self.config.vgg16.input_size, 1))
            masked_images = np.stack(
                [np.reshape(reshaped_image * mask, self.config.vgg16.input_size) for mask in masks[i]])
            losses = self.net.get_batch_loss(masked_images, top_class[i])
            top_k_neurons_relative_indices.append(list(np.argsort(losses)[:self.config.k]))
        return top_k_neurons_relative_indices

    def _get_all_bounding_boxes(self, all_masks, all_mask_indices):
        bounding_boxes = []
        for mask, mask_indices in zip(all_masks, all_mask_indices):
            bounding_boxes.append(self._get_bounding_box(mask, mask_indices))
        return bounding_boxes

    def _get_bounding_box(self, masks, mask_indices):
        # sorry, super lazy
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
        return [[x_min, y_min], [x_max, y_max]]

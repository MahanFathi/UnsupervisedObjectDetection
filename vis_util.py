import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from config import Config


def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def get_full_plot(gradients, image, bounding_box, folder, name, show=False, save=True):
    fig, ax = plt.subplots(1, len(gradients) + 1)
    # plot gradients
    for i, grad in enumerate(gradients):
        _, neg_saliency = get_positive_negative_saliency(grad)
        neg_saliency = neg_saliency.squeeze() * 255
        ax[i].imshow(neg_saliency.astype('uint8'))
        ax[i].axis('off')
    # plot the image with bounding box
    ax[len(gradients)].imshow(image.squeeze().astype('uint8'))
    width = bounding_box[1][0] - bounding_box[0][0]
    height = bounding_box[1][1] - bounding_box[0][1]
    box = patches.Rectangle(bounding_box[0], width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax[len(gradients)].add_patch(box)
    ax[len(gradients)].axis('off')
    if save:
        if not os.path.isdir(Config.gradients_plot_path):
            os.mkdir(Config.gradients_plot_path)
        if not os.path.isdir(os.path.join(Config.gradients_plot_path, folder)):
            os.mkdir(os.path.join(Config.gradients_plot_path, folder))
        plt.savefig(os.path.join(Config.gradients_plot_path, folder, name), dpi=200, bbox_inches='tight')
    if show:
        plt.show()

import os


class VGG16:
    input_size = (224, 224, 3)
    feature_layer = 'block5_conv2'


class Config(object):
    vgg16 = VGG16

    # data
    resize = True
    default_size = (600, 600, 3)
    dataset_path = os.path.abspath(os.path.join(os.path.realpath('.'), os.pardir, 'dataset'))
    dataset_allowed_folders = ['personal']
    batch_size = 32     # sorry, not enough local vram

    # hyper-parameters
    kmax = 10
    k = 5
    top_n_classes_weights = [0.05, 0.15, 0.8]
    cut_off_percentile = 20

    # TSNE
    tsne_perplexity = 30
    tsne_iter = 5000
    tsne_output_dir = os.path.abspath(os.path.join(os.path.realpath('.'), 'tsne'))
    grid_size = 30

    # plot
    do_plotting = True
    gradients_plot_path = os.path.abspath(os.path.join(os.path.realpath('.'), 'results'))
    max_per_row_image_plot = 5

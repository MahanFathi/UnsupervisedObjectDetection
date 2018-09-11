import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


class Net(object):
    def __init__(self, config):

        self.config = config
        self.sess = tf.Session()
        self.sess.__enter__()
        self.graph = tf.get_default_graph()

        # for the good of efficient guided-backprop
        self.not_guided_flag = tf.placeholder(dtype=tf.float32, shape=[])  # 1 for normal, 0 for guided
        self._add_guided_backprop()

        self._build_net()

    def _add_guided_backprop(self):
        @ops.RegisterGradient("GuidedRelu")
        def _GuidedReluGrad(op, grad):
            return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]),
                            gen_nn_ops.relu_grad(grad, op.outputs[0]) * self.not_guided_flag)

    def _build_net(self):
        with self.graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            self._vgg16()
            self._fetch_shapes()
            self._build_impact_gradient()
            self._build_guided_backprop()
            self._build_softmax_loss()

    def _vgg16(self):
        self.vgg16 = tf.keras.applications.vgg16
        self.model = self.vgg16.VGG16()
        self.feature_tensor = self.model.get_layer(self.config.vgg16.feature_layer).input
        self.fc_features = self.model.get_layer('fc2').output

    def _fetch_shapes(self):
        # lets not run on batch mode for now (gpu shortage and complexity)
        self.features_shape = self.feature_tensor.get_shape().as_list()
        self.classes_shape = self.model.output.get_shape().as_list()
        self.features_shape[0] = self.classes_shape[0] = 1

    def _build_impact_gradient(self):
        self.category_index_ph = tf.placeholder(tf.int32, shape=[])
        fake_upstream_grad = tf.one_hot(self.category_index_ph, self.classes_shape[-1], axis=-1)
        self.impact_grad = tf.gradients(self.model.output, self.feature_tensor,
                                        grad_ys=[fake_upstream_grad],
                                        name='impact_gradients')

    def _build_guided_backprop(self):
        self.neuron_index = tf.placeholder(tf.int32, shape=[])
        fake_upstream_grad = tf.one_hot(self.neuron_index, np.prod(self.features_shape), axis=-1)
        fake_upstream_grad = tf.reshape(fake_upstream_grad, shape=self.features_shape)
        self.guided_backprop = tf.gradients(self.feature_tensor, self.model.input,
                                            grad_ys=[fake_upstream_grad],
                                            name='guided_backprop')

    def _build_softmax_loss(self):
        self.top_class_index_ph = tf.placeholder(tf.int32, shape=[])
        self.top_class_batch_one_hot = tf.one_hot(tf.ones([self.config.kmax,], dtype=tf.int32) * self.top_class_index_ph,
                                                  self.classes_shape[-1], axis=-1)
        self.softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.top_class_batch_one_hot,
                                                                       logits=self.model.output)

    def get_batch_loss(self, images, top_class):
        assert images.shape[0] is self.config.kmax
        batch_loss = self.sess.run(self.softmax_loss,
                                   feed_dict={
                                       self.top_class_index_ph: top_class,
                                       self.model.input: images,
                                   })
        return batch_loss

    def get_class_scores(self, image):
        prediction_scores = np.squeeze(self.model.predict(image))
        return prediction_scores

    def get_activations(self, image):
        activations = self.sess.run(self.feature_tensor,
                                    feed_dict={self.model.input: image})
        return activations

    def get_fc_features(self, images):
        fc_features = self.sess.run(self.fc_features,
                                    feed_dict={self.model.input: images})
        return fc_features

    def get_impact_gradient_per_class(self, activations, class_index):
        numerical_impact_grad = self.sess.run(self.impact_grad,
                                              feed_dict={
                                                  self.feature_tensor: activations,
                                                  self.category_index_ph: class_index,
                                                  self.not_guided_flag: 1.0,
                                              })
        return numerical_impact_grad[0]

    def get_guided_backprop(self, image, neuron_index):
        numerical_guided_backprop = self.sess.run(self.guided_backprop,
                                                  feed_dict={
                                                      self.model.input: image,
                                                      self.neuron_index: neuron_index,
                                                      self.not_guided_flag: 0.0,
                                                  })
        return numerical_guided_backprop[0]


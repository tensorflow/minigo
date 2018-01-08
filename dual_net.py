"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

import functools
import math
import os.path
import sys
import tensorflow as tf
from tqdm import tqdm
from typing import Dict

import features
import go

# Momentum comes from the AGZ paper. Set at 0.9.
MOMENTUM = 0.9
EPSILON = 1e-5

def round_power_of_two(n: float) -> int:
    """Finds the nearest powero f 2 to a number.

    Thus, 84 -> 64, 120 -> 128, etc.
    """
    return 2 ** int(round(math.log(n, 2)))


def get_default_hyperparams() -> Dict:
    """Returns the hyperparams for the neural net.

    In other words, returns a dict whose paramaters come from the AGZ paper:
    {
        k: number of filters (AlphaGoZero used 256). We use 128 by default for
            a 19x19 go board.
        fc_width: Dimensionality of the fully connected linear layer
        num_shared_layers: number of shared residual blocks. AGZ used both 19
            and 39. Here we use 19 because it's faster to train.
        l2_strength: The L2 regularization parameter. Note AGZ paper has this
            set to 10^-4 for self-play learning.
    }
    """
    k = round_power_of_two(go.N ** 2 / 3) # width of each layer
    fc_width = k * 2
    num_shared_layers = go.N
    l2_strength = 2e-5
    return locals()


class DualNetwork(object):
    """Using TensorFlow, set up the neural network."""
    def __init__(self, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features.NEW_FEATURES)
        hyperparams = get_default_hyperparams()
        for name, param in hyperparams.items():
            setattr(self, name, param)
        self.test_summary_writer = None
        self.summary_writer = None
        self.training_stats = StatisticsCollector()
        self.session = tf.Session()
        self.name = None
        if use_cpu:
            # Set up the computation-graph context to use CPU context
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            self.set_up_network()

    def set_up_network(self):
        # a global_step variable allows epoch counts to persist through
        # multiple training sessions
        self.global_step = global_step = tf.Variable(
            0, name="global_step", trainable=False)

        # the board input features
        self.x = x = tf.placeholder(tf.float32,
            [None, go.N, go.N, self.num_input_planes])

        # the move probabilities to be predicted
        self.pi = pi = tf.placeholder(tf.float32,
            [None, (go.N * go.N) + 1])

        # the result of the game. +1 = black wins -1 = white wins
        self.outcome = outcome = tf.placeholder(tf.float32, [None])
        self.train_mode = train_mode = tf.placeholder(
                tf.bool, name='train_mode')

        my_batchn = functools.partial(tf.layers.batch_normalization,
                                      momentum=.997, epsilon=EPSILON,
                                      fused=True, center=True, scale=True,
                                      training=train_mode)

        my_conv2d = functools.partial(tf.layers.conv2d,
            filters=self.k, kernel_size=[3, 3], padding="same")

        def my_res_layer(inputs):
            int_layer1 = my_batchn(my_conv2d(inputs))
            initial_output = tf.nn.relu(int_layer1)
            int_layer2 = my_batchn(my_conv2d(initial_output))
            output = tf.nn.relu(inputs + int_layer2)
            return output

        initial_output = tf.nn.relu(my_batchn( my_conv2d(x)))

        # the shared stack
        shared_output = initial_output
        for i in range(self.num_shared_layers):
            shared_output = my_res_layer(shared_output)

        # policy head
        policy_conv = tf.nn.relu(my_batchn(
                my_conv2d(shared_output, filters=2, kernel_size=[1, 1]),
            center=False, scale=False))
        logits = tf.layers.dense(
            tf.reshape(policy_conv, [-1, go.N * go.N * 2]),
            go.N * go.N + 1)

        self.policy_output = tf.nn.softmax(logits)

        # value head
        value_conv = tf.nn.relu(my_batchn(
                my_conv2d(shared_output, filters=1, kernel_size=[1, 1]),
            center=False, scale=False))
        value_fc_hidden = tf.nn.relu(tf.layers.dense(
            tf.reshape(value_conv, [-1, go.N * go.N]),
            self.fc_width))
        self.value_output = value_output = tf.nn.tanh(
            tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1]))

        # Training ops
        self.log_likelihood_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=pi))
        self.mse_cost = tf.reduce_mean(tf.square(value_output - outcome))
        self.l2_cost = self.l2_strength * tf.add_n([tf.nn.l2_loss(v)
            for v in tf.trainable_variables() if not 'bias' in v.name])

        # Combined loss + regularization
        self.dual_cost = self.mse_cost + self.log_likelihood_cost + self.l2_cost
        learning_rate = tf.train.exponential_decay(1e-2, global_step, 10 ** 7, 0.5)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.dual_train_step = tf.train.MomentumOptimizer(
                learning_rate, MOMENTUM).minimize(self.dual_cost, global_step=global_step)

        # misc ops
        self.weight_summaries = tf.summary.merge([
            tf.summary.histogram(weight_var.name, weight_var)
            for weight_var in tf.trainable_variables()],
            name="weight_summaries"
        )

        self.saver = tf.train.Saver()

    def initialize_logging(self, tensorboard_logdir):
        """Initializes a logging-summary writer.

        This method initializes a writer to write for writing stats to
        tensorboard.
        """
        self.summary_writer = tf.summary.FileWriter(os.path.join(
                tensorboard_logdir, "training"), self.session.graph)

    def initialize_variables(self, save_file=None):
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            try:
                self.saver.restore(self.session, save_file)
                self.name = os.path.basename(save_file)
            except:
                # some wizardry here... basically, only restore variables
                # that are in the save file; otherwise, initialize them normally.
                from tensorflow.python.framework import meta_graph
                meta_graph_def = meta_graph.read_meta_graph_file(save_file + '.meta')
                stored_var_names = set([n.name
                    for n in meta_graph_def.graph_def.node
                    if n.op == 'VariableV2'])
                print(stored_var_names)
                var_list = [v for v in tf.global_variables()
                    if v.op.name in stored_var_names]
                # initialize all of the variables
                self.session.run(tf.global_variables_initializer())
                # then overwrite the ones we have in the save file
                # by using a throwaway saver, saved models are automatically
                # "upgraded" to the latest graph definition.
                throwaway_saver = tf.train.Saver(var_list=var_list)
                throwaway_saver.restore(self.session, save_file)


    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file):
        if save_file is not None:
            self.saver.save(self.session, save_file)

    def train(self, training_data, batch_size=32):
        """training_data is instanceof our custom DataSet"""
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_pi, batch_res = training_data.get_batch(batch_size)
            _, policy_err, value_err, reg_err, cost = self.session.run(
                [self.dual_train_step, self.log_likelihood_cost,
                 self.mse_cost, self.l2_cost, self.dual_cost],
                feed_dict={self.x: batch_x,
                           self.pi: batch_pi,
                           self.outcome: batch_res,
                           self.train_mode: True,})
            self.training_stats.report(policy_err, value_err, reg_err, cost)
            #print("%d: %.3f, %.3f %.3f" % (i, policy_err, value_err, reg_err))

        accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        if self.summary_writer is not None:
            weight_summaries = self.session.run(
                self.weight_summaries,
                feed_dict={self.x: batch_x,
                           self.pi: batch_pi,
                           self.outcome: batch_res})
            self.summary_writer.add_summary(weight_summaries, global_step)
            self.summary_writer.add_summary(accuracy_summaries, global_step)


    def run(self, position):
        processed_position = features.extract_features(position, features=features.NEW_FEATURES)
        probabilities, value = self.session.run([self.policy_output, self.value_output],
                                         feed_dict={self.x: processed_position[None, :], self.train_mode:False})
        return probabilities[0], value[0]

    def run_many(self, positions, use_random_symmetry=True):
        fts = functools.partial(features.extract_features, features=features.NEW_FEATURES)
        processed = list(map(fts, positions))
        if use_random_symmetry:
            syms_used, processed = features.randomize_symmetries_feat(processed)
        probabilities, value = self.session.run(
            [self.policy_output, self.value_output],
            feed_dict={self.x: processed,
                       self.train_mode: False})
        if use_random_symmetry:
            probabilities = features.invert_symmetries_pi(syms_used, probabilities)
        return probabilities, value



class StatisticsCollector(object):
    """Collect statistics on the runs and create graphs.

    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then shove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    """
    # TODO(kashomon): move to class-local parameters.
    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        policy_error = tf.placeholder(tf.float32, [])
        value_error = tf.placeholder(tf.float32, [])
        reg_error = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        policy_summary = tf.summary.scalar("Policy error", policy_error)
        value_summary = tf.summary.scalar("Value error", value_error)
        reg_summary = tf.summary.scalar("Regularization error", reg_error)
        cost_summary = tf.summary.scalar("Combined cost", cost)
        accuracy_summaries = tf.summary.merge(
                [policy_summary, value_summary, reg_summary, cost_summary],
                name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.policy_costs = []
        self.value_costs = []
        self.regularization_costs = []
        self.combined_costs = []

    def report(self, policy_cost, value_cost, regularization_cost, combined_cost):
        self.policy_costs.append(policy_cost)
        self.value_costs.append(value_cost)
        self.regularization_costs.append(regularization_cost)
        self.combined_costs.append(combined_cost)

    def collect(self):
        avg_pol = sum(self.policy_costs) / len(self.policy_costs)
        avg_val = sum(self.value_costs) / len(self.value_costs)
        avg_reg = sum(self.regularization_costs) / len(self.regularization_costs)
        avg_cost = sum(self.combined_costs) / len(self.combined_costs)
        self.policy_costs = []
        self.value_costs = []
        self.regularization_costs = []
        self.combined_costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.policy_error:avg_pol,
                       self.value_error:avg_val,
                       self.reg_error:avg_reg,
                       self.cost: avg_cost})
        return summary

if __name__ == '__main__':
    p = DualNetwork()
    p.initialize_variables()

    pos = go.Position()
    probs, val = p.run(pos)
    print(probs, val)

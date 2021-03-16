import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import time
import sys
import os
from loader import Loader
fig = plt.figure()
############################################################
####################
TRAINING_STEPS = 20000
embedding_dim = 100
batch_size = 500
learning_rate = .0001
zdim = 128
nfilt = 32
####################
############################################################


conditions = np.random.normal(0, 1, [1000, 2])
x = np.random.normal(0, 1, [1000, 100])

r_train = np.random.choice(x.shape[0], int(.8 * x.shape[0]), replace=False)
r_eval = [tmp for tmp in range(x.shape[0]) if tmp not in r_train]


conditions_train = conditions.copy()[r_train, :]
x_train = x.copy()[r_train, :]
conditions_eval = conditions.copy()[r_eval, :]
x_eval = x.copy()[r_eval, :]



loadtrain = Loader(x_train, conditions_train, shuffle=True)
loadtrain_unshuffled = Loader(x_train, conditions_train, shuffle=False)
loadeval = Loader(x_eval, conditions_eval, shuffle=False)
outdim = loadtrain.data[0].shape[1]


#############################################
#############################################
##### tf graph
def minibatch(input_, num_kernels=15, kernel_dim=10, name='',):
    with tf.variable_scope(name):
        W = tf.get_variable('{}/Wmb'.format(name), [input_.get_shape()[-1], num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], axis=-1)

def nameop(op, name):

    return tf.identity(op, name=name)

def lrelu(x, leak=0.2, name="lrelu"):

    return tf.maximum(x, leak * x)

def bn(tensor, name, is_training):
    return tf.layers.batch_normalization(tensor,
                      momentum=.9,
                      training=True,
                      name=name)

def build_config(limit_gpu_fraction=0.2, limit_cpu_fraction=10):
    if limit_gpu_fraction > 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(device_count={'GPU': 0})
    if limit_cpu_fraction is not None:
        if limit_cpu_fraction <= 0:
            # -2 gives all CPUs except 2
            cpu_count = min(
                1, int(os.cpu_count() + limit_cpu_fraction))
        elif limit_cpu_fraction < 1:
            # 0.5 gives 50% of available CPUs
            cpu_count = min(
                1, int(os.cpu_count() * limit_cpu_fraction))
        else:
            # 2 gives 2 CPUs
            cpu_count = int(limit_cpu_fraction)
        config.inter_op_parallelism_threads = cpu_count
        config.intra_op_parallelism_threads = cpu_count
        os.environ['OMP_NUM_THREADS'] = str(1)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    return config

def get_layer(sess, intensor, data, outtensor, batch_size=100):
    out = []
    for batch in np.array_split(data, data.shape[0]/batch_size):
        feed = {intensor: batch}
        batchout = sess.run(outtensor, feed_dict=feed)
        out.append(batchout)
    out = np.concatenate(out, axis=0)

    return out

def Embedder(x, nfilt, outdim, activation=lrelu, is_training=True):
    h1 = tf.layers.dense(x, nfilt * 4, activation=None, name='h1')
    h1 = activation(h1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h3 = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def Generator(z, cond, nfilt, outdim, activation=lrelu, is_training=True):
    z = tf.concat([z, cond], axis=-1)

    h1 = tf.layers.dense(z, nfilt * 4, activation=None, name='h1')
    h1 = bn(h1, 'h1', is_training)
    h1 = activation(h1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h3 = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def Discriminator(x, cond, nfilt, outdim, activation=tf.nn.relu, is_training=True):
    x = tf.concat([x, cond], axis=-1)

    h1 = tf.layers.dense(x, nfilt * 4, activation=None, name='h1')
    h1 = activation(h1)
    h1 = minibatch(h1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h3 = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def adversarial_loss(logits, labels):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

def Embedder_convolutional(x, nfilt, outdim, activation=lrelu, is_training=True):
    h1 = tf.layers.conv2d(x, nfilt, kernel_size=3, strides=2, name='h1')
    h1 = bn(h1, 'bnh1', is_training=is_training)
    h1 = activation(h1)

    h2 = tf.layers.conv2d(h1, nfilt * 2, kernel_size=3, strides=2, name='h2')
    h2 = bn(h2, 'bnh2', is_training=is_training)
    h2 = activation(h2)

    h3 = tf.layers.conv2d(h2, nfilt * 4, kernel_size=3, strides=2, name='h3')
    h3 = bn(h3, 'bnh3', is_training=is_training)
    h3 = activation(h3)

    h4 = tf.layers.conv2d(h2, nfilt * 8, kernel_size=3, strides=2, name='h4')
    h4 = bn(h4, 'bnh4', is_training=is_training)
    h4 = activation(h4)

    out = tf.layers.dense(tf.layers.flatten(h4), outdim, name='out')

    return out



tf.reset_default_graph()
tfis_training = tf.placeholder(tf.bool, [], name='tfis_training')

condition = tf.placeholder(tf.float32, [None, 2], name='condition')
pre_embedding_condition_G = condition
pre_embedding_condition_D = condition

with tf.variable_scope('generator_condition', reuse=tf.AUTO_REUSE):
    embedded_condition_G = Embedder(pre_embedding_condition_G, nfilt, outdim=2, is_training=tfis_training)
    embedded_condition_G = nameop(embedded_condition_G, 'embedded_condition_G')

with tf.variable_scope('discriminator_condition', reuse=tf.AUTO_REUSE):
    embedded_condition_D = Embedder(pre_embedding_condition_D, nfilt, outdim=2, is_training=tfis_training)
    embedded_condition_D = nameop(embedded_condition_D, 'embedded_condition_D')


# to use convolutional networks, replace with:
# imdim = 128
# nfilt = 32
# outdim = 100
# condition = tf.placeholder(tf.float32, [None, imdim, imdim, 3], name='condition')

# with tf.variable_scope('generator_condition', reuse=tf.AUTO_REUSE):
#     embedded_condition_G = Embedder_convolutional(condition, nfilt=1 * nfilt, outdim=outdim, is_training=tfis_training)
#     embedded_condition_G = pymba.nameop(embedded_condition_G, 'embedded_condition_G')

# with tf.variable_scope('discriminator_condition', reuse=tf.AUTO_REUSE):
#     embedded_condition_D = Embedder_convolutional(condition, nfilt=2 * nfilt, outdim=outdim, is_training=tfis_training)
#     embedded_condition_D = pymba.nameop(embedded_condition_D, 'embedded_condition_D')
##################################################


##################################################

z = tf.placeholder(tf.float32, [None, zdim], name='z')

with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    G = Generator(z, embedded_condition_G, nfilt * 1, outdim, is_training=tfis_training)
    G = tf.nn.sigmoid(G)
    G = nameop(G, 'G')

x = tf.placeholder(tf.float32, [None, outdim], name='x')
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    real = Discriminator(x, embedded_condition_D, nfilt=nfilt * 1, outdim=1, is_training=tfis_training)
    fake = Discriminator(G, embedded_condition_D, nfilt=nfilt * 1, outdim=1, is_training=tfis_training)


loss_D = .5 * tf.reduce_mean(adversarial_loss(logits=real, labels=tf.ones_like(real)))
loss_D += .5 * tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.zeros_like(fake)))
loss_G = tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.ones_like(fake)))

update_ops_D = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminator' in op.name]
update_ops_G = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generator' in op.name]

print('update ops G: {}'.format(len(update_ops_G)))
print('update ops D: {}'.format(len(update_ops_D)))

with tf.control_dependencies(update_ops_D):
    optD = tf.train.AdamOptimizer(learning_rate)
    train_op_D = optD.minimize(loss_D, var_list=[tv for tv in tf.trainable_variables() if 'discriminator' in tv.name])
with tf.control_dependencies(update_ops_G):
    optG = tf.train.AdamOptimizer(learning_rate)
    train_op_G = optG.minimize(loss_G, var_list=[tv for tv in tf.trainable_variables() if 'generator' in tv.name])
##################################################

sess = tf.Session(config=build_config(limit_gpu_fraction=.1))

sess.run(tf.global_variables_initializer())


t = time.time()
training_counter = 0
while training_counter < TRAINING_STEPS + 1:
    training_counter += 1
    batch_x, batch_cond = loadtrain.next_batch(batch_size)
    batchz = np.random.normal(0, 1, [batch_size, zdim])

    feed = {x: batch_x, condition: batch_cond, z: batchz, tfis_training: True}
    sess.run(train_op_G, feed_dict=feed)
    sess.run(train_op_D, feed_dict=feed)


    if training_counter % 100 == 0:
        ld, lg = sess.run([loss_D, loss_G], feed_dict=feed)
        print("{} ({:.3f} s): LossD: {:.3f} LossG: {:.3f} ".format(training_counter, time.time() - t, ld, lg))
        t = time.time()


























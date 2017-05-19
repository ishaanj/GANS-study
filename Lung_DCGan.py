import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import codecs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time
import dicom_dcgan as di
import datetime
import math

#init global variable
i = 0
fig_mat = 8
sample_size = 64
sample_list = []

imgh = 64
imgw = 64
imgc = 1
#Logger Helper Class
class Logger:
    def __init__(self, logging_file):
        self.the_log = codecs.open(logging_file, encoding='utf-8', mode='w')

    def log(self, info):
        self.the_log.write(info.replace('\n', '\r\n'))
        self.the_log.write('\r\n')
        
        
#Utility Functions
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
        
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
  
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
            
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
            
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

#Core Functions
#def sampler(m, n):
#    return np.random.uniform(-1., 1., size=[m, n])
def sampler(z):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = imgh, imgw
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0, train=False))

        h1 = deconv2d(h0, [b_size, s_h8, s_w8, gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1, train=False))

        h2 = deconv2d(h1, [b_size, s_h4, s_w4, gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=False))

        h3 = deconv2d(h2, [b_size, s_h2, s_w2, gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3, train=False))

        h4 = deconv2d(h3, [b_size, s_h, s_w, imgc], name='g_h4')

        return tf.nn.tanh(h4)

#def generator(z):
#    h = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
#    log = tf.matmul(h, g_w2) + g_b2
#    prob = tf.nn.sigmoid(log)
#
#    return prob
def generator(z):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = imgh, imgw
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        #self.z_, self.h0_w, self.h0_b = linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        z_, h0_w, h0_b = linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        #self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        #h0 = tf.nn.relu(g_bn0(self.h0))
        h0 = tf.nn.relu(g_bn0(h0))

        #self.h1, self.h1_w, self.h1_b = deconv2d(h0, [b_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1, h1_w, h1_b = deconv2d(h0, [b_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        #h1 = tf.nn.relu(g_bn1(self.h1))
        h1 = tf.nn.relu(g_bn1(h1))

        #h2, self.h2_w, self.h2_b = deconv2d(h1, [b_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2, h2_w, h2_b = deconv2d(h1, [b_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))

        #h3, self.h3_w, self.h3_b = deconv2d(h2, [b_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3, h3_w, h3_b = deconv2d(h2, [b_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        #h4, self.h4_w, self.h4_b = deconv2d(h3, [b_size, s_h, s_w, imgc], name='g_h4', with_w=True)
        h4, h4_w, h4_b = deconv2d(h3, [b_size, s_h, s_w, imgc], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

#def discriminator(x):
#    h = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
#    log = tf.matmul(h, d_w2) + d_b2
#    prob = tf.nn.sigmoid(log)
#
#    return prob, log
def discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [b_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

def plot_data_figs():
    fig_list = []
    #for samples in list(sample_list[-1]):
    fig = plt.figure(figsize=(fig_mat, fig_mat))
    fig_list.append(fig)
    gs1 = gs.GridSpec(fig_mat, fig_mat)
    gs1.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(sample_list[-1]):
        ax = plt.subplot(gs1[i])
        plt.axis('off')
        ax.set_aspect('equal')
        if imgc == 1:
            plt.imshow(sample.reshape(imgh, imgw), cmap ="Greys_r")
        else:
            plt.imshow(sample.reshape(imgh, imgw, imgc))
    plt.show(fig_list[-1])
    return fig_list[-1]

def plot_loss_figs(iterations):
    x = np.linspace(1, iterations, iterations)
    plt.plot(x, d_losses, color = "blue", label = "D loss");
    plt.plot(x, g_losses, color = "green", label = "G loss");
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

#input handling
iter = int(input("Number of iterations: ") or "15001")
b_size = int(input("Batch Size: ") or "16")
optimizer = input("Type of Optimizer: ") or "adam"
z_dim = int(input("Dimension of Noise for G: ") or "100")

#init D and G layers
#input image
#X = tf.placeholder(tf.float32, shape=[None, 262144])
inputs = tf.placeholder(tf.float32, [b_size] + [imgh,imgw,imgc], name='real_images')
sample_inputs = tf.placeholder(tf.float32, [sample_size] + [imgh,imgw,imgc], name='sample_inputs')

#D layers
#d_w1 = tf.Variable(xavier_init([262144, 128]))
#d_b1 = tf.Variable(tf.zeros(shape=[128]))
#d_w2 = tf.Variable(xavier_init([128, 1]))
#d_b2 = tf.Variable(tf.zeros(shape=[1]))
#t_d = [d_w1, d_b1, d_w2, d_b2]
df_dim = 64
d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

#Noise
z = tf.placeholder(tf.float32, shape=[None, 100], name='z')
z_sum = tf.summary.histogram("z", z)

#G layers
#g_w1 = tf.Variable(xavier_init([100, 128]))
#g_b1 = tf.Variable(tf.zeros(shape=[128]))
#g_w2 = tf.Variable(xavier_init([128, 262144]))
#g_b2 = tf.Variable(tf.zeros(shape=[262144]))
#t_g = [g_w1, g_b1, g_w2, g_b2]
gf_dim = 64
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')

#Read MNIST data
lung_data = di.lung_data()

#Inititalize Logger
out = Logger("log.txt")

#Build model
g_sample = generator(z)
d_prob, d_log = discriminator(inputs)
d_probf, d_logf = discriminator(g_sample, reuse=True)
splr = sampler(z)

d_sum1 = tf.summary.histogram("d", d_prob)
d__sum = tf.summary.histogram("d_", d_probf)
G_sum = tf.summary.image("G", g_sample)

#d_loss = -tf.reduce_mean(tf.log(d_prob) + tf.log(1. - d_probf))
#g_loss = -tf.reduce_mean(tf.log(d_probf))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_log, labels=tf.ones_like(d_prob)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logf, labels=tf.zeros_like(d_probf)))
d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logf, labels=tf.ones_like(d_probf)))
g_loss_sum = tf.summary.scalar("g_loss", g_loss)
d_loss_sum = tf.summary.scalar("d_loss", d_loss)

t_vars = tf.trainable_variables()
t_d = [var for var in t_vars if 'd_' in var.name]
t_g = [var for var in t_vars if 'g_' in var.name]

saver = tf.train.Saver()



#Select optimizer
if (optimizer.lower() == "adam"):
    d_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=t_d)
    g_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=t_g)
elif (optimizer.lower() == "grad"):
    d_solver = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(d_loss, var_list=t_g)
    g_solver = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(g_loss, var_list=t_g)
elif (optimizer.lower() == "rmsprop"):
    d_solver = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(d_loss, var_list=t_d)
    g_solver = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(g_loss, var_list=t_g)
elif (optimizer.lower() == "adagrad"):
    d_solver = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(d_loss, var_list=t_d)
    g_solver = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(g_loss, var_list=t_g)
elif (optimizer.lower() == "adadelta"):
    d_solver = tf.train.AdadeltaOptimizer().minimize(d_loss, var_list=t_d)
    g_solver = tf.train.AdadeltaOptimizer().minimize(g_loss, var_list=t_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
d_sum = tf.summary.merge([z_sum, d_sum1, d_loss_real_sum, d_loss_sum])

#List of losses
# d_losses = []
# g_losses = []

#Training
for j in range(iter):
    if j % 100 == 0:
        print(j)
    if j % 500 == 0:
        #samples, _, _ = sess.run([splr, d_loss, g_loss], feed_dict={z: np.random.uniform(-1., 1., size=[sample_size, z_dim], inputs: sample_inputs)})
        samples = sess.run(g_sample, feed_dict={z: np.random.uniform(-1., 1., size=[sample_size, z_dim])})
        sample_list.append(samples)
        path = "cpts/model-" + datetime.datetime.now().strftime("%I %M%p on %B %d, %Y") + " -iter " + str(j) + ".ckpt"
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    x_mb = lung_data.next_batch(b_size)
    batch_z = np.random.uniform(-1, 1, [b_size, z_dim]).astype(np.float32)
    

    temp, d_loss_curr = sess.run([d_solver, d_sum], feed_dict={inputs: x_mb, z: batch_z})
    temp, g_loss_curr1 = sess.run([g_solver, g_sum], feed_dict={z: batch_z})
    temp, g_loss_curr2 = sess.run([g_solver, g_sum], feed_dict={z: batch_z})

    # d_losses.append(d_loss_curr)
    # g_losses.append(g_loss_curr)
    #if j % 500 == 0:
        #print('Iteration:',j)
        #print('Discriminator loss:', d_loss_curr)
        #print('Generator loss1:', g_loss_curr1)
        #print('Generator loss2:', g_loss_curr2)
        # print()

#Plotting
fig = plot_data_figs()
# plot_loss_figs(iter)

#Logging
out.log(time.asctime( time.localtime(time.time()) ))
out.log("Iterations: " + str(iter))
out.log("Batch Size: " + str(b_size))
out.log("Optimizer: " + optimizer)
out.log("Noise Dimensions: "+str(z_dim))


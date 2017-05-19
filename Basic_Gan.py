import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import codecs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time
import datetime

#init global variable
i = 0
fig_mat = 4
sample_size = 16
sample_list = []
#Logger Helper Class
class Logger:
    def __init__(self, logging_file):
        self.the_log = codecs.open(logging_file, encoding='utf-8', mode='w')

    def log(self, info):
        self.the_log.write(info.replace('\n', '\r\n'))
        self.the_log.write('\r\n')

#Core Functions
def sampler(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    h = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
    log = tf.matmul(h, g_w2) + g_b2
    prob = tf.nn.sigmoid(log)

    return prob

def discriminator(x):
    h = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    log = tf.matmul(h, d_w2) + d_b2
    prob = tf.nn.sigmoid(log)

    return prob, log

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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.show(fig_list[-1])
    return fig_list[-1]

def plot_loss_figs(iterations):
    x = np.linspace(1, iterations, iterations)
    plt.plot(x, d_losses, color = "blue", label = "D loss");
    plt.plot(x, g_losses, color = "green", label = "G loss");
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()

#input handling
iter = int(input("Number of iterations: ") or "10000")
b_size = int(input("Batch Size: ") or "128")
optimizer = input("Type of Optimizer: ") or "adagrad"
z_dim = int(input("Dimension of Noise for G: ") or "100")

#init D and G layers
#input image
X = tf.placeholder(tf.float32, shape=[None, 784])

#D layers
d_w1 = tf.get_variable("DW1", shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
d_b1 = tf.Variable(tf.zeros(shape=[128]))
d_w2 = tf.get_variable("DW2", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
d_b2 = tf.Variable(tf.zeros(shape=[1]))
t_d = [d_w1, d_b1, d_w2, d_b2]

#Noise
z = tf.placeholder(tf.float32, shape=[None, 100])

#G layers
g_w1 = tf.get_variable("GW1", shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
g_b1 = tf.Variable(tf.zeros(shape=[128]))
g_w2 = tf.get_variable("GW2", shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
g_b2 = tf.Variable(tf.zeros(shape=[784]))
t_g = [g_w1, g_b1, g_w2, g_b2]

#Read MNIST data
mnist_data = mnist.input_data.read_data_sets('../../MNIST_data', one_hot=True)

#Inititalize Logger
out = Logger("log.txt")

#Build model
g_sample = generator(z)
d_prob, d_log = discriminator(X)
d_probf, d_logf = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_prob) + tf.log(1. - d_probf))
g_loss = -tf.reduce_mean(tf.log(d_probf))

#Select optimizer
if (optimizer.lower() == "adam"):
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=t_d)
    g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=t_g)
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

#List of losses
d_losses = []
g_losses = []

#Training
for j in range(iter):
    if j % 1000 == 0:
        samples = sess.run(g_sample, feed_dict={z: sampler(sample_size, z_dim)})
        sample_list.append(samples)

    x_mb, temp = mnist_data.train.next_batch(b_size)

    temp, d_loss_curr = sess.run([d_solver, d_loss], feed_dict={X: x_mb, z: sampler(b_size, z_dim)})
    temp, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: sampler(b_size, z_dim)})

    d_losses.append(d_loss_curr)
    g_losses.append(g_loss_curr)
    if j % 1000 == 0:
        print('Iteration:',j)
        print('Discriminator loss:', d_loss_curr)
        print('Generator loss:', g_loss_curr)
        print()

#Plotting
fig = plot_data_figs()
plot_loss_figs(iter)

#Logging
out.log(time.asctime( time.localtime(time.time()) ))
out.log("Iterations: " + str(iter))
out.log("Batch Size: " + str(b_size))
out.log("Optimizer: " + optimizer)
out.log("Noise Dimensions: "+str(z_dim))


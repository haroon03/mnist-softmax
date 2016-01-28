import input_data
import tensorflow as tf

# 55,000 data points for training, 10,000 for testing
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create placeholder
x = tf.placeholder(tf.float32, [None, 784])

# define parameters to learn
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# create new placeholder for cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])

# sum cross-entropies for all images in a batch
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# define optimization algorithm
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize variables
init = tf.initialize_all_variables()

# launch model 
sess = tf.Session()

# run initialization operation
sess.run(init)

# train with stochastic gradient descent
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
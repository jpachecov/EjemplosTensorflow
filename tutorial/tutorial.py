# -*- coding: utf-8 -*-
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)


sess = tf.Session()
#print(sess.run([node1 , node2]))
node3 = tf.add(node1, node2)

print "node 3 : ", node3
print "run node3 : ", sess.run(node3)
print "run node1 + node2 : ", sess.run(node1 + node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

print "placeholder a : ", a

adder_node = a + b

print sess.run(adder_node, {a: 15, b:15})

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
#print sess.run(linear_model, {x:[0,1]})

y = tf.placeholder(tf.float32)

resta = linear_model - y
squared_deltas = tf.square(resta)
loss = tf.reduce_sum(squared_deltas)

xl = [1,2,3,4]
yl = [0,-1,-2,-3]

print "W * x + b : ", sess.run(linear_model, {x:xl})
print "resta : ", sess.run(resta, {x:xl, y:yl})
print "squared_deltas : ", sess.run(squared_deltas, {x:xl, y:yl})
print "loss: ", sess.run(loss, {x:xl, y:yl})


print "Entrenando..."

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
	sess.run(train, {x:xl, y:yl})

print "Después de entrenar: "
print "W = ", sess.run(W)
print "b = ", sess.run(b)

print "Vemos como queda la funcion de perdida despues de entrenar: "
print "loss = ", sess.run(loss, {x:xl, y:yl})

import tensorflow as tf 
 
# we define the computational graph here
# this all is happening in the backend of the program
x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
# actually nothing is actually computed here
# so it prints the tf object only
print(result)


# this is the interative session of the program
# when we call this, then only is computes anything
# tf is back with answers now
with tf.Session() as sess:
	output = sess.run(result)
	print(output)

print(output)

import tensorflow as tf
import numpy as np

print('TF version:',tf.__version__)
print(tf.executing_eagerly())

def fizzbuzz(limit):
    print('Is limit a tensor? %s' %tf.is_tensor(limit))
    if(not tf.is_tensor(limit)):
        limit = tf.convert_to_tensor(limit)
        print('Is it a tensor now? %s' %tf.is_tensor(limit))

    for i in tf.range(1, limit+1):
        if i.numpy() % 3 == 0 and i.numpy() % 5 != 0:
            print("Fizz")
        elif i.numpy() % 5 == 0 and i.numpy() % 3 != 0:
            print("Buzz")
        elif i.numpy() % 5 == 0 and i.numpy() % 3 == 0 :
            print("FizzBuzz")
        else:
            print(i.numpy())

fizzbuzz(tf.constant(15))
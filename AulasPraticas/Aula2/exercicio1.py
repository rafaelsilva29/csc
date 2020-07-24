import tensorflow as tf

tf.random.set_seed(10)

print(50*'*'+' Parte 1 '+50*'*')
#ex1
print('--------- Exercicio 1 -----------')
'''
Criar dois tensors de rank 0, a e b, de qualquer valor. Retornar a+b se a>b senão a-
b;
'''
a = tf.random.uniform(shape=[],dtype=tf.dtypes.float32)
b = tf.random.uniform(shape=[],dtype=tf.dtypes.float32)
print(a)
print(b)
if a > b:
    print(a+b)
else:
    print(a-b)
print('---------------------------------')

#ex2
print('--------- Exercicio 2 -----------')
'''
Criar dois tensors de rank 0, a e b, de qualquer valor aleatório entre -1 e 1. Retornar
a+b se a<b; a-b se a>b; e 0 como default;
'''
x = tf.random.uniform(shape=[],minval=-1,maxval=1,dtype=tf.dtypes.float32)
y = tf.random.uniform(shape=[],minval=-1,maxval=1,dtype=tf.dtypes.float32)
print(x)
print(y)

if x > y:
    print(x+y)
elif x < y:
    print(x-y)
else:
    print('0')
print('---------------------------------')

#ex3
print('--------- Exercicio 3 -----------')
'''
Criar um tensor do tipo variável, a, com o valor [[1, 2, 0], [3, 0, 2]], e um tensor de
zeros, b, com o mesmo shape de a (shape=(2, 3)). Retornar um tensor booleano
com o valor True para cada elemento de a igual a b;
'''
result = tf.math.equal(tf.Variable([[1, 2, 0], [3, 0, 2]]),tf.Variable([[0, 0, 0], [0, 0, 0]]))
print(result)
print('---------------------------------')

#ex4
print('--------- Exercicio 4 -----------')
'''
Criar um tensor 1d, a, com 20 elementos compreendidos entre 1 e 10. Retornar um tensor com 
os elementos de a cujo valor é superior a 7.
'''
z = tf.random.uniform([20], minval=1, maxval=10)
print(z)
out = tf.gather(z, tf.where( tf.greater(z, tf.constant(7, dtype=tf.float32))))
print(out)
print('---------------------------------')

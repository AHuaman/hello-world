import numpy as np
import tensorflow as tf

# Declaramos lo siguiente:
# 1) Variable: Los pesos y bias que se añadirán a las neuronas
# del modelo
# 2) placeholder: Cáscara donde se insetarán las variables
# explicativas ("X's") al momento del entrenamiento
b = tf.Variable(tf.zeros((100,)))
W = tf.Variable(tf.random_uniform((784, 100), -1, 1))
x = tf.placeholder(tf.float32, (100, 784)) # 100 obs y 784 features

# Función de activación de la CAPA OCULTA y operaciones con los pesos:
h = tf.nn.relu(tf.matmul(x, W) + b)
# Ahora la última capa, con activación softmax par obtener probabilidades:
prediction = tf.nn.softmax(...)

# "Y" correctos que deseo usar para clasificar:
label = tf.placeholder(tf.float32, [None, 10])

# Función de pérdida:     (ojo: las funciones "reduce" colapsan dimensiones)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction)), 
								reduction_indices = [1]) # suma horizontal

# Computación automática de gradientes y actualización de variables por backpropagation
# en cada iteración:
train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(loss = cross_entropy)

# En tensorflow, un modelo se ejcuta mediante un "session",
# donde inicializamos el ambiente, los "Variables" e
# introducimos los valores de los "placeholder":
sess = tf.session() # lazyeval on runtime
# sess.run(fectches, feeds)
sess.run(tf.initialize_all_variables()) # pesos de neuronas
sess.run(train_step, {x: np.random.random(100, 784)}) # 100 obs con 784 features
# OJO: el diccionario que se ingresa como segundo argumento
# asigna una lista de valores aleatorios al "placeholder". Por
# ahora queremos utilizar batches para alimentar muestras de datos
# al algoritmo de entrenamiento
#for i in range(1000):
#	batch_x, batch_label = data.next_batch()
#	sess.run(train_step, feed_dict = {x: batcn_x, label: batch_label})
#	pass

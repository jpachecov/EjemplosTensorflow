import tensorflow as tf
import numpy as np

# Declamos lista de features, en este caso tenemos solo una caracteristica numerica
# Por que estamos usando el modelo lineal a una varible
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Llamamos al estimador, un estimador es el punto de entrada para invocar
# entrenamiento (fitting) y evaluacion (inference)
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)


x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)


print "x_train : ", x_train
print "input_fn : ", input_fn()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
# Inicializa la neurona con pesos al azar
def initialize_neuron(input_size):
    weights = np.random.rand(input_size)
    bias = np.random.rand(1)
    return weights, bias

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLu
def relu(x):
    return np.maximum(0, x)

# Define the step function
def step(x):
    return np.where(x >= 0, 1, 0)

# Forward pass through the neuron
def forward_pass(inputs, weights, bias, fun):
    z = np.dot(inputs, weights) + bias
    if fun == 'sigmoid':
        return sigmoid(z)
    elif fun == 'step':
        return step(z)
    elif fun == 'relu':
        return relu(z)
    

#%%
# ejemplo de uso ***********************************
# seteo de valores a mano
weights = np.array([1, 1])
bias = np.array([-1.5])
output = forward_pass([0.5,1],weights,bias, 'sigmoid')[0]
print(output)

# %%
# Restringiendo los valores de entrada de cada neurona entre 0 y 1 analice brevemente como se
# comporta la neurona con inputs de 1, 2 y 3 dimensiones (¿que pasa en dimensiones
# superiores?).
# Pruebe cambiar (y programar) a otras funciones de activacion como ReLu, tanh etc.

# Neurona seteada "a mano"
weights = np.array([1])
bias = np.array([-1.5])

# Neurona con 1 input 
n = 20
res = np.zeros((3, n))
inputs_1 = np.linspace(0, 1, n)


# Analizo comportamiento de la misma neurona con distintos
# valores de entrada

for i, act_fuction in enumerate(['sigmoid', 'step', 'relu']):
    for j, input in enumerate(inputs_1):
        output = forward_pass(input, weights, bias, act_fuction)[0]
        res[i, j] = res[i, j] + output

plt.plot(inputs_1, res[0], marker='o', linestyle='-', color = '#550527', label = 'Sigmoid')
plt.plot(inputs_1, res[1], marker='o', linestyle='-', color = '#688E26', label = 'Step')
plt.plot(inputs_1, res[2], marker='o', linestyle='-', color = '#F44708', label = 'ReLu')
plt.xlabel("Inputs")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.show()

#%%

# 2 inputs

weights = np.array([1, 1])
bias = np.array([-1.5])

inputs_2 = np.array(np.meshgrid(inputs_1, inputs_1)).T.reshape(-1, 2)

fig = plt.figure(figsize=(15, 5))

for i, act_fuction in enumerate(['sigmoid', 'step', 'relu']):
    res = np.zeros(inputs_2.shape[0])
    for j, input in enumerate(inputs_2):
        output = forward_pass(input, weights, bias, act_fuction)[0]
        res[j] = res[j] + output
    
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.scatter(inputs_2[:, 0], inputs_2[:, 1], res, color='b', marker='o')
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    ax.set_title(f'{act_fuction.capitalize()} Results')
plt.tight_layout()
plt.show()

# Pendiente 3 inputs

#%%
# Verifique que con los siguientes paràmetros (w1=1, w2=1, b=−1.5 ) la neurona resuelve
# correctamente la función AND, por ejemplo: para el input [1,1] el output debería ser 1. ¿Se
# anima a programar y verificar las funciones OR, o NOT?.
# ¿Que pasa si en lugar de utilizar como activación “step” utiliza la sigmoidea o ReLu?

# FUNCION AND
weights = np.array([1, 1])
bias = np.array([-1.5])
inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
res = np.zeros(4)

expected_res = [1, 0, 0, 0]

for i, input in enumerate(inputs):
    output = forward_pass(input, weights, bias, 'step')[0]
    res[i] = res[i] + output

if np.array_equal(res, expected_res):
    print('Resuelve AND!')
else:
    print('No resuelve AND')

#%%
# FUNCION OR
weights = np.array([1, 1])
bias = np.array([-0.5]) # para que la funcion lineal de negativa en el caso [0,0]
inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
res = np.zeros(4)

expected_res = [1, 1, 1, 0]

for i, input in enumerate(inputs):
    output = forward_pass(input, weights, bias, 'step')[0]
    res[i] = res[i] + output

if np.array_equal(res, expected_res):
    print('Resuelve OR!')
else:
    print('No resuelve OR')

#%%
# FUNCION NOT

weights = np.array([-1])
bias = np.array([0.5])
inputs = np.array([0, 1])
res = np.zeros(2)

expected_res = [1, 0]

for i, input in enumerate(inputs):
    output = forward_pass(input, weights, bias, 'step')[0]
    res[i] = res[i] + output

if np.array_equal(res, expected_res):
    print('Resuelve NOT!')
else:
    print('No resuelve NOT')

#%%
# 1.1.2 Enseñandole (o entrenando) una neurona.

#Gradiente para una neurona simple con activacion sigmoidea
def compute_gradients(inputs, weights, bias, y_true):
    # Forward pass
    z = np.dot(weights, inputs) + bias
    y_pred = sigmoid(z)
    # Compute loss
    loss = binary_cross_entropy_loss(y_pred, y_true)
    # Backward pass
    d_loss = binary_cross_entropy_derivative(y_pred, y_true) # Gradient of loss w.r.t. y_pred
    d_activation = sigmoid_derivative(z) # Gradient of sigmoid w.r.t. z
    # Gradients for weights and bias
    d_weights = d_loss * d_activation * inputs # Chain rule: dL/dw
    d_bias = d_loss * d_activation # Chain rule: dL/db
    return d_weights, d_bias, loss, y_pred

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Es importante notar que para este caso NO podemos usar la step
# function como función de activación ya que no es derivable, por eso uitilizamos la sigmoidea.

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Binary cross-entropy loss function
def binary_cross_entropy_loss(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# LF derivative
def binary_cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true
# %%
# Como ejemplo vamos a entrenar nuestra neurona para que a partir de 2 inputs binarios realice
# la operación lógica AND.
# Elija parámetros cercanos a los correctos (por ejemplo w1=5, w2=3, b=-7) y calcule para cada
# uno de los cuatro input posibles ([0,0][1,0][0,1][1,1]) el output de la neurona, el valor de la LF y
# los 3 gradientes.
# Repita la operación pero cambiando los parámetros (por ejemplo eligiendolos al azar).
# Analice y comente los resultados.

# Parametros cercanos a los correctos
weights = np.array([5, 3])
bias = np.array([-7])

inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
expected_res = np.array([1, 0, 0, 0])

print("Resultados con parámetros cercanos a los correctos")
for i in range(expected_res.shape[0]):
    d_weights_correctos, d_bias_correctos, loss_correctos, y_pred_correctos = compute_gradients(inputs[i], weights, bias, expected_res[i])
    print(d_weights_correctos, d_bias_correctos, loss_correctos, y_pred_correctos)
# Con pesos y sesgo random
weights, bias = initialize_neuron(2)

print("Resultados con parámetros random")
for i in range(expected_res.shape[0]):
    d_weights_random, d_bias_random, loss_random, y_pred_random = compute_gradients(inputs[i], weights, bias, expected_res[i])
    print(d_weights_random, d_bias_random, loss_random, y_pred_random)

#%%

def optimizacion(old_value, gradient, LR):
    new_value = old_value - gradient * LR
    return new_value

# Elija unos valores de w1,w2,b de partida y uno de los casos testigo. Determine la LF y los
# gradientes. Luego aplique la formula para actualizar los pesos (puede probar diferentes LR).
# Verifique que cuando los parámetros son actualizados en la dirección correcta la LF baja y el
# valor predicho se acerca al real.

weights = np.array([1, 1])
bias = np.array([-1])

input = np.array([1, 1])
expected_res = np.array([1])

d_weights, d_bias, loss, y_pred = compute_gradients(input, weights, bias, expected_res)

print('Params w, b:', weights, bias)
print('Inputs: ', input)
print('Output: ', y_pred)
print('Loss: ', loss)
print('Gradients: ', d_weights, d_bias)

print('\nLR = 1')

w1 = optimizacion(weights[0], d_weights[0], 1)
w2 = optimizacion(weights[1], d_weights[1], 1)
weights = np.array([w1, w2])
bias = optimizacion(bias, d_bias, 1)

d_weights, d_bias, loss, y_pred = compute_gradients(input, weights, bias, expected_res)

print('Params w, b:', weights, bias)
print('Inputs: ', input)
print('Output: ', y_pred)
print('Loss: ', loss)
print('Gradients: ', d_weights, d_bias)

print('\nLR = 5')
weights = np.array([1, 1])
bias = np.array([-1])

w1 = optimizacion(weights[0], d_weights[0], 5)
w2 = optimizacion(weights[1], d_weights[1], 5)
weights = np.array([w1, w2])
bias = optimizacion(bias, d_bias, 1)

d_weights, d_bias, loss, y_pred = compute_gradients(input, weights, bias, expected_res)

print('Params w, b:', weights, bias)
print('Inputs: ', input)
print('Output: ', y_pred)
print('Loss: ', loss)
print('Gradients: ', d_weights, d_bias)

# Se verifica que la LF baja, y los valores de y_pred se acercan a 1
# (valor esperado)

#%%
# Analice ahora qué pasa para los gradientes obtenidos con los “mismos” parámetros pero para
# cada una de las 4 posibles entradas-salidas del AND. ¿Que le parece?

weights = np.array([1, 1])
bias = np.array([-1])

inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
expected_res = [1, 0, 0, 0]

for i, input in enumerate(inputs):
    d_weights, d_bias, loss, y_pred = compute_gradients(input, weights, bias, expected_res[i])
   
    print('Inputs: ', input)
    print('Params w, b:', weights, bias)
    print('Output: ', y_pred)
    print('Loss: ', loss)
    print('Gradients: ', d_weights, d_bias, '\n')

# Actualizacion de parametros usando los gradientes correspondientes al input [1, 1]
print('LR = 1\n')
w1 = optimizacion(weights[0],-0.05287709, 1)
w2 = optimizacion(weights[1], -0.05287709, 1)
weights = np.array([w1, w2])
bias = optimizacion(bias, -0.05287709, 1)

for i, input in enumerate(inputs):
    d_weights, d_bias, loss, y_pred = compute_gradients(input, weights, bias, expected_res[i])
   
    print('Inputs: ', input)
    print('Params w, b:', weights, bias)
    print('Output: ', y_pred)
    print('Loss: ', loss)
    print('Gradients: ', d_weights, d_bias, '\n')

# La LF disminuye en el caso del input [1,1] pero aumenta en los demas casos

# Las optimizaciones de parámetros se
# hacen utilizando el promedio de los gradientes obtenidos para todo (batch) un pequeño
# conjunto (mini batch) de datos de entrenamiento, y este valor promedio es el que se utiliza para
# actualizar los parámetros.

#%% 
# Ejercicio 1.1: a) Utilizando una neurona de dos entradas (x1, x2) y la función de activacion
# sigmoidea, programe un código que tomando valores de w1,w2 y b de entrada al azar,
# determine los gradientes y los vaya actualizando hasta encontrar un valor optimo, de modo que
# la neurona ejecute el AND.
# Controle del mismo la cantidad de ciclos de entrenamiento, y el LR.
# Consejo: pruebe diferentes valores de LR, e incluso puede utilizar una funciòn donde el LR
# decrece a medida que avanza el entrenamiento. Ademas utilice los gradientes promedio (batch)
# para la actualizaciòn de los parámetros
# Para monitorear el entrenamiento calcule (y grafique) el avance de la LF en función del número
# de ciclo y calcule y muestre los valores finales que obtiene la neurona para cada una de las 4
# condiciones de entrada del AND.
# Analice los resultados
# b) Que pasa si en lugar de una sigmoidea utiliza ReLu?
# c) Se anima a repetir el proceso pero para la función lógica “OR”
# d) ¿que pasa si intenta resolver el “XOR”?
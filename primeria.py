import tensorflow as tf #para inteligencia artificial hecha por google 
import numpy as np # facilita el trabajo con arreglos numericos

celcius= np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit=np.array([-40,14,32,46,59,72,100],dtype=float)# arreglos con los cuales va a practicar porque tiene la respuesta correcta a la prediccion que el hace

capa= tf.keras.layers.Dense(units=1,input_shape=[1]) #inicilizamos una capa de tipo densa, estas son las cuales tienen conexion de cada neurona hacia todas las neuronas de la siguiente capa, como aca tenemos solo dos neuronas es facil porque no hay mucho mas que conectar
modelo=tf.keras.Sequential([capa]) # creamos modelo para poder trabajar con el, un modelo secuencial


# tenemos que preparar el modelo para que sea entrenado por lo que tengo que decirle de alguna forma como quiero que entrene 
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), #le permite a la red saber como ajustas los pesos y sesgos de manera eficiente para que poco a poco vaya mejorando
    loss='mean_squared_error' #esta funcion considera que una pequeña cantidad de errores grandes es peor que una gran cantidad de errores pequeños
    
)

print("comenzando entrenamiento...")
historial=modelo.fit(celcius,fahrenheit,epochs=1000,verbose=False)
print("ya se entreno el modelo!")

entrada=np.array([100.0])
resultado=modelo.predict(entrada)
print("resultado es " + str(resultado))

# tenemos que tener en cuenta que aca no se necesito ni muchos datos ni mucho entrenamiento por ser la convercion de celcius a f una formula lineal = c*1.8+32
# pero si esto no fuese asi se complejisa todo.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import gym
import random

model = Sequential()
model.add(Dense(16,input_dim=5,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="linear"))
adam = adam(lr=0.001)
model.compile(optimizer="adam",loss="mse")

try:
    model.load_weights("pesos.h5")
    print("Pesos cargados")
except:
    print("No hay pesos guardados")
env = gym.make("CartPole-v1")
estado = env.reset()
accion = 0
ep = 0
r = 0
epsilon = 0.02
epsi = epsilon
render = True


while True:
    r += 1
    x0 = np.array([[estado[0]],[estado[1]],[estado[2]],[estado[3]],[0]])
    x0 = np.reshape(x0,[1,5])
    x1 = np.array([[estado[0]],[estado[1]],[estado[2]],[estado[3]],[1]])
    x1 = np.reshape(x1, [1, 5])
    #Seleccion accion
    pre0 = model.predict(x0)
    pre1 = model.predict(x1)
    #print(pre0)
    if random.uniform(0,1)>epsi:
        if (abs(pre0[0][3]) * abs(pre0[0][2])  ) > ( abs(pre1[0][3]) * abs(pre1[0][2]) ):
            accion = 1

        else:
            accion = 0
    else:
        accion = random.randint(0,1)

    estadonuevo,reward,done,info = env.step(accion)
    y = np.array([[estadonuevo[0]], [estadonuevo[1]], [estadonuevo[2]], [estadonuevo[3]]])
    #print(estadonuevo," Nuevo")
    y = np.reshape(y, [1, 4])

    estado = estadonuevo

    if accion==1:
        model.fit(x1,y,verbose=0)
    else:
        model.fit(x0,y,verbose=0)


    if done:
        print("Episodio --- ",ep, "Reward --- ",r)
        if ep%20 == 0:
            model.save_weights("pesos.h5")
            print("Pesos guardados")
            print(model.evaluate(x0,y))
        epsi = epsilon
        r = 0
        ep += 1
        estado = env.reset()



    if render:
        env.render()


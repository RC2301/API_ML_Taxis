from flask import Flask, app, jsonify, render_template, request
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random
warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route('/')
def index():
    repositorio = pd.read_csv("proyecto3.csv")
    r = np.asmatrix(repositorio)
    x = r[:, 5:7].astype(float)
    y = r[:, 4]
    train = 0.7  # entrenamiento
    test = 0.3  # prueba
    x1, x2, y1, y2 = train_test_split(x, y, test_size=test, random_state=100)
    modelo = SVC(C=100, kernel='linear', random_state=123)
    modelo.fit(x1, y1)
    predicciones = modelo.predict(x)
    accuracy = accuracy_score(
        y_true=y,
        y_pred=predicciones,
        normalize=True
    )
    r = []
    f = 0
    while f < len(predicciones):
        r.append(predicciones[f])
        f = f+1
    nombres = ['CESAREO ALVARADO', 'ALVARO BARRIONUEVO', 'BRYAN BORJA', 'JESSICA BORJA', 'BYRON BORJA', 'CARLOS CASTILLO', 'ANGEL CUENCA',
               'CRISTOBAL ENRIQUEZ', 'DIEGO FALCON', 'ALEXIS GONZALES', 'JORGE INGUILLAY', 'FERNANDO LIVISACA', 'MAYRA ORTIZ', 'LUIS PILATAXI', 'CESAR QUINGATUÑA', 'JOSE QUISHPI', 'JOSE QUITIO', 'MARCO SANCHEZ', 'ABEL SANTANA', 'DIEGO SOTALIN', 'FRANKLIN TARCO', 'FAUSTO TIGASI', 'LUIS TIGASI', 'EDWIN TOAPANTA', 'RICARDO VASCO', 'EDUARDO VILLAVICENCIO', 'MARCO YANEZ', 'DANIEL YEPEZ', 'MONICA ZAMORA']
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    c = []
    n = []
    o = 0
    while o < len(r):
        if (r[o] == 'SUR 1'):
            s1.append(nombres[o])
            o = o+1
        elif(r[o] == 'SUR 2'):
            s2.append(nombres[o])
            o = o+1
        elif(r[o] == 'SUR 3'):
            s3.append(nombres[o])
            o = o+1
        elif(r[o] == 'SUR 4'):
            s4.append(nombres[o])
            o = o+1
        elif(r[o] == 'CENTRO'):
            c.append(nombres[o])
            o = o+1
        else:
            n.append(nombres[o])
            o = o+1
    ########## Lunes #########
    aleatorio = random.choice(s2)
    aleatorio2 = random.choice(s2)
    s2.remove(aleatorio)
    s1.append(aleatorio)
    s2.remove(aleatorio2)
    s4.append(aleatorio2)
    aleatorio3 = random.choice(s3)
    aleatorio4 = random.choice(s3)
    aleatorio5 = random.choice(s3)
    s3.remove(aleatorio3)
    s1.append(aleatorio3)
    s3.remove(aleatorio4)
    s4.append(aleatorio4)
    s3.remove(aleatorio5)
    n.append(aleatorio5)
    aleatorio6 = random.choice(c)
    c.remove(aleatorio6)
    s4.append(aleatorio6)
    aleatorio7 = random.choice(c)
    c.remove(aleatorio7)
    s1.append(aleatorio7)
    aleatorio8 = random.choice(c)
    c.remove(aleatorio8)
    s4.append(aleatorio8)
    aleatorio9 = random.choice(c)
    c.remove(aleatorio9)
    n.append(aleatorio9)
    aleatorio10 = random.choice(s2)
    s2.remove(aleatorio10)
    s1.append(aleatorio10)
    aleatorio11 = random.choice(s3)
    s3.remove(aleatorio11)
    s4.append(aleatorio11)
    ########### LUNES ################
    R1 = random.choice(s1)
    s1.remove(R1)
    R2 = random.choice(s2)
    s2.remove(R2)
    R3 = random.choice(s3)
    s3.remove(R3)
    R4 = random.choice(s4)
    s4.remove(R4)
    R5 = random.choice(c)
    c.remove(R5)
    R6 = random.choice(n)
    n.remove(R6)
    L = [R1, R2, R3, R4, R5, R6]
    print("El Lunes esta asignado para SUR 1 ", L[0], " para SUR 2 ", L[1], "para SUR 3 ",
          L[2], " para SUR 4 ", L[3], " para CENTRO ", L[4], " y para NORTE ", L[5])
    ######### MARTES ######################
    R7 = random.choice(s1)
    s1.remove(R7)
    R8 = random.choice(s2)
    s2.remove(R8)
    R9 = random.choice(s3)
    s3.remove(R9)
    R10 = random.choice(s4)
    s4.remove(R10)
    R11 = random.choice(c)
    c.remove(R11)
    R12 = random.choice(n)
    n.remove(R12)
    M = [R7, R8, R9, R10, R11, R12]
    print("El Martes esta asignado para SUR 1 ", M[0], " para SUR 2 ", M[1], "para SUR 3 ",
          M[2], " para SUR 4 ", M[3], " para CENTRO ", M[4], " y para NORTE ", M[5])
    ######## MIERCOLES #########################
    R13 = random.choice(s1)
    s1.remove(R13)
    R14 = random.choice(s2)
    s2.remove(R14)
    R15 = random.choice(s3)
    s3.remove(R15)
    R16 = random.choice(s4)
    s4.remove(R16)
    R17 = random.choice(c)
    c.remove(R17)
    R18 = random.choice(n)
    n.remove(R18)
    MI = [R13, R14, R15, R16, R17, R18]
    print("El Miercoles esta asignado para SUR 1 ", MI[0], " para SUR 2 ", MI[1], "para SUR 3 ",
          MI[2], " para SUR 4 ", MI[3], " para CENTRO ", MI[4], " y para NORTE ", MI[5])
    ######### JUEVES #########################
    R19 = random.choice(s1)
    s1.remove(R19)
    R20 = random.choice(s2)
    s2.remove(R20)
    R21 = random.choice(s3)
    s3.remove(R21)
    R22 = random.choice(s4)
    s4.remove(R22)
    R23 = random.choice(c)
    c.remove(R23)
    R24 = random.choice(n)
    n.remove(R24)
    J = [R19, R20, R21, R22, R23, R24]
    print("El Jueves esta asignado para SUR 1 ", J[0], " para SUR 2 ", J[1], "para SUR 3 ",
          J[2], " para SUR 4 ", J[3], " para CENTRO ", J[4], " y para NORTE ", J[5])
    ######### VIERNES ############################
    R25 = random.choice(s1)
    s1.remove(R25)
    R26 = random.choice(s2)
    s2.remove(R26)
    R27 = random.choice(s3)
    s3.remove(R27)
    R28 = random.choice(s4)
    s4.remove(R28)
    R29 = random.choice(c)
    c.remove(R29)
    V = [R25, R26, R27, R28, R29]
    print("El Viernes esta asignado para SUR 1 ",
          V[0], " para SUR 2 ", V[1], "para SUR 3 ", V[2], " para SUR 4 ", V[3], " para CENTRO ", V[4])
    return render_template('index.html', v0=V[0])

if __name__ == '__main__':
    app.run(port=5000)

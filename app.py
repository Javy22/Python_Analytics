
'''
API Machine Learning para indicar los valores de Obesidad
----------------------------------------------------------
Autor: Salinas Javier
Version: 0.1
 
Descripcion:
Programa para indicar predicciones sobre la salud. 

Ejecución: Lanzar el programa y abrir en un navegador la siguiente dirección URL
http://127.0.0.1:5000/

'''

__author__ = "Salinas Javier"
__email__ = "salinasjom@hotmail.com"
__version__ = "0.1"

import traceback
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
model = pickle.load(open('Obesidad_model.pkl', 'rb'))
le = pickle.load(open('Obesidad_encoder.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


# Cuando se presione el botón el sistema llama a esta función
# para realizar la predicción con el modelo de inteligencia artificial
@app.route('/predict', methods=['POST'])
def predict():
    try:

        Sexo = str(request.form.get('Sexo'))
        Altura = float(request.form.get('Altura'))
        Peso = float(request.form.get('Peso'))
        Imc = float(request.form.get('Imc'))

       
        sex_encoded = int(le.transform([Sexo]))
        
        features = np.array([sex_encoded, Altura, Peso, Imc])
       
        numpy_features = features.reshape(1, -1)
        prediction = model.predict(numpy_features)

        if prediction[0] == 1:
            image = url_for('static', filename='media/Peso_estandar.jpg')
            return render_template('index.html', prediction_text='Normal_Weight', prediction_image=image)
        if prediction[0] == 2:
            image = url_for('static', filename='media/Sobrepeso.jpg')
            return render_template('index.html', prediction_text='Obesity_Type_I', prediction_image=image)
        if prediction[0] == 3:
            image = url_for('static', filename='media/Sobrepeso.jpg')
            return render_template('index.html', prediction_text='Obesity_Type_II', prediction_image=image)
        if prediction[0] == 4:
            image = url_for('static', filename='media/Sobrepeso.jpg')
            return render_template('index.html', prediction_text='Obesity_Type_III', prediction_image=image)
        if prediction[0] == 5:
            image = url_for('static', filename='media/Sobrepeso.jpg')
            return render_template('index.html', prediction_text='Overweight_Level_I', prediction_image=image)
        if prediction[0] == 6:
            image = url_for('static', filename='media/Sobrepeso.jpg')
            return render_template('index.html', prediction_text='Overweight_Level_II', prediction_image=image)
        else:
            image = url_for('static', filename='media/Peso_insuficiente.jpg')
            return render_template('index.html', prediction_text='Insufficient_Weight', prediction_image=image)

    except Exception as e:
        print(e)
        return render_template('index.html', prediction_text='Datos mal ingresados')


if __name__ == "__main__":
    app.run(debug=True)

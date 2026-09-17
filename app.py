# -*- coding: utf-8 -*-
"""Despliegue de 4_Validacion_Cruzada_Class_MEJORADO.ipynb

# Despliegue

- Cargamos el modelo (Pipeline: StandardScaler + MLP-tanh)
- Capturamos los datos con Streamlit
- Preparamos los datos: dummies + reindex (igual que en el notebook)
- Aplicamos el modelo para la predicción
"""

# Cargamos librerías principales
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Cargamos el modelo
import streamlit as st  # solo se ejecuta en un servidor web

filename = Path(__file__).parent / 'modelo_final_despliegue.pkl'
with open(filename, 'rb') as f:
    contenido = joblib.load(f)

# El pkl guarda (modelo, columnas); si el notebook se re-ejecuta y guarda
# el modelo solo, se leen las columnas desde columnas_modelo.pkl
if isinstance(contenido, tuple):
    modelo, variables = contenido
else:
    modelo = contenido
    with open(Path(__file__).parent / 'columnas_modelo.pkl', 'rb') as f:
        variables = joblib.load(f)

# Interfaz gráfica
st.title('Predicción de ataque al corazón (stroke)')

# Configuramos las variables que hay que recibir, los nombres deben ser exactamente iguales
age = st.slider('Edad', min_value=1, max_value=82, value=45, step=1)
avg_glucose_level = st.number_input('Nivel promedio de glucosa', min_value=55.12, max_value=271.74,
                                    value=100.0, step=0.01, format='%.2f')
hypertension = st.selectbox('Hipertensión', ['No', 'Yes'])
heart_disease = st.selectbox('Enfermedad cardíaca', ['No', 'Yes'])
ever_married = st.selectbox('Alguna vez casado(a)', ['No', 'Yes'])
smoking_status = st.selectbox('Estado de fumador', ["'never smoked'", 'Unknown', "'formerly smoked'", 'smokes'])

# Dataframe que integra esos datos
datos = [[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]]
data = pd.DataFrame(datos, columns=['age', 'hypertension', 'heart_disease', 'ever_married',
                                    'avg_glucose_level', 'smoking_status'])

# Se realiza la preparación de datos (igual que en el notebook)
data_preparada = data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['smoking_status'], drop_first=False, dtype=int)
data_preparada = pd.get_dummies(data_preparada, columns=['hypertension', 'heart_disease', 'ever_married'],
                                drop_first=True, dtype=int)

# Se adicionan las columnas faltantes
# quita las variables que sobran y las que faltan las agrega con valor 0
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# Predicciones
Y_pred = modelo.predict(data_preparada)
proba = modelo.predict_proba(data_preparada)[0][1]  # probabilidad de la clase 1 = 'Yes'

data['Prediccion'] = ['Sí' if p == 1 else 'No' for p in Y_pred]
data['Probabilidad'] = round(proba * 100, 2)

st.dataframe(data)

if Y_pred[0] == 1:
    st.error(f'Riesgo de ataque al corazón: Sí ({proba * 100:.2f}% de probabilidad)')
else:
    st.success(f'Riesgo de ataque al corazón: No ({proba * 100:.2f}% de probabilidad)')

# Recordar medida del modelo
st.info('Modelo: MLP (tanh, 100-50-25) con F1-macro ≈ 0.906 en validación cruzada de 10 folds (gap train-test < 0.05)')

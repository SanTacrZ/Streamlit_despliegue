# -*- coding: utf-8 -*-
"""Despliegue de 4_Validacion_Cruzada_Class_MEJORADO.ipynb

# Despliegue

- Cargamos el modelo (Pipeline: StandardScaler + MLP-tanh)
- Capturamos los datos con Streamlit
- Preparamos los datos: dummies + reindex (igual que en el notebook)
- Aplicamos el modelo para la predicción
"""

# ------------------------------------------------------------------ Librerías
import pandas as pd
import joblib
from pathlib import Path
import altair as alt
import streamlit as st  # solo se ejecuta en un servidor web

# ------------------------------------------------------------------ Configuración de página
st.set_page_config(
    page_title='Predicción de Ataque al Corazón',
    page_icon='❤️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ------------------------------------------------------------------ Estilos
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }
    .stApp {
        background:
            radial-gradient(900px 500px at 8% 0%, #eef2ff 0%, rgba(238,242,255,0) 60%),
            radial-gradient(900px 500px at 100% 10%, #fdf2f8 0%, rgba(253,242,248,0) 55%),
            #f5f7fb;
    }
    #MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

    /* Tarjetas */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #e8ecf4;
        box-shadow: 0 12px 32px rgba(15, 23, 42, .06);
    }

    /* Encabezado */
    .hero {
        position: relative;
        background: linear-gradient(120deg, #be123c 0%, #7c3aed 58%, #4f46e5 100%);
        border-radius: 26px;
        padding: 2.1rem 2.4rem 2rem;
        color: #fff;
        overflow: hidden;
        box-shadow: 0 20px 45px rgba(79, 70, 229, .32);
        margin-bottom: 1.4rem;
    }
    .hero::after {
        content: ''; position: absolute; right: -70px; top: -90px;
        width: 300px; height: 300px; border-radius: 50%;
        background: rgba(255, 255, 255, .10);
    }
    .hero::before {
        content: ''; position: absolute; right: 130px; bottom: -120px;
        width: 230px; height: 230px; border-radius: 50%;
        background: rgba(255, 255, 255, .07);
    }
    .hero .badge {
        display: inline-block; padding: .28rem .85rem; border-radius: 999px;
        background: rgba(255, 255, 255, .18); border: 1px solid rgba(255, 255, 255, .38);
        font-size: .76rem; font-weight: 600; letter-spacing: .1em;
    }
    .hero h1 { color: #fff; font-size: 2.2rem; font-weight: 800; margin: .7rem 0 .4rem; }
    .hero p { margin: 0; max-width: 780px; opacity: .95; font-size: 1.02rem; }
    .hero .chips { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
    .hero .chip {
        background: rgba(255, 255, 255, .14); border: 1px solid rgba(255, 255, 255, .28);
        padding: .3rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 500;
    }

    /* Títulos de tarjeta y secciones */
    .card-title { font-size: 1.12rem; font-weight: 700; color: #0f172a; margin: 0; }
    .card-sub { color: #64748b; font-size: .85rem; margin: .1rem 0 .5rem; }
    .seccion {
        display: flex; align-items: center; gap: .55rem;
        font-weight: 600; color: #334155; font-size: .92rem; margin: .6rem 0 .1rem;
    }
    .seccion .num {
        background: #eef2ff; color: #4f46e5; width: 1.55rem; height: 1.55rem;
        border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
        font-size: .8rem; font-weight: 700;
    }

    /* Veredicto */
    .veredicto { border-radius: 18px; padding: 1rem 1.2rem; text-align: center; color: #fff; }
    .veredicto-alto { background: linear-gradient(135deg, #dc2626, #f97316); }
    .veredicto-bajo { background: linear-gradient(135deg, #059669, #34d399); }
    .veredicto .label { font-size: .74rem; letter-spacing: .16em; opacity: .92; font-weight: 600; }
    .veredicto .valor { font-size: 2.3rem; font-weight: 800; line-height: 1.2; }

    /* Factores de riesgo */
    .factor {
        display: inline-block; background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca;
        padding: .28rem .75rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
        margin: 0 .35rem .4rem 0;
    }
    .factor-ok {
        background: #f0fdf4; color: #15803d; border-color: #bbf7d0;
    }

    /* Mini métricas */
    .mini {
        background: #f8fafc; border: 1px solid #eef2f7; border-radius: 14px;
        padding: .8rem 1rem; text-align: center;
    }
    .mini .lbl { color: #64748b; font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; }
    .mini .val { font-size: 1.5rem; font-weight: 700; color: #0f172a; line-height: 1.3; }

    /* Panel lateral */
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ecf4; }

    /* Animación */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    .fade-in { animation: fadeUp .55s ease-out; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Modelo
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

# ------------------------------------------------------------------ Encabezado
st.markdown("""
<div class="hero fade-in">
    <span class="badge">RED NEURONAL MLP · F1-MACRO ≈ 0.906</span>
    <h1>Predicción de Ataque al Corazón</h1>
    <p>Completa los datos del paciente y el modelo estimará el riesgo de sufrir un ataque
    al corazón en tiempo real. Entrenado con validación cruzada de 10 folds y datos balanceados 1:1.</p>
    <div class="chips">
        <span class="chip">StratifiedKFold (10 folds)</span>
        <span class="chip">SMOTENC 1:1</span>
        <span class="chip">9 variables predictoras</span>
        <span class="chip">Gap train-test &lt; 0.05</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Panel lateral
with st.sidebar:
    st.markdown('## Sobre el modelo')
    st.markdown(
        '<span class="seccion"><span class="num">1</span> Balanceo SMOTENC 1:1</span><br>'
        '<span class="seccion"><span class="num">2</span> Dummies + LabelEncoder</span><br>'
        '<span class="seccion"><span class="num">3</span> Escalado + MLP en Pipeline</span><br>'
        '<span class="seccion"><span class="num">4</span> Validación cruzada (10 folds)</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('### Ficha técnica')
    st.markdown(
        '- **Algoritmo:** MLP (tanh) 100&#8209;50&#8209;25\n'
        '- **F1-macro:** ≈ 0.906\n'
        '- **Gap train-test:** < 0.05\n'
        '- **Registros:** 5.110 → 9.722 (balanceado)\n'
        '- **Variables:** 9 (edad, glucosa y categóricas)'
    )
    st.divider()
    st.caption('Herramienta académica con fines educativos. No reemplaza un diagnóstico médico.')

# ------------------------------------------------------------------ Interfaz
col_izq, col_der = st.columns([1.05, 1], gap='large')

with col_izq:
    with st.container(border=True):
        st.markdown('<p class="card-title">Datos del paciente</p>'
                    '<p class="card-sub">Ajusta los valores y la predicción se actualiza automáticamente.</p>',
                    unsafe_allow_html=True)

        st.markdown('<p class="seccion"><span class="num">1</span> Datos generales</p>', unsafe_allow_html=True)
        age = st.slider('Edad', min_value=1, max_value=82, value=45, step=1,
                        help='Edad del paciente en años (rango del dataset: 1 a 82).')
        avg_glucose_level = st.slider('Nivel promedio de glucosa', min_value=55.12, max_value=271.74,
                                      value=100.0, step=0.01, format='%.2f',
                                      help='Glucosa promedio en sangre (mg/dL). Valores ≥ 140 se consideran elevados.')

        st.markdown('<p class="seccion"><span class="num">2</span> Antecedentes clínicos</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            hypertension = st.radio('Hipertensión', ['No', 'Sí'], horizontal=True)
        with c2:
            heart_disease = st.radio('Enfermedad cardíaca', ['No', 'Sí'], horizontal=True)

        st.markdown('<p class="seccion"><span class="num">3</span> Estilo de vida</p>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            ever_married = st.radio('Alguna vez casado(a)', ['No', 'Sí'], horizontal=True)
        with c4:
            smoking_status = st.selectbox('Estado de fumador',
                                          ['Nunca ha fumado', 'Fuma actualmente', 'Ex fumador', 'No se sabe'])

# Traducimos a los valores exactos usados en el entrenamiento
si_no = {'Sí': 'Yes', 'No': 'No'}
fumador = {
    'Nunca ha fumado': "'never smoked'",
    'Fuma actualmente': 'smokes',
    'Ex fumador': "'formerly smoked'",
    'No se sabe': 'Unknown',
}

# Dataframe que integra esos datos
datos = [[age, si_no[hypertension], si_no[heart_disease], si_no[ever_married],
          avg_glucose_level, fumador[smoking_status]]]
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
proba = float(modelo.predict_proba(data_preparada)[0][1])  # clase 1 = 'Yes'
riesgo_alto = Y_pred[0] == 1
color = '#dc2626' if riesgo_alto else '#059669'

# Factores de riesgo detectados en los datos ingresados
factores = []
if age >= 60:
    factores.append('Edad ≥ 60 años')
if hypertension == 'Sí':
    factores.append('Hipertensión')
if heart_disease == 'Sí':
    factores.append('Enfermedad cardíaca')
if smoking_status == 'Fuma actualmente':
    factores.append('Fuma actualmente')
elif smoking_status == 'Ex fumador':
    factores.append('Ex fumador')
if avg_glucose_level >= 140:
    factores.append('Glucosa elevada')


def grafica_gauge(probabilidad):
    """Donut de probabilidad con Altair."""
    color_riesgo = '#dc2626' if probabilidad >= 0.5 else '#059669'
    datos_g = pd.DataFrame({'tipo': ['riesgo', 'resto'], 'valor': [probabilidad, 1 - probabilidad]})
    arco = (
        alt.Chart(datos_g)
        .mark_arc(innerRadius=84, outerRadius=110, cornerRadius=10, padAngle=0.035)
        .encode(
            theta=alt.Theta('valor:Q', stack=True),
            color=alt.Color('tipo:N',
                            scale=alt.Scale(domain=['riesgo', 'resto'],
                                            range=[color_riesgo, '#e9edf5']),
                            legend=None),
            tooltip=[alt.Tooltip('tipo:N'), alt.Tooltip('valor:Q', format='.2%')],
        )
    )
    texto = (
        alt.Chart(pd.DataFrame({'t': [f'{probabilidad * 100:.1f}%']}))
        .mark_text(font='Poppins', fontSize=38, fontWeight=800, color=color_riesgo, dy=0)
        .encode(text='t:N')
    )
    sub = (
        alt.Chart(pd.DataFrame({'t': ['probabilidad de ataque']}))
        .mark_text(font='Poppins', fontSize=11, color='#64748b', dy=32)
        .encode(text='t:N')
    )
    return (arco + texto + sub).properties(width=430, height=260)


with col_der:
    with st.container(border=True):
        st.markdown('<p class="card-title">Resultado de la predicción</p>'
                    '<p class="card-sub">Probabilidad estimada por la red neuronal.</p>',
                    unsafe_allow_html=True)

        st.altair_chart(grafica_gauge(proba), width='stretch')

        st.markdown(f"""
        <div class="veredicto {'veredicto-alto' if riesgo_alto else 'veredicto-bajo'} fade-in">
            <div class="label">RIESGO DE ATAQUE AL CORAZÓN</div>
            <div class="valor">{'ALTO' if riesgo_alto else 'BAJO'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="seccion" style="margin-top:.9rem;">Factores detectados</p>',
                    unsafe_allow_html=True)
        if factores:
            chips = ''.join(f'<span class="factor">{f}</span>' for f in factores)
        else:
            chips = '<span class="factor factor-ok">Sin factores de riesgo evidentes</span>'
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; gap:.7rem; margin-top:.9rem;">
            <div class="mini" style="flex:1;">
                <div class="lbl">Prob. de ataque</div>
                <div class="val" style="color:{color};">{proba * 100:.1f}%</div>
            </div>
            <div class="mini" style="flex:1;">
                <div class="lbl">Prob. de no ataque</div>
                <div class="val">{(1 - proba) * 100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if riesgo_alto:
            st.warning('Se recomienda consultar con un profesional de la salud y controlar los factores de riesgo.')
        else:
            st.success('No se detecta un riesgo elevado con los datos ingresados. Mantén hábitos saludables.')

# ------------------------------------------------------------------ Pestañas de detalle
tab_datos, tab_modelos, tab_como = st.tabs(['Datos preparados', 'Comparación de modelos', '¿Cómo funciona?'])

with tab_datos:
    st.markdown('<p class="card-sub">Estas son las 9 variables que recibe la red neuronal, '
                'en el mismo orden del entrenamiento (dummies + columnas alineadas).</p>',
                unsafe_allow_html=True)
    st.dataframe(data_preparada, width='stretch')

with tab_modelos:
    comparacion = pd.DataFrame({
        'Modelo': ['Red Neuronal (MLP)', 'KNN (k=3)', 'Random Forest', 'Árbol de Decisión', 'SVM (RBF)'],
        'F1-macro': [0.906, 0.899, 0.875, 0.860, 0.840],
    }).set_index('Modelo')
    st.markdown('<p class="card-sub">F1-macro promedio en validación cruzada estratificada de 10 folds. '
                'Modelo desplegado: <b>Red Neuronal (MLP)</b>, el mejor F1 con gap train-test &lt; 0.05.</p>',
                unsafe_allow_html=True)
    c_tab, c_graf = st.columns([1, 1.3])
    with c_tab:
        st.dataframe(comparacion, width='stretch')
    with c_graf:
        st.bar_chart(comparacion, horizontal=True, color='#e11d48')

with tab_como:
    st.markdown('<p class="card-sub">Pipeline de preparación y entrenamiento seguido en el notebook.</p>',
                unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown('#### 1. Balanceo\nSe aplica **SMOTENC 1:1** con semilla fija (42) para equilibrar '
                'las clases generando casos sintéticos que respetan las variables categóricas.')
    p2.markdown('#### 2. Codificación\nLas variables categóricas se convierten con **get_dummies** '
                '(`drop_first=True` en binarias) y la variable objetivo con **LabelEncoder**.')
    p3.markdown('#### 3. Modelo\nRed neuronal **MLP tanh** de capas 100-50-25, con escalado '
                '**StandardScaler dentro del Pipeline** para evitar fuga de información.')
    p4.markdown('#### 4. Validación\n**StratifiedKFold de 10 folds** con métricas F1-macro, accuracy, '
                'precisión y recall. Se elige el modelo con mejor F1 y gap train-test < 0.05.')

st.markdown('<p style="color:#94a3b8; font-size:.8rem; margin-top:1rem;">Proyecto de Analítica de Datos · '
            'Validación Cruzada Clasificación · Desplegado con Streamlit</p>', unsafe_allow_html=True)

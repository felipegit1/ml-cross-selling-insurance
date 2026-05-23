import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================
# CONFIGURACIÓN APP
# =========================================

st.set_page_config(
    page_title="Modelo Venta Cruzada",
    layout="centered"
)

st.title("Modelo de Propensión a Venta Cruzada")

st.markdown("""
Aplicación basada en Machine Learning para estimar la probabilidad de adquisición de un producto SOAT por clientes del ramo Salud.
""")

# =========================================
# CARGAR MODELO
# =========================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_xgboost.pkl")

modelo = cargar_modelo()


# =========================================
# INPUTS USUARIO
# =========================================

st.subheader("Información del Cliente")

sexo = st.selectbox(
    "Sexo",
    ["M", "F"]
)

estado_civil = st.selectbox(
    "Estado Civil",
    [
        "Soltero",
        "Casado",
        "Un. Libre",
        "Separado",
        "Viudo",
        "No Especificado"
    ]
)

plan_pr = st.selectbox(
    "Plan",
    [
        "original plus",
        "ALTERNO AMPARADO",
        "Original Amparado",
        "Alterno plus",
        "PLAN AMBULATORIO",
        "OTRAS"
    ]
)

ciudad = st.selectbox(
    "Ciudad",
    [
        "BOGOTA",
        "MEDELLIN",
        "BUCARAMANGA",
        "CALI",
        "CARTAGENA",
        "OTRAS"
    ]
)

edad = st.slider(
    "Edad",
    18,
    100,
    35
)

cantidad_siniestros = st.number_input(
    "Cantidad de Siniestros",
    min_value=0,
    max_value=100,
    value=0
)

antiguedad_poliza = st.number_input(
    "Antigüedad Póliza (días)",
    min_value=0,
    max_value=20000,
    value=365
)

dias_restantes = st.number_input(
    "Días Restantes Vigencia",
    min_value=0,
    max_value=1000,
    value=180
)

cliente_premium = st.selectbox(
    "Cliente Premium",
    [0, 1]
)

prima_negativa = st.selectbox(
    "¿Presenta Prima Negativa?",
    [0, 1]
)


# =========================================
# FEATURE ENGINEERING
# =========================================

cliente_siniestros = (
    1 if cantidad_siniestros > 0 else 0
)

log_siniestros = np.log1p(
    cantidad_siniestros
)

log_antiguedad = np.log1p(
    antiguedad_poliza
)

# =========================================
# BOTÓN PREDICCIÓN
# =========================================

if st.button("Predecir Propensión"):

    data = pd.DataFrame({

        "SEXO_GENERO": [sexo],

        "ESTADO_CIVIL": [estado_civil],

        "PLAN_PR": [plan_pr],

        "CANTIDAD_SINIESTROS": [cantidad_siniestros],

        "CIUDAD": [ciudad],

        "EDAD": [edad],

        "ANTIGUEDAD_POLIZA": [antiguedad_poliza],

        "DIAS_RESTANTES_VIGENCIA": [dias_restantes],

        "CLIENTE_SINIESTROS": [cliente_siniestros],

        "LOG_SINIESTROS": [log_siniestros],

        "LOG_ANTIGUEDAD": [log_antiguedad],

        "CLIENTE_PREMIUM": [cliente_premium],

        "PRIMA_NEGATIVA": [prima_negativa]
    })

    # =====================================
    # PREDICCIÓN
    # =====================================

    probabilidad = modelo.predict_proba(data)[0][1]

    porcentaje = round(
        probabilidad * 100,
        2
    )

    st.subheader(
        f"Probabilidad de Compra: {porcentaje}%"
    )

    st.progress(int(porcentaje))

    # =====================================
    # RESULTADO NEGOCIO
    # =====================================

    if porcentaje >= 60:

        st.success(
            "Alta propensión a compra"
        )

        st.markdown("""
        ### Recomendación Comercial
        
        Cliente priorizable para campañas premium de cross-selling y estrategias de retención.
        """)

    elif porcentaje >= 40:

        st.warning(
            "Propensión media"
        )

        st.markdown("""
        ### Recomendación Comercial
        
        Cliente apto para campañas segmentadas y ofertas dirigidas.
        """)

    else:

        st.error(
            "Baja propensión"
        )

        st.markdown("""
        ### Recomendación Comercial
        
        Cliente no priorizable actualmente para campañas comerciales.
        """)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption("""
Proyecto Final Análisis de Machine Learning - Modelo de Propensión a Venta Cruzada
""")
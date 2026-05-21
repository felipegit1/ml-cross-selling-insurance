import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================
# CARGAR MODELO
# =========================================

modelo = joblib.load("modelo_xgboost.pkl")

# =========================================
# CONFIGURACIÓN
# =========================================

st.set_page_config(
    page_title="Modelo Venta Cruzada",
    layout="centered"
)

st.title("Modelo de Propensión a Venta Cruzada")

st.markdown("""
Esta aplicación estima la probabilidad de que un cliente adquiera un producto SOAT.
""")

# =========================================
# INPUTS
# =========================================

edad = st.slider(
    "Edad",
    18,
    100,
    35
)

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

ciudad = st.selectbox(
    "Ciudad",
    [
        "BOGOTA",
        "MEDELLIN",
        "BUCARAMANGA",
        "CALI",
        "BARRANQUILLA",
        "CARTAGENA",
        "OTRAS"
    ]
)

departamento = st.selectbox(
    "Departamento",
    [
        "BOGOTA",
        "ANTIOQUIA",
        "SANTANDER",
        "ATLANTICO",
        "CUNDINAMARCA",
        "BOLIVAR",
        "OTROS"
    ]
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
    max_value=15000,
    value=365
)

dias_restantes = st.number_input(
    "Días Restantes Vigencia",
    min_value=0,
    max_value=1000,
    value=180
)

total_primas = st.number_input(
    "Total Primas",
    min_value=0.0,
    value=1000000.0
)

cliente_premium = st.selectbox(
    "Cliente Premium",
    [0,1]
)

# =========================================
# BOTÓN PREDICCIÓN
# =========================================

if st.button("Predecir Propensión"):

    data = pd.DataFrame({
        
    	"EDAD": [edad],

    	"SEXO_GENERO": [sexo],

    	"ESTADO_CIVIL": [estado_civil],

    	"CIUDAD": [ciudad],

    	"DEPARTAMENTO": [departamento],

    	"CANTIDAD_SINIESTROS": [cantidad_siniestros],

    	"ANTIGUEDAD_POLIZA": [antiguedad_poliza],

    	"DIAS_RESTANTES_VIGENCIA": [dias_restantes],

    	"TOTAL_PRIMAS": [total_primas],

    	"CLIENTE_PREMIUM": [cliente_premium],

    # =====================================
    # VARIABLES FALTANTES
    # =====================================

    	"TIPO_PERSONA": ["Persona"],

    	"TIPO_DE_DOCUMENTO": ["CC"],

    	"PRODUCTO_x": ["SALUD"],

    	"CLIENTE_SINIESTROS": [
        1 if cantidad_siniestros > 0 else 0
    	],

    	"LOG_TOTAL_PRIMAS": [
        np.log1p(total_primas)
    	],

    	"LOG_SINIESTROS": [
        np.log1p(cantidad_siniestros)
    	],

    	"LOG_ANTIGUEDAD": [
        np.log1p(antiguedad_poliza)
    	]
    })

    probabilidad = modelo.predict_proba(data)[0][1]

    porcentaje = round(probabilidad * 100, 2)

    st.subheader(f"Probabilidad de Compra: {porcentaje}%")

    # =====================================
    # SEMÁFORO
    # =====================================

    if porcentaje >= 70:

        st.success(
            "Alta propensión a compra"
        )

        st.markdown("""
        Recomendación:
        Priorizar cliente en campañas premium de cross-selling.
        """)

    elif porcentaje >= 40:

        st.warning(
            "Propensión media"
        )

        st.markdown("""
        Recomendación:
        Cliente elegible para campañas segmentadas.
        """)

    else:

        st.error(
            "Baja propensión"
        )

        st.markdown("""
        Recomendación:
        No priorizar en campañas comerciales.
        """)
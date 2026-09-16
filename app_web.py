import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import tf_keras as keras
import streamlit as st

st.set_page_config(page_title="Evaluación Estructural", page_icon="🏗️", layout="centered")

st.title("🏗️ Evaluación de Salud Estructural")
st.write("Carga o toma una foto para analizar la estabilidad y grietas.")

@st.cache_resource
def cargar_modelo():
    modelo = keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r') as f:
        clases = [linea.strip() for linea in f.readlines()]
    return modelo, clases

with st.spinner("Cargando modelo de Inteligencia Artificial..."):
    modelo, clases = cargar_modelo()

opcion = st.radio("Selecciona origen de la imagen:", ("Usar Cámara", "Subir Imagen (Galería)"))

img_file_buffer = None
if opcion == "Usar Cámara":
    img_file_buffer = st.camera_input("Toma una foto de la estructura")
else:
    img_file_buffer = st.file_uploader("Sube una foto de la pared", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame_vis = cv2_img.copy()

    # 1. IA TEACHABLE MACHINE
    img_ia = cv2.resize(cv2_img, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(img_ia, dtype=np.float32).reshape(1, 224, 224, 3)
    img_normalizada = (img_array / 127.5) - 1

    prediccion = modelo.predict(img_normalizada, verbose=0)
    indice_clase = np.argmax(prediccion)
    nombre_clase = clases[indice_clase]
    probabilidad = prediccion[0][indice_clase] * 100
    es_grieta = "positivo" in nombre_clase.lower() or "crack" in nombre_clase.lower()

    # 2. OPENCV GEOMETRÍA
    gris = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    desenfocado = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(desenfocado, 50, 150)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ancho_grieta_px = 0
    angulo_grieta = 0.0

    if contornos and es_grieta:
        c = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            ancho_grieta_px = max(w, h)
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
            rect = cv2.minAreaRect(c)
            angulo_grieta = abs(rect[2])

    # 3. OPENCV INCLINACIÓN
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    inclinacion_muro = 0.0

    if lineas is not None:
        angulos_muro = []
        for linea in lineas:
            coords = linea.ravel()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                cv2.line(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                angulos_muro.append(np.degrees(np.arctan2(dx, dy)))
        if angulos_muro:
            inclinacion_muro = np.mean(angulos_muro)

    # 4. DIAGNÓSTICO DE RIESGO
    es_diagonal = 30.0 <= angulo_grieta <= 60.0

    st.subheader("📋 Resultados de la Evaluación")
    if es_grieta:
        if es_diagonal or inclinacion_muro > 3.0 or ancho_grieta_px > 150:
            st.error("🚨 **NIVEL DE RIESGO: CRÍTICO / PELIGROSO**\nGrieta estructural diagonal o desaplome detectado.")
        elif ancho_grieta_px > 50:
            st.warning("⚠️ **NIVEL DE RIESGO: PRECAUCIÓN / MODERADO**\nGrieta relevante por asentamiento.")
        else:
            st.info("🟡 **NIVEL DE RIESGO: LEVE / FISURA**\nFisura superficial sin riesgo severo.")
    else:
        st.success("✅ **NIVEL DE RIESGO: SEGURO**\nEstructura en buenas condiciones.")

    col1, col2, col3 = st.columns(3)
    col1.metric("IA Detección", f"{nombre_clase}", f"{probabilidad:.1f}%")
    col2.metric("Ángulo Grieta", f"{angulo_grieta:.1f}°")
    col3.metric("Ancho Grieta", f"{ancho_grieta_px} px")

    st.image(frame_vis, channels="BGR", caption="Trazado geométrico (Rojo: Grieta | Verde: Muro)")
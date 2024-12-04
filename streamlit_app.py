import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import gdown
import os
import asyncio

# Personalización con CSS
st.markdown("""
    <style>
    /* Estilo general para el diseño de las tarjetas */
    .card {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
        text-align: left;
        width: 90%;
        max-width: 500px;
        margin: 10px auto;
    }
    .card-title {
        font-size: 1.4em;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 10px;
    }
    .card-content {
        font-size: 1em;
        color: black;
    }
    .card-content b {
        color: #FF4B4B;
    }
    /* Estilo para ocultar el texto del uploader */
    .css-1p0chx5 {
        visibility: hidden;
    }
    .stFileUpload {
        visibility: hidden;
    }
    /* Mejorar la apariencia del componente file_uploader */
    .css-1v3vtrj {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #FF4B4B;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'Evaluacion.pt'
    gdrive_url = 'https://drive.google.com/file/d/1I4wO0UFpuPGOM9W24PdBPWuzDuYvNsmv/view?usp=sharing'
    if not os.path.exists(model_path):
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)
    return model

model = load_model()

classes = ['Acreo Corroído', 'Desprendimiento', 'Grietas']

# Diccionario de posibles causas por clase
causas_por_clase = {
    'Acreo Corroído': [
        "Exposición prolongada a ambientes húmedos.",
        "Presencia de sales en el entorno.",
        "Deficiente protección del recubrimiento de hormigón."
    ],
    'Desprendimiento': [
        "Fallas en el proceso constructivo.",
        "Impactos mecánicos o físicos.",
        "Reacciones químicas en los materiales."
    ],
    'Grietas': [
        "Sobrecarga estructural.",
        "Contracción del hormigón durante el curado.",
        "Asentamiento desigual de la base."
    ]
}

async def process_image(image, model, confidence):
    img = Image.open(image)
    results = await asyncio.to_thread(model, img, conf=confidence)
    return results

def main():
    st.title("Evaluación de Patologías en Hormigón")
    activities = ["Principal", "Subir imagen", "Tomar foto"]
    choice = st.sidebar.selectbox("Selecciona actividad", activities)
    st.sidebar.markdown('---')

    if choice == "Principal":
        st.markdown("<h4 style='color:white;'>Esta aplicación permite identificar y evaluar patologías en estructuras de hormigón, como grietas y corrosión, utilizando imágenes. Solo necesitas cargar una foto de la estructura, y la aplicación te mostrará los posibles defectos y sus causas para facilitar su diagnóstico y mantenimiento.</h4><br>", unsafe_allow_html=True)
        html_classesp = [f'<span style="padding:4px;border-radius:5px;background-color:#FF4B4B;color:white;">{cls}</span>' for cls in classes]
        st.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'>"
                    f"<h4 style='color:#FF4B4B;text-align:center;'>Patologías</h4>"
                    f"<p style='color:white;text-align:center;'>{' '.join(html_classesp)}</p>"
                    f"</div><br>", unsafe_allow_html=True)

    elif choice == "Subir imagen":
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        
        # Personalización del uploader para ocultar el texto
        image = st.file_uploader("", type=["png", "jpg", "jpeg", "gif"])
        
        if image:
            col1, col2 = st.columns([1, 1])
            col1.image(image, caption='Imagen original', use_container_width=True)

            with col2:
                with st.spinner('Procesando imagen...'):
                    results = asyncio.run(process_image(image, model, confidence_slider))
                    
                    if results:
                        annotated_frame = results[0].plot()
                        col2.image(annotated_frame, caption='Area Evaluada', use_container_width=True)

                        # Contador de detecciones
                        detections_count = {cls: 0 for cls in classes}

                        for result in results[0].boxes:
                            idx = int(result.cls.cpu().numpy()[0])
                            detected_class = classes[idx]
                            detections_count[detected_class] += 1

                        # Mostrar resultados consolidados en tarjetas
                        for detected_class, count in detections_count.items():
                            if count > 0:
                                # Organizar el contenido de las causas en HTML con listas
                                causes_html = "<ul>"
                                for causa in causas_por_clase[detected_class]:
                                    causes_html += f"<li>{causa}</li>"
                                causes_html += "</ul>"
                                
                                # Crear la tarjeta con el resultado
                                st.markdown(f"<div class='card'>"
                                            f"<div class='card-title'>{detected_class} ({count} detección(es))</div>"
                                            f"<div class='card-content'>"
                                            f"<b>Posibles causas:</b><br>"
                                            f"{causes_html} "
                                            f"</div>"
                                            f"</div>", unsafe_allow_html=True)
                    else:
                        st.write("No se detectaron objetos.")

    elif choice == "Tomar foto":
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        captured_image = st.camera_input("Toma una foto")

        if captured_image:
            col1, col2 = st.columns([1, 1])
            col1.image(captured_image, caption='Foto tomada', use_container_width=True)

            with col2:
                with st.spinner('Procesando foto...'):
                    results = asyncio.run(process_image(captured_image, model, confidence_slider))

                    if results:
                        annotated_frame = results[0].plot()
                        col2.image(annotated_frame, caption='Area Evaluada', use_container_width=True)

                        # Contador de detecciones
                        detections_count = {cls: 0 for cls in classes}

                        for result in results[0].boxes:
                            idx = int(result.cls.cpu().numpy()[0])
                            detected_class = classes[idx]
                            detections_count[detected_class] += 1

                        # Mostrar resultados consolidados en tarjetas
                        for detected_class, count in detections_count.items():
                            if count > 0:
                                # Organizar el contenido de las causas en HTML con listas
                                causes_html = "<ul>"
                                for causa in causas_por_clase[detected_class]:
                                    causes_html += f"<li>{causa}</li>"
                                causes_html += "</ul>"
                                
                                # Crear la tarjeta con el resultado
                                st.markdown(f"<div class='card'>"
                                            f"<div class='card-title'>{detected_class} ({count} detección(es))</div>"
                                            f"<div class='card-content'>"
                                            f"<b>Posibles causas:</b><br>"
                                            f"{causes_html} "
                                            f"</div>"
                                            f"</div>", unsafe_allow_html=True)
                    else:
                        st.write("No se detectaron objetos.")

if __name__ == "__main__":
    main()

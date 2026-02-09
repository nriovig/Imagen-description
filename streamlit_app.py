from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, MarianMTModel, MarianTokenizer
import torch
from PIL import Image
import streamlit as st

# --- Modelos ---
modelo = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
preprocesador = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizador = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Modelo de traducción inglés -> español
tokenizador_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
modelo_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# --- Dispositivo ---
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(dispositivo)
modelo_es.to(dispositivo)

# --- Parámetros ---
max_length = 16
num_beams = 4
parametros_generacion = {"max_length": max_length, "num_beams": num_beams}

# --- Función de traducción ---
def traducir_a_es(texto_en):
    tokens = tokenizador_es(texto_en, return_tensors="pt", padding=True).to(dispositivo)
    salida_ids = modelo_es.generate(**tokens)
    return tokenizador_es.decode(salida_ids[0], skip_special_tokens=True)

# --- Función para generar descripciones ---
def generar_descripcion(imagenes_subidas):
    imagenes = []
    for img in imagenes_subidas:
        i = Image.open(img)
        if i.mode != "RGB":
            i = i.convert("RGB")
        imagenes.append(i)

    valores_pixeles = preprocesador(images=imagenes, return_tensors="pt").pixel_values
    valores_pixeles = valores_pixeles.to(dispositivo)

    ids_salida = modelo.generate(valores_pixeles, **parametros_generacion)
    descripciones = tokenizador.batch_decode(ids_salida, skip_special_tokens=True)
    descripciones = [d.strip() for d in descripciones]

    # Traducir cada caption
    descripciones_es = [traducir_a_es(d) for d in descripciones]
    return descripciones_es

# --- Interfaz ---
def main():
    st.title("Aplicación de Descripción de Imágenes (Español)")

    imagen_subida = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if imagen_subida is not None:
        st.image(imagen_subida, caption="Imagen subida", use_column_width=True)

        if st.button("Generar Descripción"):
            descripciones = generar_descripcion([imagen_subida])
            st.write("Descripción generada:", descripciones[0])

if __name__ == "__main__":
    main()

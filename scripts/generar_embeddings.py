import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# --- Configuración ---
ARCHIVO_SMS = os.path.join('data', 'combined_limited.csv')  # Ruta al archivo .CSV o .py
EXTENSION = ARCHIVO_SMS.split('.')[-1]
COLUMNA_TEXTO_SMS = 'sms_text'  # Nombre de la columna con los SMS
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'
RUTA_EMBEDDINGS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_embeddings.npy'))
RUTA_TEXTOS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_texts.npy'))

print(f"Cargando el modelo de embeddings: {MODELO_EMBEDDING}...")
model = SentenceTransformer(MODELO_EMBEDDING)
print("Modelo cargado.")

# --- Cargar los SMS desde el archivo ---

# si el archivo es .py, cargar el archivo de sms desde la carpeta data
if ARCHIVO_SMS.endswith('.py'):
    sms_collection = pd.read_py(ARCHIVO_SMS)
    sms_collection = sms_collection[COLUMNA_TEXTO_SMS].dropna().tolist()
    print(f"Se cargaron {len(sms_collection)} SMS de la colección.")
else:
    df_sms = pd.read_csv(ARCHIVO_SMS)
    sms_collection = df_sms[COLUMNA_TEXTO_SMS].dropna().tolist()
    print(f"Se cargaron {len(sms_collection)} SMS de la colección.")

# --- Generar Embeddings para cada SMS ---
print("Generando embeddings para la colección de SMS...")
print(f"Total de SMS a procesar: {len(sms_collection)}")

# Iniciar cronómetro
start_time = time.time()

# Procesar en lotes para mostrar progreso más detallado
batch_size = 100
total_sms = len(sms_collection)
sms_embeddings = []

for i in range(0, total_sms, batch_size):
    batch = sms_collection[i:i + batch_size]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    sms_embeddings.extend(batch_embeddings)
    
    # Calcular progreso y ETA
    processed = min(i + batch_size, total_sms)
    progress = (processed / total_sms) * 100
    
    print(f"Procesados: {processed}/{total_sms} SMS ({progress:.1f}%)")
    
    # ETA simple basado en tiempo promedio por lote
    if i > 0:
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (i // batch_size + 1)
        remaining_batches = (total_sms - processed) // batch_size + (1 if total_sms % batch_size > 0 else 0)
        eta_seconds = remaining_batches * avg_time_per_batch
        eta_minutes = eta_seconds / 60
        print(f"ETA aproximado: {eta_minutes:.1f} minutos")

sms_embeddings = np.array(sms_embeddings)
print("Embeddings generados.")

# Crear carpeta embeddings si no existe
os.makedirs('embeddings', exist_ok=True)

# Crear la estructura de directorios completa para las rutas de salida
os.makedirs(os.path.dirname(RUTA_EMBEDDINGS), exist_ok=True)
os.makedirs(os.path.dirname(RUTA_TEXTOS), exist_ok=True)

# Guardar embeddings y textos
np.save(RUTA_EMBEDDINGS, sms_embeddings)
np.save(RUTA_TEXTOS, np.array(sms_collection))

print(f"Embeddings y textos guardados en la carpeta 'embeddings/'.")
print("Proceso de generación de embeddings completado.") 
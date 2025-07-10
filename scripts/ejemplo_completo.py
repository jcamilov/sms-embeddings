"""
Ejemplo completo del sistema de embeddings y búsqueda semántica para SMS.

Este script demuestra:
1. Carga de embeddings pre-generados
2. Búsqueda semántica con diferentes consultas
3. Análisis de resultados
4. Exportación de resultados
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from utils import exportar_resultados_json, calcular_estadisticas_similitud, formatear_similitud

# --- about the model ---
# Model	                                   Size	    Speed	    Quality
# all-MiniLM-L3-v2	                       ~60MB	⚡⚡⚡	    ⭐⭐
# all-MiniLM-L6-v2	                       ~80MB	⚡⚡	      ⭐⭐⭐
# paraphrase-multilingual-MiniLM-L12-v2	   ~117MB	⚡  	       ⭐⭐⭐⭐

# --- Configuración ---
ARCHIVO_SMS = os.path.join('data', 'combined_limited.csv')
EXTENSION = ARCHIVO_SMS.split('.')[-1]
COLUMNA_TEXTO_SMS = 'sms_text'
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'

# Rutas de los archivos de embeddings
RUTA_EMBEDDINGS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_embeddings.npy'))
RUTA_TEXTOS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_texts.npy'))

def cargar_sistema():
    """Carga el modelo y los embeddings."""
    print("Cargando sistema de embeddings...")
    
    # Cargar modelo
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    print("✓ Modelo cargado")
    
    # Cargar embeddings y textos
    try:
        embeddings = np.load(RUTA_EMBEDDINGS)
        textos = np.load(RUTA_TEXTOS, allow_pickle=True)
        print(f"✓ Embeddings cargados: {embeddings.shape}")
        print(f"✓ Textos cargados: {len(textos)} SMS")
        return modelo, embeddings, textos
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos de embeddings.")
        print("Ejecuta primero: python scripts/generar_embeddings.py")
        return None, None, None

def buscar_sms_similares(consulta, embeddings, textos, modelo, top_k=3):
    """Busca SMS similares a la consulta."""
    embedding_consulta = modelo.encode([consulta])
    similitudes = cosine_similarity(embedding_consulta, embeddings)[0]
    indices_top = np.argsort(similitudes)[::-1][:top_k]
    
    resultados = []
    for idx in indices_top:
        resultados.append({
            'texto': textos[idx],
            'similitud': similitudes[idx],
            'indice': idx
        })
    
    return resultados

def mostrar_resultados(consulta, resultados):
    """Muestra los resultados de forma clara."""
    print(f"\n🔍 Resultados para: '{consulta}'")
    print("=" * 60)
    
    if not resultados:
        print("No se encontraron resultados.")
        return
    
    for i, resultado in enumerate(resultados, 1):
        print(f"\n{i}. 📱 SMS (Similitud: {formatear_similitud(resultado['similitud'])})")
        print(f"   📍 Índice: {resultado['indice']}")
        print(f"   💬 Texto: {resultado['texto']}")
        print("-" * 40)

# Imprime en consola un par de ejemplos de SMS y estadisticas de la coleccion
def analizar_coleccion(textos):
    """Analiza la colección de SMS."""
    print("\n📊 ANÁLISIS DE LA COLECCIÓN")
    print("=" * 40)
    
    # Estadísticas básicas
    longitudes = [len(texto) for texto in textos]
    print(f"Total de SMS: {len(textos)}")
    print(f"Longitud promedio: {np.mean(longitudes):.1f} caracteres")
    print(f"Longitud mínima: {min(longitudes)} caracteres")
    print(f"Longitud máxima: {max(longitudes)} caracteres")
    
    # Ejemplos de SMS
    print(f"\n📝 Ejemplos de SMS en la colección:")
    for i in range(min(3, len(textos))):
        print(f"   {i+1}. {textos[i][:50]}...")

def ejemplo_busquedas(modelo, embeddings, textos):
    """Ejecuta búsquedas de ejemplo."""
    consultas_ejemplo = [
        "Shop till u Drop, IS IT YOU, either 10K, 5K, £500 Cash or £100 Travel voucher, Call now, 09064011000. NTT PO Box CR01327BT fixedline Cost 150ppm mobile vary",
        "Te extraño mucho",
        "Your K.Y.C has been updated successfully, you will get 1205 cashback in your wallet, To get cashback click here Link http://8629a7f1.ngrok.io",
        "hey mom, I'll be there at 3pm",
        "Hello baby, Karla here. You make any plans for today? If not, email me...http://zglptv.com/no2hrZB"
    ]
    
    print("\n🎯 BÚSQUEDAS DE EJEMPLO")
    print("=" * 40)
    
    for consulta in consultas_ejemplo:
        resultados = buscar_sms_similares(consulta, embeddings, textos, modelo, top_k=3)
        mostrar_resultados(consulta, resultados)

def exportar_ejemplo(consulta, resultados):
    """Exporta resultados de ejemplo."""
    nombre_archivo = exportar_resultados_json(resultados, consulta, f"ejemplo_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    print(f"\n💾 Resultados exportados a: {nombre_archivo}")

def main():
    """Función principal del ejemplo."""
    print("🚀 SISTEMA DE EMBEDDINGS Y BÚSQUEDA SEMÁNTICA")
    print("=" * 60)
    
    # Cargar sistema
    modelo, embeddings, textos = cargar_sistema()
    if modelo is None:
        return
    
    # Analizar colección
    analizar_coleccion(textos)
    
    # Ejecutar búsquedas de ejemplo
    #ejemplo_busquedas(modelo, embeddings, textos)
    
    # Ejemplo de exportación
    consulta_ejemplo = "Hola, ¿cómo estás?"
    resultados_ejemplo = buscar_sms_similares(consulta_ejemplo, embeddings, textos, modelo, top_k=3)
    exportar_ejemplo(consulta_ejemplo, resultados_ejemplo)
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\n💡 Para usar el sistema interactivamente:")
    print("   python scripts/busqueda_semantica.py")
    print("   python scripts/busqueda_avanzada.py")

if __name__ == "__main__":
    main() 
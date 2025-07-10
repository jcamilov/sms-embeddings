import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Configuración ---
ARCHIVO_SMS = os.path.join('data', 'combined_limited.csv')
EXTENSION = ARCHIVO_SMS.split('.')[-1]
COLUMNA_TEXTO_SMS = 'sms_text'
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'

# Rutas de los archivos de embeddings
RUTA_EMBEDDINGS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_embeddings.npy'))
RUTA_TEXTOS = os.path.join('embeddings', ARCHIVO_SMS.replace('.' + EXTENSION, '_texts.npy'))

def cargar_embeddings_y_textos():
    """Carga los embeddings y textos guardados previamente."""
    try:
        embeddings = np.load(RUTA_EMBEDDINGS)
        textos = np.load(RUTA_TEXTOS, allow_pickle=True)
        print(f"Embeddings cargados: {embeddings.shape}")
        print(f"Textos cargados: {len(textos)} SMS")
        return embeddings, textos
    except FileNotFoundError:
        print("Error: No se encontraron los archivos de embeddings.")
        print("Ejecuta primero: python scripts/generar_embeddings.py")
        return None, None

def buscar_sms_similares(consulta, embeddings, textos, modelo, top_k=3):
    """
    Busca los SMS más similares a la consulta usando similitud coseno.
    
    Args:
        consulta: Texto de búsqueda
        embeddings: Array de embeddings de la colección
        textos: Lista de textos originales
        modelo: Modelo de SentenceTransformer
        top_k: Número de resultados a retornar
    
    Returns:
        Lista de tuplas (texto, similitud, índice)
    """
    # Generar embedding de la consulta
    embedding_consulta = modelo.encode([consulta])
    
    # Calcular similitud coseno con todos los embeddings
    similitudes = cosine_similarity(embedding_consulta, embeddings)[0]
    
    # Obtener los índices de los top_k más similares
    indices_top = np.argsort(similitudes)[::-1][:top_k]
    
    # Crear lista de resultados
    resultados = []
    for idx in indices_top:
        resultados.append({
            'texto': textos[idx],
            'similitud': similitudes[idx],
            'indice': idx
        })
    
    return resultados

def mostrar_resultados(resultados):
    """Muestra los resultados de búsqueda de forma clara."""
    print("\n" + "="*60)
    print("RESULTADOS DE BÚSQUEDA SEMÁNTICA")
    print("="*60)
    
    for i, resultado in enumerate(resultados, 1):
        print(f"\n{i}. SMS (Similitud: {resultado['similitud']:.3f})")
        print(f"   Índice: {resultado['indice']}")
        print(f"   Texto: {resultado['texto']}")
        print("-" * 40)

def main():
    """Función principal del script de búsqueda semántica."""
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    print("Modelo cargado.")
    
    # Cargar embeddings y textos
    embeddings, textos = cargar_embeddings_y_textos()
    if embeddings is None:
        return
    
    print("\n" + "="*60)
    print("BÚSQUEDA SEMÁNTICA DE SMS")
    print("="*60)
    print("Escribe 'salir' o 'q' para terminar.")
    print("Escribe 'ayuda' para ver comandos disponibles.")
    
    while True:
        print("\n" + "-"*40)
        consulta = input("Ingresa tu consulta SMS: ").strip()
        
        if consulta.lower() == 'salir' or consulta.lower() == 'q':
            print("¡Hasta luego!")
            break
        elif consulta.lower() == 'ayuda':
            print("\nComandos disponibles:")
            print("- 'salir' o 'q': Terminar el programa")
            print("- 'ayuda': Mostrar esta ayuda")
            print("- Cualquier texto: Buscar SMS similares")
            continue
        elif not consulta:
            print("Por favor, ingresa una consulta válida.")
            continue
        
        # Realizar búsqueda
        print(f"\nBuscando SMS similares a: '{consulta}'")
        resultados = buscar_sms_similares(consulta, embeddings, textos, modelo, top_k=3)
        
        # Mostrar resultados
        mostrar_resultados(resultados)

if __name__ == "__main__":
    main() 
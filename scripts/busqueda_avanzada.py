import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from utils import exportar_resultados_json, calcular_estadisticas_similitud, formatear_similitud

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

def buscar_sms_similares(consulta, embeddings, textos, modelo, top_k=5, umbral_similitud=0.0):
    """
    Busca los SMS más similares a la consulta usando similitud coseno.
    
    Args:
        consulta: Texto de búsqueda
        embeddings: Array de embeddings de la colección
        textos: Lista de textos originales
        modelo: Modelo de SentenceTransformer
        top_k: Número de resultados a retornar
        umbral_similitud: Umbral mínimo de similitud (0.0 a 1.0)
    
    Returns:
        Lista de diccionarios con resultados
    """
    # Generar embedding de la consulta
    embedding_consulta = modelo.encode([consulta])
    
    # Calcular similitud coseno con todos los embeddings
    similitudes = cosine_similarity(embedding_consulta, embeddings)[0]
    
    # Filtrar por umbral de similitud
    indices_filtrados = np.where(similitudes >= umbral_similitud)[0]
    
    if len(indices_filtrados) == 0:
        return []
    
    # Obtener los índices de los top_k más similares
    similitudes_filtradas = similitudes[indices_filtrados]
    indices_top = indices_filtrados[np.argsort(similitudes_filtradas)[::-1][:top_k]]
    
    # Crear lista de resultados
    resultados = []
    for idx in indices_top:
        resultados.append({
            'texto': textos[idx],
            'similitud': similitudes[idx],
            'indice': idx
        })
    
    return resultados

def exportar_resultados_csv(resultados, consulta):
    """
    Exporta los resultados de búsqueda a formato CSV.
    
    Args:
        resultados: Lista de resultados de búsqueda
        consulta: Texto de la consulta original
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    datos_exportar = []
    for i, resultado in enumerate(resultados, 1):
        datos_exportar.append({
            'ranking': i,
            'texto': resultado['texto'],
            'similitud': float(resultado['similitud']),  # Convertir float32 a float
            'indice': resultado['indice']
        })
    
    df = pd.DataFrame(datos_exportar)
    nombre_archivo = f"resultados_busqueda_{timestamp}.csv"
    df.to_csv(nombre_archivo, index=False, encoding='utf-8')
    
    return nombre_archivo

def mostrar_estadisticas(resultados):
    """Muestra estadísticas de los resultados de búsqueda."""
    if not resultados:
        print("No se encontraron resultados.")
        return
    
    stats = calcular_estadisticas_similitud(resultados)
    print(f"\nEstadísticas de similitud:")
    print(f"  Máxima: {formatear_similitud(stats['maxima'])}")
    print(f"  Mínima: {formatear_similitud(stats['minima'])}")
    print(f"  Promedio: {formatear_similitud(stats['promedio'])}")
    print(f"  Mediana: {formatear_similitud(stats['mediana'])}")
    print(f"  Total: {stats['total']}")

def main():
    """Función principal del script de búsqueda avanzada."""
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    print("Modelo cargado.")
    
    # Cargar embeddings y textos
    embeddings, textos = cargar_embeddings_y_textos()
    if embeddings is None:
        return
    
    print("\n" + "="*60)
    print("BÚSQUEDA SEMÁNTICA AVANZADA")
    print("="*60)
    print("Comandos disponibles:")
    print("- 'config': Configurar parámetros de búsqueda")
    print("- 'exportar': Exportar últimos resultados")
    print("- 'estadisticas': Mostrar estadísticas")
    print("- 'salir': Terminar el programa")
    print("- Cualquier texto: Buscar SMS similares")
    
    # Configuración por defecto
    config = {
        'top_k': 5,
        'umbral_similitud': 0.0,
        'ultimos_resultados': None,
        'ultima_consulta': None
    }
    
    while True:
        print("\n" + "-"*40)
        comando = input("Comando o consulta: ").strip()
        
        if comando.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        elif comando.lower() == 'config':
            print("\nConfiguración actual:")
            print(f"  Top K: {config['top_k']}")
            print(f"  Umbral de similitud: {config['umbral_similitud']}")
            
            try:
                nuevo_top_k = input("Nuevo Top K (Enter para mantener): ").strip()
                if nuevo_top_k:
                    config['top_k'] = int(nuevo_top_k)
                
                nuevo_umbral = input("Nuevo umbral de similitud (0.0-1.0, Enter para mantener): ").strip()
                if nuevo_umbral:
                    config['umbral_similitud'] = float(nuevo_umbral)
                
                print("Configuración actualizada.")
            except ValueError:
                print("Error: Valores inválidos. Configuración no cambiada.")
        
        elif comando.lower() == 'exportar':
            if config['ultimos_resultados'] is not None:
                formato = input("Formato (json/csv): ").strip().lower()
                if formato not in ['json', 'csv']:
                    formato = 'json'
                
                if formato == 'json':
                    nombre_archivo = exportar_resultados_json(
                        config['ultimos_resultados'], 
                        config['ultima_consulta']
                    )
                    print(f"Resultados exportados a: {nombre_archivo}")
                else:
                    # Exportar a CSV
                    nombre_archivo = exportar_resultados_csv(
                        config['ultimos_resultados'], 
                        config['ultima_consulta']
                    )
                    print(f"Resultados exportados a: {nombre_archivo}")
            else:
                print("No hay resultados para exportar. Realiza una búsqueda primero.")
        
        elif comando.lower() == 'estadisticas':
            if config['ultimos_resultados'] is not None:
                mostrar_estadisticas(config['ultimos_resultados'])
            else:
                print("No hay resultados para analizar. Realiza una búsqueda primero.")
        
        elif not comando:
            print("Por favor, ingresa un comando o consulta válida.")
            continue
        
        else:
            # Realizar búsqueda
            print(f"\nBuscando SMS similares a: '{comando}'")
            resultados = buscar_sms_similares(
                comando, 
                embeddings, 
                textos, 
                modelo, 
                top_k=config['top_k'],
                umbral_similitud=config['umbral_similitud']
            )
            
            # Guardar para exportar
            config['ultimos_resultados'] = resultados
            config['ultima_consulta'] = comando
            
            # Mostrar resultados
            if resultados:
                print(f"\nEncontrados {len(resultados)} resultados:")
                for i, resultado in enumerate(resultados, 1):
                    print(f"\n{i}. SMS (Similitud: {formatear_similitud(resultado['similitud'])})")
                    print(f"   Índice: {resultado['indice']}")
                    print(f"   Texto: {resultado['texto']}")
                    print("-" * 40)
            else:
                print("No se encontraron resultados con los criterios especificados.")

if __name__ == "__main__":
    main() 
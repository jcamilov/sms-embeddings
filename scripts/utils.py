"""
Utilidades para el sistema de embeddings y búsqueda semántica.
"""

import json
import numpy as np
from datetime import datetime

def serializar_resultados(resultados):
    """
    Convierte los resultados de búsqueda a un formato JSON serializable.
    
    Args:
        resultados: Lista de diccionarios con resultados de búsqueda
    
    Returns:
        Lista de diccionarios serializables
    """
    resultados_serializables = []
    for resultado in resultados:
        resultados_serializables.append({
            'texto': str(resultado['texto']),
            'similitud': float(resultado['similitud']),  # Convertir float32 a float
            'indice': int(resultado['indice'])
        })
    return resultados_serializables

def exportar_resultados_json(resultados, consulta, nombre_archivo=None):
    """
    Exporta resultados a formato JSON.
    
    Args:
        resultados: Lista de resultados de búsqueda
        consulta: Texto de la consulta original
        nombre_archivo: Nombre del archivo (opcional)
    
    Returns:
        Nombre del archivo generado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convertir resultados a formato serializable
    resultados_serializables = serializar_resultados(resultados)
    
    # Preparar datos para exportar
    datos_json = {
        'consulta': consulta,
        'timestamp': timestamp,
        'total_resultados': len(resultados_serializables),
        'resultados': resultados_serializables
    }
    
    # Generar nombre de archivo si no se proporciona
    if nombre_archivo is None:
        nombre_archivo = f"resultados_busqueda_{timestamp}.json"
    
    # Escribir archivo
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        json.dump(datos_json, f, ensure_ascii=False, indent=2)
    
    return nombre_archivo

def calcular_estadisticas_similitud(resultados):
    """
    Calcula estadísticas de similitud de los resultados.
    
    Args:
        resultados: Lista de resultados de búsqueda
    
    Returns:
        Diccionario con estadísticas
    """
    if not resultados:
        return {
            'maxima': 0.0,
            'minima': 0.0,
            'promedio': 0.0,
            'mediana': 0.0,
            'total': 0
        }
    
    similitudes = [float(r['similitud']) for r in resultados]
    
    return {
        'maxima': max(similitudes),
        'minima': min(similitudes),
        'promedio': np.mean(similitudes),
        'mediana': np.median(similitudes),
        'total': len(similitudes)
    }

def formatear_similitud(valor):
    """
    Formatea un valor de similitud para mostrar.
    
    Args:
        valor: Valor de similitud (float32 o float)
    
    Returns:
        String formateado
    """
    return f"{float(valor):.3f}" 
# SMS Classifier Embeddings

Este proyecto permite generar embeddings y realizar búsqueda semántica sobre una colección de mensajes SMS.

## Estructura del proyecto

- `data/`: Archivos de datos de entrada (ej: `mis_sms.csv`)
- `embeddings/`: Embeddings y textos generados
- `scripts/`: Scripts de procesamiento y utilidades
- `requirements.txt`: Dependencias del proyecto

## Requisitos
- Python 3.13.5
- Recomendado: uso de entorno virtual

## Instalación

```bash
python3 -m venv venv
# Activar el entorno virtual:
# En Windows PowerShell:
.\venv\Scripts\Activate.ps1
# En Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

### 1. Generar Embeddings

Coloca tu archivo CSV con los SMS en la carpeta `data/` y asegúrate de que la columna de texto se llame `sms_text` (o ajusta el script). Luego ejecuta:

```bash
python scripts/generar_embeddings.py
```

Esto generará los archivos de embeddings en la carpeta `embeddings/`.

### 2. Búsqueda Semántica

#### Búsqueda Básica
```bash
python scripts/busqueda_semantica.py
```

#### Búsqueda Avanzada (con exportación y configuración)
```bash
python scripts/busqueda_avanzada.py
```

**Características de la búsqueda avanzada:**
- Configurar número de resultados (Top K)
- Establecer umbral de similitud mínima
- Exportar resultados a JSON o CSV
- Ver estadísticas de similitud
- Interfaz interactiva con comandos

### 3. Estructura de Archivos Generados

```
embeddings/
├── combined_limited_embeddings.npy  # Embeddings vectoriales
└── combined_limited_texts.npy       # Textos originales
```

## Características

- **Generación de embeddings**: Procesamiento por lotes con barra de progreso
- **Búsqueda semántica**: Usando similitud coseno
- **Exportación**: Resultados en formato JSON o CSV
- **Configuración flexible**: Ajuste de parámetros de búsqueda
- **Interfaz interactiva**: Comandos fáciles de usar 
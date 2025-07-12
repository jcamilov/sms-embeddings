# Convierte un archivo .py a un archivo .csv
import os
import sys
import csv
import re

def convert_py_to_csv():
    """
    Lee un archivo Python de la carpeta /data y lo convierte a CSV.
    El nombre del archivo está hardcodeado.
    """
    
    # Nombre del archivo hardcodeado
    input_filename = "testdataset20.py"
    input_path = os.path.join("data", input_filename)
    
    # Generar nombre del archivo de salida
    output_filename = input_filename.replace(".py", ".csv")
    output_path = os.path.join("data", output_filename)
    
    try:
        # Leer el archivo Python
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print("🔄 Procesando archivo...")
        
        # Crear un namespace local para ejecutar el código
        local_namespace = {}
        
        # Ejecutar el archivo Python en el namespace local
        exec(content, {}, local_namespace)
        
        # Obtener la variable sms_data
        if 'sms_data' not in local_namespace:
            raise ValueError("No se encontró la variable 'sms_data' en el archivo")
        
        sms_data = local_namespace['sms_data']
        
        if sms_data:
            # Recopilar todos los campos posibles de todos los registros
            all_fieldnames = set()
            for row in sms_data:
                all_fieldnames.update(row.keys())
            
            # Convertir a lista y ordenar para consistencia
            fieldnames = sorted(list(all_fieldnames))
            
            # Escribir a CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Escribir todos los datos
                for row in sms_data:
                    writer.writerow(row)
                
                print(f"✅ Archivo convertido exitosamente!")
                print(f"📁 Entrada: {input_path}")
                print(f"📁 Salida: {output_path}")
                print(f"📊 Registros procesados: {len(sms_data)}")
                print(f"📋 Campos encontrados: {', '.join(fieldnames)}")
        else:
            print("❌ No se encontraron datos para convertir")
            
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {input_path}")
    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_py_to_csv()


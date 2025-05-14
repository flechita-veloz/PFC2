import os
import subprocess

# Lista de escenas
# 'tree', 
escenas = ['alley', 'bicycle', 'dormitory', 'fence', 
           'flower', 'kitchen', 'livingroom', 'snowman', 
           'staircase', 'street', 'piano']  # Añade todas las escenas que necesites

# Directorio donde están los datasets y las salidas
dataset_dir = 'datasets/'
output_dir = 'output/'

# Puerto para el entrenamiento
port = 1111

# Iterar sobre cada escena y ejecutar el entrenamiento
for escena in escenas:
    # Comando para ejecutar
    command = [
        'python', 'train.py',
        '-r', '4',  # Número de repeticiones
        '-s', os.path.join(dataset_dir, escena),  # Ruta al dataset
        '-m', os.path.join(output_dir, escena + "_improved16"),  # Ruta al output
        '--port', str(port),  # Puerto
        '--eval'  # Evaluación al final
    ]
    
    # Imprimir el comando para depuración (opcional)
    print(f"Ejecutando: {' '.join(command)}")
    
    # Ejecutar el comando
    subprocess.run(command)

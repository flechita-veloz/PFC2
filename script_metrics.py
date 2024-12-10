import os
import subprocess

escenas =  ['tree', 'piano', 'alley', 'bicycle', 'dormitory', 'fence', 
           'flower', 'kitchen', 'livingroom', 'snowman', 
           'staircase', 'street'] 

# Directorio donde están los datasets y las salidas
output_dir = 'output/'

# Iterar sobre cada escena y ejecutar el entrenamiento
for escena in escenas:
    # Comando para ejecutar
    command = [
        'python', 'metrics.py',
        '-m', os.path.join(output_dir, escena) # Ruta al output
    ]
    
    # Imprimir el comando para depuración (opcional)
    print(f"Ejecutando: {' '.join(command)}")
    
    # Ejecutar el comando
    subprocess.run(command)

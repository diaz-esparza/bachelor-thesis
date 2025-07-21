# Modelos de aprendizaje profundo para la detección de melanomas en imágenes
Este repositorio contiene todo el *codebase* empleado para la realización de mi Trabajo de Fin de Grado.

En este Trabajo, empleamos principalmente PyTorch y Lightning para entrenar arquitecturas modernas en la tarea de la detección de patologías de la piel. El código contiene implementaciones de un sistema rudimentario, pero también introduce opciones para la realización de sistemas más complejos, incluyendo el entrenamiento con un conglomerado de todos los conjuntos de datos, la incorporación de Redes Neuronales Adversarias en cuanto al Dominio, y la introducción de pasos intermedios de pre-entrenamiento.

## Instalación

El programa de momento es únicamente compatible con sistemas Linux, principalmente porque usamos `/dev/shm` para una optimización en el cargado de datos. Para ejecutar el programa en sistemas Windows, el uso de WSL2 debería de funcionar sin mayores problemas.

Para ejecutar el código y reproducir los resultados del trabajo, lo primero que se necesita es un entorno con `Python3.11`. También se necesitarán instalar los paquetes empleados en el proyecto mediante el siguiente comando:

```
python3 -m pip install -r requirements.txt
```

Será necesario además obtener los conjuntos de datos empleados en el trabajo. Estos mismos no están incluidos en el repositorio dado su gran tamaño, pero el archivo `DESCARGA_DATASETS.md` incluye toda la información necesaria para su descarga, e incluimos además un herramienta en el mismo direcorio para ejecutar automáticamente todos los pasos de preprocesamiento empleados originalmente.

Una vez se ha preparado el repositorio, se pueden ejecutar los experimentos realizados en el trabajo mediante el script `train.py`. Simplemente consulta su ayuda para acceder a todas las configuraciones que ofrece el programa:

```
python3 train.py -h
```

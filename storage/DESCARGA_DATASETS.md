Para poder emplear todos los conjuntos de datos empleados en el proyecto,
uno debe primero descargar todos los zips de sus respectivos repositorios públicos.

A continuación se muestran todos los enlaces para obtener estos archivos. 4/6 de estos *datasets* están en Kaggle, por lo que su descarga en servidores es sencilla gracias a la API de descarga del sitio (y en el caso de SIIM-ISIC y BCN-20000, es posible obtener los enlaces directamente o interceptar la petición de descarga en un navegador convencional para pasar la dirección a cURL):
- [MED-NODE](https://www.kaggle.com/datasets/prabhavsanga/med-node/data)
- [HAM-10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- En el caso de [SIIM-ISIC](https://www.kaggle.com/c/siim-isic-melanoma-classification/data), su tamaño en Kaggle es demasiado grande, pero su [conjunto de imágenes en formato JPEG](https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip) puede ser obtenido a partir de un enlace directo.
- [SLICE-3D](https://www.kaggle.com/c/isic-2024-challenge/data)
- [PAD-UFES](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer)
- [BCN-20000](https://api.isic-archive.com/collections/249/)

Una vez descargados, el resto del proceso puede ser completado renombrano cada comprimido de la siguiente manera:
- HAM-10000 -> **ham.zip**
- MED-NODE -> **med.zip**
- SIIM-ISIC -> **siim.zip**
- SLICE-3D -> **slice.zip**
- PAD-UFES -> **pad.zip**
- BCN-20000 -> **bcn.zip**

A partir de ahí, el script de instalación se encargará del resto de pasos,
incluyendo la extracción de paquetes y la realización de dos pasos de preprocesamiento (redimensionamiento en SIIM-ISIC y recortado en BCN-20000). Simplemente ejecuta `python3 dataset_installation.py`

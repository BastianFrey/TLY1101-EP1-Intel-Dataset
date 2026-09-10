# Clasificación de Escenarios Naturales (Intel Image Dataset) - MLP

## 1. Comprensión del Negocio (Business Understanding)
**Descripción del problema de negocio:**
En aplicaciones de turismo, bienes raíces o indexación de imágenes satelitales, clasificar visualmente el entorno natural o urbano es un proceso costoso si se hace manualmente. El objetivo de este proyecto es automatizar la clasificación de escenarios fotográficos mediante técnicas de Deep Learning.

**Objetivos del proyecto:**
Desarrollar un modelo base de red neuronal (Perceptrón Multicapa - MLP) capaz de clasificar imágenes en 6 categorías distintas de escenarios naturales y urbanos, estableciendo un punto de partida (baseline) para futuras arquitecturas más complejas.

**Definición de KPIs:**
* **Macro F1-Score:** Alcanzar o superar un 0.50 global, priorizando el balance métrico entre todas las clases.
* **Accuracy General:** Superar el 45% de exactitud en el set de validación (considerando que el azar puro en 6 clases es ~16.6%).

---

## 2. Comprensión de los Datos (Data Understanding)
**Descripción de las fuentes de datos:**
Se utilizó el **Intel Image Classification Dataset**, el cual contiene imágenes a color (RGB) distribuidas en 6 clases: *buildings, forest, glacier, mountain, sea, street*.
* **Variante del proyecto:** Para adaptar los datos a la arquitectura MLP y evitar explosiones de memoria, se aplicó una reducción de dimensionalidad redimensionando las imágenes a `64x64` píxeles y estableciendo una semilla global (`SEED = 42`) para garantizar la reproducibilidad.

---

## 3. Preparación de los Datos (Data Preparation / EDA)
El proceso de preparación se implementó directamente en el flujo de Keras:
1. **Carga estructurada:** Se utilizaron generadores de directorios (`image_dataset_from_directory`) para extraer etiquetas basadas en las subcarpetas del dataset.
2. **Normalización:** Se implementó una capa `Rescaling(1./255)` para escalar los valores de los píxeles (0-255) a un rango continuo entre `[0, 1]`, facilitando la convergencia del gradiente.
3. **Aplanamiento:** Transformación de la matriz 2D (64x64x3) a un vector 1D de 12.288 dimensiones mediante la capa `Flatten`.

---

## 4. Modelamiento (Modeling)
Se diseñó un Perceptrón Multicapa (MLP) utilizando TensorFlow/Keras con la siguiente arquitectura en forma de embudo:
* **Entrada:** Capa Flatten (12.288 parámetros de entrada).
* **Capas Ocultas:** 
  * Densa (512 neuronas, ReLU) + Dropout (0.3)
  * Densa (256 neuronas, ReLU) + Dropout (0.3)
  * Densa (128 neuronas, ReLU)
* **Salida:** Densa (6 neuronas, Softmax) para clasificación multiclase.
* **Compilación:** Optimizador Adam, función de pérdida `sparse_categorical_crossentropy`.

---

## 5. Evaluación (Evaluation)
El modelo alcanzó una exactitud cercana al 51%. 
Al auditar los resultados a través de la **Matriz de Confusión**, se identificaron las siguientes limitaciones:
* La clase **Sea** (mar) presenta altos índices de confusión con **Buildings** (edificios).
* **Limitación detectada:** El aplanamiento de la imagen destruye la topología espacial. Un MLP no puede detectar patrones locales (como bordes rectos de un edificio vs. texturas de agua), limitándose a evaluar la intensidad de los píxeles desordenados.

---

## 6. Despliegue (Deployment)
**Instrucciones de reproducción:**
1. Clonar este repositorio.
2. Asegurar que las imágenes estén dentro de la ruta `data/seg_train/` y `data/seg_test/`.
3. Instalar las dependencias (`tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`).
4. Ejecutar el notebook `notebooks/EP1_Intel_MLP.ipynb` de principio a fin.
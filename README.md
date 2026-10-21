# Clasificación de Escenarios Naturales mediante Redes Neuronales Convolucionales (CNN) - Intel Image Dataset

**Asignatura:** TLY1101 - Técnicas Avanzadas de Machine Learning I  
**Evaluación:** Evaluación Parcial N°2 (Presentación y Defensa Técnica)

---

## 1. Descripción del Problema de Negocio, Objetivos y KPIs
En la gestión de activos digitales, monitoreo ambiental y catalogación turística, clasificar manualmente grandes volúmenes de fotografías geográficas y urbanas resulta costoso y propenso a errores humanos. Este proyecto aborda la clasificación automatizada de escenarios en 6 categorías (`buildings`, `forest`, `glacier`, `mountain`, `sea` y `street`) utilizando el conjunto de datos **Intel Image Classification Dataset**.

En la Evaluación Parcial 1 se abordó este problema mediante un Perceptrón Multicapa (MLP), el cual presentó limitaciones severas de generalización (Accuracy de **51,77%**) debido a la pérdida de la estructura espacial al aplanar la imagen (`Flatten`) desde la entrada. En esta segunda etapa se diseña e implementa una **Red Neuronal Convolucional (CNN)** para extraer características espaciales jerárquicas de forma automática.

### Objetivos del Proyecto
* **Objetivo General:** Diseñar, entrenar y evaluar una arquitectura CNN en TensorFlow/Keras que supere objetivamente el desempeño del modelo MLP previo sobre el mismo conjunto de prueba, reduciendo la complejidad paramétrica.
* **KPIs (Indicadores Clave de Desempeño):**
  1. **KPI 1 - Precisión Global (Accuracy en Test):** Superar el 75% de exactitud en el conjunto de prueba (frente al 51,77% del MLP). **Resultado alcanzado: 79,27% (Cumplido).**
  2. **KPI 2 - Eficiencia Computacional (Reducción de Parámetros):** Disminuir en al menos un 80% la cantidad total de parámetros entrenables respecto al MLP (6.456.966 parámetros). **Resultado alcanzado: 1.142.726 parámetros, reducción del 82,3% (Cumplido).**
  3. **KPI 3 - Consistencia Multiclase (F1-Score Macro):** Alcanzar un F1-Score Macro superior a 0.75 en las 6 clases. **Resultado alcanzado: 0.7946 (Cumplido).**

---

## 2. Fuente de Datos, Metodología (CRISP-DM) y Variante Propia

### Descripción del Dataset
El **Intel Image Classification Dataset** contiene aproximadamente 25.000 imágenes RGB distribuidas en directorios de entrenamiento (`seg_train` con ~14.000 imágenes) y prueba (`seg_test` con 3.000 imágenes) divididas en 6 clases: `buildings`, `forest`, `glacier`, `mountain`, `sea` y `street`.

### Definición y Justificación Técnica de la Variante Propia
Para resguardar la trazabilidad experimental con la Evaluación Parcial 1 y asegurar una validación metodológica rigurosa bajo el estándar **CRISP-DM**, se definió la siguiente variante:
1. **Conservación de Resolución (64×64×3) y Semilla (`SEED = 42`):** Se mantuvo el redimensionamiento a `64x64` píxeles y `BATCH_SIZE = 32` utilizados en el MLP para garantizar una comparación experimental 100% justa bajo las mismas condiciones de entrada.
2. **Partición Tripartita (Entrenamiento 80% / Validación 20% / Prueba Independiente):** A diferencia de la Evaluación 1, el directorio `seg_train` se subdividió utilizando `validation_split=0.2` y `seed=42` en un conjunto de **Entrenamiento (80%)** y uno de **Validación (20%)** para monitorear el sobreajuste época por época, reservando las **3.000 imágenes de `seg_test`** exclusivamente para la evaluación final a ciegas.

---

## 3. Preparación de Datos y Análisis Exploratorio (EDA)
* **Formato Tensorial 3D vs. Vector 1D:** Mientras el MLP obligaba a transformar cada imagen en un vector unidimensional de $64 \times 64 \times 3 = 12.288$ valores mediante una capa `Flatten` inicial (destruyendo la vecindad entre píxeles verticales y horizontales), en la CNN se conserva el tensor tridimensional `(64, 64, 3)`.
* **Normalización:** Se aplicó una capa `Rescaling(1./255)` para escalar los valores de píxeles del rango `[0, 255]` al intervalo continuo `[0, 1]`, evitando saturaciones numéricas y acelerando la convergencia del optimizador Adam.
* **Distribución de Clases en Prueba (3.000 imágenes):** El conjunto de prueba se encuentra balanceado: `buildings` (437), `forest` (474), `glacier` (553), `mountain` (525), `sea` (510) y `street` (501).

---

## 4. Diseño y Justificación de la Arquitectura CNN

Se implementó una arquitectura secuencial compuesta por **3 bloques de extracción de características (Convolución + MaxPooling)** seguidos de una **etapa de clasificación densa regularizada**:

| Capa | Configuración | Dimensión de Salida | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Input** | Tensor de entrada RGB | `64 × 64 × 3` | Conserva la topología espacial 2D y los 3 canales de color. |
| **Conv2D (`conv_1`)** | 32 filtros `3×3`, ReLU, `padding='same'` | `64 × 64 × 32` | Extrae características visuales de bajo nivel: bordes, líneas de horizonte y colores básicos. |
| **MaxPooling2D (`pool_1`)** | Ventana `2×2` | `32 × 32 × 32` | Reduce las dimensiones espaciales a la mitad, conservando las activaciones máximas y otorgando invarianza traslacional. |
| **Conv2D (`conv_2`)** | 64 filtros `3×3`, ReLU, `padding='same'` | `32 × 32 × 64` | Combina bordes simples para detectar texturas y patrones locales (follaje, roca, oleaje, ladrillos). |
| **MaxPooling2D (`pool_2`)** | Ventana `2×2` | `16 × 16 × 64` | Comprime la información espacial reduciendo el número de parámetros y evitando el sobreajuste. |
| **Conv2D (`conv_3`)** | 128 filtros `3×3`, ReLU, `padding='same'` | `16 × 16 × 128` | Captura ensamblajes complejos y partes de objetos (picos nevados, fachadas de edificios, copas de árboles). |
| **MaxPooling2D (`pool_3`)** | Ventana `2×2` | `8 × 8 × 128` | Genera la representación espacial compacta final de alta profundidad semántica. |
| **Flatten** | Aplanado de mapas | `8192` | Convierte el tensor `8 × 8 × 128` en un vector 1D de 8.192 características extraídas. |
| **Dense + Dropout** | 128 neuronas (ReLU) + `Dropout(0.4)` | `128` | Combina no-linealmente las características; el Dropout apaga el 40% de las neuronas al azar durante el entrenamiento para mitigar el *overfitting*. |
| **Dense (Salida)** | 6 neuronas (`softmax`) | `6` | Entrega la distribución de probabilidades para clasificar la imagen en una de las 6 clases. |

* **Hiperparámetros de Compilación y Entrenamiento:** Optimizador `Adam`, función de pérdida `sparse_categorical_crossentropy`, `BATCH_SIZE = 32` y `EPOCHS = 20`.

---

## 5. Evaluación del Desempeño de la CNN

### Reporte de Clasificación por Clase (Set de Prueba: 3.000 imágenes)

| Clase | Precision | Recall | F1-Score | Support (Imágenes) | Interpretación |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **buildings** | 0.80 | 0.74 | 0.77 | 437 | Buen desempeño, con confusión moderada frente a la clase `street`. |
| **forest** | **0.96** | **0.89** | **0.92** | 474 | **Mejor clase:** patrones de textura vertical (troncos/follaje) altamente distinguibles. |
| **glacier** | 0.75 | 0.69 | 0.72 | 553 | Clase más desafiante por su similitud cromática y morfológica con `mountain` y `sea`. |
| **mountain** | 0.76 | 0.74 | 0.75 | 525 | Desempeño sólido; sus errores se concentran en la frontera visual con `glacier`. |
| **sea** | 0.74 | 0.86 | 0.80 | 510 | Alto Recall (detecta el 86% de las imágenes de mar gracias a la línea del horizonte). |
| **street** | 0.78 | 0.85 | 0.81 | 501 | Excelente sensibilidad gracias a la perspectiva lineal de las calles. |
| **Accuracy Global** | | | **0.79** | **3000** | **79,27% de exactitud general en prueba.** |
| **Macro Avg** | **0.80** | **0.79** | **0.79** | **3000** | Desempeño equilibrado en todas las categorías. |

---

## 6. Comparación Experimental: Modelo MLP (EP1) vs. Modelo CNN (EP2)

| Métrica / Característica | Modelo MLP (EP1) | Modelo CNN (EP2) | Diferencia / Mejora |
| :--- | :---: | :---: | :--- |
| **Cantidad Total de Parámetros** | 6.456.966 | **1.142.726** | **-82,30% parámetros** (5,65 veces más eficiente). |
| **Accuracy (Prueba)** | 0.5177 (51,77%) | **0.7927 (79,27%)** | **+27,50 puntos porcentuales** de exactitud. |
| **Precision Macro** | 0.5157 | **0.7983** | **+28,26%** en precisión promedio por clase. |
| **Recall Macro** | 0.5199 | **0.7944** | **+27,45%** en sensibilidad promedio. |
| **F1-Score Macro** | 0.4979 | **0.7946** | **+29,67%** en equilibrio global Precision/Recall. |

### Conclusión de la Comparación
El modelo **MLP** requería conectar los $12.288$ píxeles de entrada a 512 neuronas densas, generando más de 6,29 millones de pesos solo en su primera capa, lo que produjo sobreajuste (*overfitting*) y destruyó la noción de cercanía entre píxeles. La **CNN**, mediante **pesos compartidos** (filtros de $3 \times 3$) y reducción progresiva por **MaxPooling**, redujo los parámetros a 1,14 millones mientras elevó el Accuracy del **51,77% al 79,27%**.

---

## 7. Interpretación de Feature Maps y Análisis de Errores

### Evolución de los Mapas de Características (Feature Maps)
1. **Capa `conv_1` (64×64, 32 filtros):** Actúa como detector de bordes y contrastes cromáticos primarios (separa el cielo del suelo, siluetas de montañas y bordes rectos de edificios).
2. **Capa `conv_2` (32×32, 64 filtros):** Agrupa los bordes en texturas locales diferenciadas (textura densa de hojas en `forest`, superficies lisas en `sea`, tramas de ventanas en `buildings`).
3. **Capa `conv_3` (16×16, 128 filtros):** Presenta menor resolución espacial pero alta abstracción semántica, activando zonas específicas que corresponden a estructuras completas del paisaje antes de pasar al clasificador denso.

### Análisis de la Matriz de Confusión y Causas de Error
Al inspeccionar la Matriz de Confusión sobre las 3.000 imágenes de prueba, se identifican dos focos principales de error explicables visualmente:
1. **Confusión entre `glacier` y `mountain` / `sea`:** De 553 imágenes reales de `glacier`, 82 fueron predichas como `mountain` y 72 como `sea` (y 73 montañas fueron clasificadas como glaciar). **Causa técnica:** Ambas clases comparten picos rocosos con nieve y tonalidades azules/blancas; además, los bloques de hielo flotando en agua en `glacier` comparten texturas con `sea`.
2. **Confusión entre `buildings` y `street`:** 65 imágenes de `buildings` fueron clasificadas como `street`, y 42 de `street` como `buildings`. **Causa técnica:** Las fotografías de calles (`street`) contienen fachadas de edificios a ambos lados, por lo que si la calzada asfaltada no es predominante en la imagen de $64 \times 64$, los filtros convolucionales priorizan las ventanas y muros.

---

## 8. Conclusiones y Trabajo Futuro
* La arquitectura CNN demostró ser ampliamente superior al Perceptrón Multicapa para datos no estructurados de imágenes, logrando casi un **80% de exactitud** con una fracción de los parámetros.
* **Posibles mejoras futuras (Regularización y Optimización):**
  1. Incorporar **Data Augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) para mejorar la generalización entre clases visualmente solapadas (`glacier` vs. `mountain`).
  2. Añadir capas de **BatchNormalization** después de cada convolución y un callback de **EarlyStopping** para detener el entrenamiento en el punto óptimo de validación.
  3. Evaluar **Transfer Learning** (ej. MobileNetV2 o ResNet50) manteniendo la resolución nativa de $150 \times 150$ píxeles.

---

## 9. Estructura del Proyecto
```text
├── data/                   # Conjunto de datos Intel Image Classification (seg_train y seg_test)
├── images/                 # Evidencias gráficas exportadas (Curvas, Matriz de Confusión, Feature Maps)
├── models/                 # Modelos entrenados (.keras)
│   ├── mlp_intel_model.keras
│   └── cnn_intel_model.keras
├── notebooks/              # Notebooks documentados y ejecutables
│   ├── EP1_Intel_MLP.ipynb
│   └── EP2_Intel_CNN.ipynb
└── README.md               # Informe técnico reproducible del proyecto
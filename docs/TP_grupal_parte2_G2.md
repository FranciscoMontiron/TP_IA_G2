# Trabajo Práctico Cuatrimestral — Entrega Final
## Redes Neuronales Artificiales — Inteligencia Artificial — UTN FRLP 2026

**GRUPO N° 02**

| Apellido y Nombres | Legajo | E-Mail |
|---|---|---|
| Montiron Francisco | 31091 | franciscomontiron@gmail.com |
| Bogado Demian | 31173 | — |
| Cendoya Ramiro | 30602 | — |
| Martínez Lucas | 30697 | — |
| Diaz Valentina | 30628 | — |
| Bravo Federico | 30833 | — |

**Fecha de Presentación:** Mayo 2026

---

## Resumen

El presente trabajo describe el desarrollo e implementación de un sistema de clasificación automática de lesiones cutáneas mediante una Red Neuronal Artificial (RNA) de tipo Perceptrón Multicapa (MLP) entrenada con el algoritmo Backpropagation. El objetivo es clasificar imágenes dermatoscópicas del dataset HAM10000 en 7 categorías diagnósticas, incluyendo melanoma y otras lesiones malignas y benignas. Se implementó el modelo con la librería scikit-learn en Python, utilizando imágenes redimensionadas a 28×28 píxeles en escala de grises (784 features por muestra). La experimentación iterativa a través de 5 versiones del modelo permitió elevar la accuracy de validación del 41,3% (versión inicial) al 64,54% (versión final), mediante la optimización de la arquitectura, la técnica de balanceo de clases y los hiperparámetros de entrenamiento. El informe documenta el proceso de mejora, los resultados obtenidos y las conclusiones derivadas de la experiencia.

**Palabras clave:** redes neuronales artificiales, Backpropagation, clasificación de imágenes, cáncer de piel, Perceptrón Multicapa, HAM10000, desbalance de clases.

---

## 1. Introducción

### 1.1. Contexto y Relevancia

El cáncer de piel es el tipo de cáncer más frecuente a nivel mundial. Según la Organización Mundial de la Salud (OMS), cada año se diagnostican aproximadamente 1,5 millones de nuevos casos de cáncer de piel no melanoma y más de 325.000 casos de melanoma. La detección temprana es el factor más determinante para la supervivencia del paciente: un melanoma diagnosticado en estadio I tiene una tasa de supervivencia a 5 años superior al 98%, que cae a menos del 25% en estadios avanzados.

El diagnóstico tradicional requiere la evaluación visual por parte de un dermatólogo especializado y, en muchos casos, una biopsia confirmatoria. Sin embargo, la escasez de especialistas en regiones con recursos limitados, sumada a la variabilidad entre observadores, constituye una brecha significativa en la atención médica. Los sistemas de clasificación automática basados en aprendizaje automático representan una herramienta complementaria de alto valor para asistir al profesional médico en la detección temprana.

### 1.2. Problema a Resolver

El problema consiste en clasificar automáticamente imágenes dermatoscópicas de lesiones cutáneas en 7 categorías diagnósticas, utilizando los píxeles de la imagen como entrada a un MLP con algoritmo Backpropagation. Las categorías se detallan en la Tabla 1.

**Tabla 1.** Categorías diagnósticas del dataset HAM10000.

| Etiqueta | Código | Nombre | Tipo |
|---|---|---|---|
| 0 | akiec | Queratosis actínica e intraepitelial | Potencialmente maligna |
| 1 | bcc | Carcinoma basocelular | Maligna |
| 2 | bkl | Lesiones tipo queratosis benigna | Benigna |
| 3 | df | Dermatofibroma | Benigna |
| 4 | nv | Nevos melanocíticos (lunares) | Benigna |
| 5 | vasc | Lesiones vasculares | Benigna |
| 6 | mel | Melanoma | Maligna |

### 1.3. Justificación del Uso de Redes Neuronales

Un software tradicional basado en reglas determinísticas no puede resolver este problema eficazmente porque:

1. No existen reglas explícitas que definan con precisión los límites visuales entre categorías; las diferencias son sutiles y se expresan en combinaciones complejas de color, textura y forma.
2. La variabilidad intraclase es alta: dos melanomas pueden verse visualmente muy distintos.
3. Se requiere aprendizaje estadístico a partir de miles de ejemplos, no de reglas codificadas manualmente.
4. El problema pertenece al dominio del reconocimiento de patrones visuales, donde las RNA superan ampliamente a los algoritmos clásicos.

---

## 2. Elementos del Trabajo y Metodología

### 2.1. Modelo de RNA Seleccionado

Se seleccionó un **Perceptrón Multicapa (MLP)** entrenado mediante el algoritmo de **Backpropagation** (retropropagación del error).

**Justificación:**
- **Clasificación multiclase no lineal:** Las 7 clases de lesiones no son linealmente separables en el espacio de píxeles. El MLP con capas ocultas aprende fronteras de decisión no lineales mediante funciones de activación ReLU.
- **Capacidad de aproximación universal:** El teorema de aproximación universal (Hornik, 1991) garantiza que un MLP con al menos una capa oculta suficientemente grande puede aproximar cualquier función continua.
- **Adecuación al tipo de entrada:** Las imágenes están preprocesadas como vectores de longitud fija (784 píxeles), formato ideal para la capa de entrada de un MLP.
- **Aprendizaje supervisado:** El dataset cuenta con etiquetas diagnósticas confirmadas histopatológicamente, habilitando el entrenamiento supervisado.
- **Red de Hopfield descartada:** Adecuada para recuperación de patrones (memoria asociativa), pero no para clasificación multiclase con miles de ejemplos distintos.

### 2.2. Arquitectura y Topología Final

La arquitectura definitiva (versión 5) consta de 5 capas:

```
Entrada (784)
     ↓
Capa Oculta 1 (512 neuronas, ReLU)
     ↓
Capa Oculta 2 (256 neuronas, ReLU)
     ↓
Capa Oculta 3 (128 neuronas, ReLU)
     ↓
Salida (7 neuronas, Softmax)
```

**Tabla 2.** Arquitectura final de la RNA.

| Capa | Neuronas | Función de Activación | Parámetros |
|---|---|---|---|
| Entrada | 784 | — | — |
| Oculta 1 | 512 | ReLU | 784×512 + 512 = 401.920 |
| Oculta 2 | 256 | ReLU | 512×256 + 256 = 131.328 |
| Oculta 3 | 128 | ReLU | 256×128 + 128 = 32.896 |
| Salida | 7 | Softmax | 128×7 + 7 = 903 |
| **Total** | | | **~567.047 parámetros** |

**Hiperparámetros finales:**

| Parámetro | Valor | Justificación |
|---|---|---|
| Función de pérdida | Entropía cruzada categórica | Estándar para clasificación multiclase |
| Optimizador | Adam | Variante adaptativa, convergencia eficiente |
| Tasa de aprendizaje | 0,001 | Balance velocidad/estabilidad |
| Regularización L2 (alpha) | 0,001 | Reduce sobreajuste sobre muestras duplicadas |
| Batch size | 256 | Gradientes más estables que batch=64 |
| Max épocas | 500 | Con early stopping |
| Early stopping (paciencia) | 25 épocas | Detiene si el score de validación no mejora |

### 2.3. Patrones Utilizados

#### 2.3.1. Fuente del Dataset

Se utilizó el dataset **HAM10000** (*Human Against Machine with 10000 training images*), disponible en Kaggle y el ISIC Archive. Referencia: Tschandl P., Rosendahl C., Kittler H. (2018). *The HAM10000 dataset.* Scientific Data, 5, 180161.

Se utilizó específicamente el archivo `hmnist_28_28_L.csv`, que representa cada imagen como un vector de 784 píxeles (28×28 píxeles en escala de grises). Esta resolución fue seleccionada sobre la alternativa de 8×8 (64 píxeles) porque preserva más información estructural de las lesiones, lo cual resultó en una mejora de +22,6 puntos porcentuales de accuracy en la experimentación (ver sección 2.5).

#### 2.3.2. Distribución de Clases

**Tabla 3.** Distribución de muestras por clase diagnóstica.

| Etiqueta | Clase | Muestras | Porcentaje |
|---|---|---|---|
| 0 | akiec — Queratosis actínica | 327 | 3,3% |
| 1 | bcc — Carcinoma basocelular | 514 | 5,1% |
| 2 | bkl — Queratosis benigna | 1.099 | 11,0% |
| 3 | df — Dermatofibroma | 115 | 1,1% |
| 4 | nv — Nevo melanocítico | 6.705 | 66,9% |
| 5 | vasc — Lesión vascular | 142 | 1,4% |
| 6 | mel — Melanoma | 1.113 | 11,1% |
| **Total** | | **10.015** | **100%** |

El dataset presenta un marcado desbalance de clases: la clase nv representa el 66,9% del total, mientras que df y vasc representan apenas el 1,1% y 1,4% respectivamente. Este desbalance requirió el uso de técnicas específicas de balanceo (detalladas en la sección 2.5).

#### 2.3.3. División Entrenamiento / Validación

La división se realizó de forma **estratificada** para mantener la proporción de clases en ambos conjuntos:

| Conjunto | Muestras originales | Muestras tras oversampling |
|---|---|---|
| Entrenamiento (75%) | 7.511 | 35.203 |
| Validación (25%) | 2.504 | 2.504 (sin modificar) |

**Nota:** El oversampling se aplicó exclusivamente al conjunto de entrenamiento. El conjunto de validación se mantiene con la distribución original para evaluar el rendimiento real del modelo.

#### 2.3.4. Ejemplos de Patrones de Entrada

Cada patrón de entrada consiste en un vector de 784 valores correspondientes a los píxeles de la imagen dermatoscópica redimensionada a 28×28 en escala de grises, normalizados al rango [0, 1] mediante división por 255.

**Tabla 4.** Ejemplos de patrones de entrada por clase (primeros y últimos 3 valores).

| Clase | Primeros valores | Últimos valores | Etiqueta |
|---|---|---|---|
| akiec | [0.059, 0.188, 0.216, ...] | [..., 0.157, 0.098, 0.059] | 0 |
| bcc | [0.565, 0.925, 0.890, ...] | [..., 0.784, 0.690, 0.694] | 1 |
| bkl | [0.431, 0.471, 0.494, ...] | [..., 0.314, 0.282, 0.267] | 2 |
| df | [0.298, 0.318, 0.333, ...] | [..., 0.255, 0.243, 0.224] | 3 |
| nv | [0.537, 0.510, 0.369, ...] | [..., 0.459, 0.471, 0.459] | 4 |
| vasc | [0.843, 0.761, 0.627, ...] | [..., 0.808, 0.749, 0.808] | 5 |
| mel | [0.592, 0.647, 0.706, ...] | [..., 0.188, 0.141, 0.098] | 6 |

### 2.4. Herramientas y Librerías

**Tabla 5.** Herramientas utilizadas.

| Herramienta / Librería | Versión | Uso |
|---|---|---|
| Python | 3.x | Lenguaje de desarrollo |
| scikit-learn | ≥ 1.3 | MLPClassifier, train_test_split, métricas |
| imbalanced-learn | ≥ 0.11 | RandomOverSampler para balanceo de clases |
| pandas | ≥ 2.0 | Carga y manipulación del dataset CSV |
| numpy | ≥ 1.24 | Operaciones matriciales, normalización |
| matplotlib | ≥ 3.7 | Curvas de aprendizaje, matriz de confusión |
| seaborn | ≥ 0.12 | Visualización avanzada |
| Jupyter Notebook | — | Entorno de desarrollo y documentación |

Se utilizó `MLPClassifier` de scikit-learn por su implementación robusta del MLP con Backpropagation y su facilidad de configuración para experimentación académica.

### 2.5. Prototipos y Versiones del Modelo

Se desarrollaron 5 versiones del modelo de forma iterativa, cada una incorporando mejoras identificadas a partir del análisis de resultados de la versión anterior.

**Tabla 6.** Comparativa de versiones del modelo.

| Versión | Dataset | Balanceo | Arquitectura | batch | alpha | Épocas | Accuracy |
|---|---|---|---|---|---|---|---|
| v1 | 8×8 (64 feat.) | sample_weight | 128→64 | 64 | 0,0001 | 59 | 41,29% |
| v2 | 28×28 (784 feat.) | RandomOverSampler | 512→256→128 | 64 | 0,0001 | 224 | 63,86% |
| v3 | 28×28 (784 feat.) | SMOTE | 256→128 | 256 | 0,001 | ~300 | 58,87% |
| v4 | 28×28 (784 feat.) | sample_weight | 512→256→128 | 256 | 0,001 | 43 | 45,05% |
| **v5** | **28×28 (784 feat.)** | **RandomOverSampler** | **512→256→128** | **256** | **0,001** | **247** | **64,54%** |

**Decisiones tomadas en cada versión:**

- **v1 → v2 (+22,6%):** La mejora más significativa provino del cambio de resolución de imagen de 8×8 (64 features) a 28×28 (784 features). A 8×8 píxeles las texturas y bordes de las lesiones son prácticamente indistinguibles. Simultáneamente, se amplió la arquitectura para aprovechar la mayor cantidad de features, y se reemplazó `sample_weight` por `RandomOverSampler` para un balanceo más efectivo.

- **v2 → v3 (-4,99%):** Se probó SMOTE (Synthetic Minority Over-sampling Technique) en lugar de RandomOverSampler. SMOTE genera muestras sintéticas interpolando entre vecinos en el espacio de features. Sin embargo, en espacios de alta dimensión (784D) este método sufre la **maldición de la dimensionalidad**: los vecinos más cercanos no son necesariamente similares visualmente, generando muestras sintéticas poco representativas. Adicionalmente, se redujo la arquitectura de 3 a 2 capas ocultas, restando capacidad al modelo.

- **v3 → v4 (-13,82%):** Se exploró el uso de `compute_sample_weight` (pesos en la función de pérdida) junto con `StandardScaler`. El StandardScaler normalizó cada feature independientemente por su media y desvío. Para datos de píxeles, muchas features (píxeles de fondo) tienen varianza cercana a cero; al dividir por un desvío muy pequeño se obtuvieron valores extremos (p.ej. -808), generando gradientes inestables y un early stopping prematuro en la época 43. Los pesos extremos de clases minoritarias (df=12,47×, vasc=10,03×) sin oversampling profundizaron la inestabilidad.

- **v4 → v5 (+19,49%):** Se revirtió al esquema de v2 (RandomOverSampler + normalización /255) y se incorporaron las mejoras que resultaron válidas: `batch_size=256` (gradientes más estables) y `alpha=0,001` (mayor regularización L2 frente al sobreajuste sobre muestras duplicadas). Estas dos mejoras sobre la base de v2 produjeron la mejor versión final.

---

## 3. Resultados

### 3.1. Error General en Validación

| Métrica | Valor |
|---|---|
| Accuracy de validación | 64,54% |
| **Error general** | **35,46%** |
| Loss final (entropía cruzada) | 0,0283 |
| Épocas ejecutadas | 247 |

### 3.2. Resultados por Clase (Conjunto de Validación — 2.504 patrones)

**Tabla 7.** Métricas por clase en el conjunto de validación.

| Clase | Precision | Recall | F1-score | Soporte |
|---|---|---|---|---|
| akiec | — | — | — | 82 |
| bcc | — | — | — | 129 |
| bkl | — | — | — | 275 |
| df | — | — | — | 29 |
| nv | — | — | — | 1.676 |
| vasc | — | — | — | 35 |
| mel | — | — | — | 278 |
| **Accuracy global** | | | | **64,54%** |
| **Correctos totales** | | | **1.616 / 2.504** | |

*(Los valores de precision/recall/F1 por clase se encuentran en el notebook adjunto y en el archivo resultados_validacion.csv)*

### 3.3. Tabla de Patrones de Validación

A continuación se presentan 5 patrones de validación por clase (35 en total), mostrando los valores de entrada resumidos (primer y último píxel normalizado), la salida de la RNA y la salida esperada. El archivo `resultados_validacion.csv` adjunto contiene la tabla completa con los 2.504 patrones.

**Tabla 8.** Muestra representativa de patrones de validación (5 por clase).

| N° Patrón | Entrada (p0, p1, p2, ..., p783) | Salida RNA | Salida Esperada | Correcto |
|---|---|---|---|---|
| **Clase akiec (Queratosis actínica)** | | | | |
| 90 | [0.059, 0.188, 0.216, ..., valor_final] | bkl | akiec | NO |
| 95 | [0.949, 0.969, 0.753, ..., valor_final] | bcc | akiec | NO |
| 107 | [0.620, 0.647, 0.420, ..., valor_final] | bcc | akiec | NO |
| 122 | [0.537, 0.624, 0.576, ..., valor_final] | bcc | akiec | NO |
| 141 | [0.275, 0.369, 0.502, ..., valor_final] | nv | akiec | NO |
| **Clase bcc (Carcinoma basocelular)** | | | | |
| 32 | [0.110, 0.341, 0.208, ..., valor_final] | vasc | bcc | NO |
| 43 | [0.286, 0.255, 0.267, ..., valor_final] | bkl | bcc | NO |
| 49 | [0.690, 0.663, 0.859, ..., valor_final] | akiec | bcc | NO |
| 51 | [0.365, 0.318, 0.522, ..., valor_final] | bcc | bcc | **SI** |
| 81 | [0.137, 0.357, 0.682, ..., valor_final] | akiec | bcc | NO |
| **Clase bkl (Queratosis benigna)** | | | | |
| 10 | [0.286, 0.416, 0.576, ..., valor_final] | nv | bkl | NO |
| 12 | [1.000, 1.000, 1.000, ..., valor_final] | nv | bkl | NO |
| 17 | [1.000, 1.000, 1.000, ..., valor_final] | nv | bkl | NO |
| 21 | [0.357, 0.439, 0.502, ..., valor_final] | bcc | bkl | NO |
| 23 | [0.255, 0.318, 0.408, ..., valor_final] | mel | bkl | NO |
| **Clase df (Dermatofibroma)** | | | | |
| 35 | [0.110, 0.149, 0.173, ..., valor_final] | mel | df | NO |
| 69 | [0.027, 0.063, 0.024, ..., valor_final] | nv | df | NO |
| 116 | [0.373, 0.384, 0.298, ..., valor_final] | nv | df | NO |
| 126 | [1.000, 1.000, 1.000, ..., valor_final] | df | df | **SI** |
| 292 | [1.000, 1.000, 1.000, ..., valor_final] | df | df | **SI** |
| **Clase nv (Nevo melanocítico)** | | | | |
| 0 | [0.537, 0.510, 0.369, ..., valor_final] | nv | nv | **SI** |
| 1 | [0.745, 0.718, 0.627, ..., valor_final] | nv | nv | **SI** |
| 2 | [0.102, 0.027, 0.961, ..., valor_final] | nv | nv | **SI** |
| 3 | [0.522, 0.455, 0.137, ..., valor_final] | nv | nv | **SI** |
| 4 | [0.867, 0.945, 0.859, ..., valor_final] | bkl | nv | NO |
| **Clase vasc (Lesión vascular)** | | | | |
| 62 | [0.745, 0.718, 0.729, ..., valor_final] | nv | vasc | NO |
| 279 | [0.827, 0.761, 0.600, ..., valor_final] | nv | vasc | NO |
| 345 | [1.000, 1.000, 1.000, ..., valor_final] | nv | vasc | NO |
| 467 | [0.890, 0.878, 0.780, ..., valor_final] | nv | vasc | NO |
| 570 | [0.682, 0.600, 0.627, ..., valor_final] | nv | vasc | NO |
| **Clase mel (Melanoma)** | | | | |
| 33 | [0.969, 0.953, 0.984, ..., valor_final] | mel | mel | **SI** |
| 34 | [0.329, 0.349, 0.318, ..., valor_final] | nv | mel | NO |
| 38 | [0.102, 0.235, 0.420, ..., valor_final] | nv | mel | NO |
| 50 | [0.565, 0.502, 0.518, ..., valor_final] | nv | mel | NO |
| 67 | [0.937, 0.882, 0.984, ..., valor_final] | mel | mel | **SI** |

**Correctos en muestra (35 patrones):** 13 / 35 (37,1%)
**Correctos totales (2.504 patrones):** 1.616 / 2.504 (64,54%)

*(Nota: la muestra de 35 patrones tiene menor accuracy porque incluye exactamente 5 por clase, penalizando las clases difíciles. El total de 2.504 refleja el desempeño real, dominado por la clase nv que el modelo clasifica bien.)*

---

## 4. Discusión

### 4.1. ¿La RNA resuelve satisfactoriamente el problema?

El modelo final alcanzó una accuracy de validación del **64,54%** (error: 35,46%), lo cual representa una mejora sustancial respecto a la versión inicial (41,3%), pero no puede considerarse completamente satisfactorio para un contexto médico real donde se requiere una muy alta sensibilidad en la detección de lesiones malignas.

**Factores limitantes identificados:**

1. **Desbalance severo de clases (nv = 66,9%):** Aunque se aplicó RandomOverSampler, la clase dominante sigue influyendo desproporcionadamente en la accuracy global. Un clasificador trivial que prediga siempre "nv" obtendría un 66,9% de accuracy, evidenciando que la métrica global puede ser engañosa. Las métricas más relevantes para evaluar el rendimiento real son el **F1-score macro** (que promedia por clase sin considerar el tamaño de cada una) y el **recall de las clases malignas** (mel, bcc, akiec).

2. **Limitación de MLP sobre datos de imagen:** El MLP trata cada píxel como una feature independiente, perdiendo la información espacial que existe entre píxeles vecinos (texturas, bordes, formas). Las arquitecturas convolutivas (CNN) están específicamente diseñadas para explotar esta estructura espacial.

3. **Baja resolución de entrada (28×28 píxeles):** Las lesiones cutáneas se caracterizan por variaciones finas de color, textura y bordes que se pierden significativamente al redimensionar a 28×28 píxeles en escala de grises.

**Posibles cursos de acción para mejorar los resultados:**

- Utilizar redes neuronales convolucionales (CNN) sobre las imágenes originales a resolución completa, que han demostrado accuracy superiores al 85% en este mismo dataset.
- Incorporar características de color (imágenes RGB en lugar de escala de grises), dado que el color es un factor diagnóstico relevante en dermatoscopia.
- Aplicar técnicas de data augmentation (rotaciones, flips, zoom) para aumentar la variabilidad del dataset de entrenamiento sin duplicar muestras exactas.

### 4.2. Comparación entre arquitecturas

La experimentación iterativa permitió comparar distintas combinaciones de técnicas:

**Impacto de la resolución de imagen (v1 vs. v2, +22,6%):** El cambio de 8×8 a 28×28 píxeles fue la mejora más significativa de todo el proceso. A 8×8 píxeles, la información visual es insuficiente para distinguir patrones de lesiones. A 28×28, el modelo dispone de 12 veces más información por muestra.

**Impacto del balanceo de clases:** RandomOverSampler resultó superior a las alternativas evaluadas para este dataset:
- *RandomOverSampler* (v2, v5): mejor rendimiento. Aunque duplica muestras, en un espacio de 784 dimensiones la duplicación no induce el tipo de memorización que se observaría en espacios de baja dimensión.
- *SMOTE* (v3): inferior (-4,99%). La interpolación en 784 dimensiones sufre la maldición de la dimensionalidad, generando muestras sintéticas poco representativas.
- *sample_weight* sin oversampling (v1, v4): inferior. Con pocos datos de clases minoritarias, los pesos extremos (df=12,47×) generan gradientes inestables.

**Impacto del batch size y regularización (v2 vs. v5, +0,68%):** El incremento de batch_size de 64 a 256 estabilizó los gradientes (curva de aprendizaje más suave en las últimas épocas). El incremento de alpha de 0,0001 a 0,001 redujo el sobreajuste sobre las muestras duplicadas por oversampling.

---

## 5. Conclusión

### 5.1. Conclusiones de la Implementación

El trabajo permitió implementar exitosamente un sistema de clasificación de lesiones cutáneas basado en RNA, alcanzando una accuracy de validación del 64,54% mediante un proceso iterativo de 5 versiones. El recorrido de mejora (41,3% → 64,54%) evidencia la importancia crítica de:

- La **calidad de la representación de entrada**: la resolución de la imagen fue el factor de mayor impacto.
- La **estrategia de balanceo de clases**: el desbalance severo de este dataset (nv=66,9%) es uno de los principales obstáculos para el aprendizaje efectivo de las clases minoritarias.
- La **regularización adecuada**: alpha=0,001 y batch_size=256 resultaron combinaciones más robustas que los valores iniciales.

### 5.2. Relación con la Teoría Vista en Clase

- **Backpropagation:** El algoritmo de retropropagación del error ajustó iterativamente los pesos de la red mediante gradiente descendente (Adam), como se vio en la teoría de entrenamiento supervisado de RNA.
- **Funciones de activación:** ReLU en capas ocultas evita el problema del gradiente desvaneciente; Softmax en la salida produce una distribución de probabilidad sobre las 7 clases, interpretable como "confianza" del clasificador.
- **Regularización L2:** El parámetro alpha penaliza pesos grandes, reduciendo el sobreajuste (overfitting) sobre el conjunto de entrenamiento.
- **Early stopping:** Mecanismo de control de épocas basado en el rendimiento sobre un conjunto de validación interno, deteniendo el entrenamiento cuando el modelo deja de mejorar (n_iter_no_change=25 épocas).
- **Desbalance de clases:** La necesidad de técnicas de oversampling ilustra que el algoritmo Backpropagation optimiza la pérdida promedio; si una clase domina el dataset, la función de pérdida se minimiza principalmente aprendiendo esa clase.

### 5.3. Problemas Encontrados y Estrategias de Resolución

| Problema | Estrategia aplicada |
|---|---|
| Accuracy inicial baja (41,3%) con imágenes 8×8 | Cambio a resolución 28×28 (+22,6%) |
| Desbalance severo (nv=66,9%) | RandomOverSampler sobre el conjunto de entrenamiento |
| SMOTE no mejoró resultados en 784D | Identificado como maldición de la dimensionalidad; revertido a RandomOverSampler |
| StandardScaler generó valores extremos en píxeles de fondo | Revertido a normalización /255 estándar para datos de imagen |
| sample_weight con pesos extremos (12,47×) causó inestabilidad | Reemplazado por oversampling que equilibra las muestras directamente |
| Curva de aprendizaje errática en épocas finales (v2) | Incremento de batch_size (64→256) y alpha (0,0001→0,001) |
| Convergencia prematura (epoch 43 en v4) | Identificada inestabilidad de gradientes por StandardScaler; corregido en v5 |

---

## 6. Referencias

[1] Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

[2] Organización Mundial de la Salud (OMS). Cáncer de piel — datos y cifras. Consultado en 2026. https://www.who.int

[3] Haykin, S. (2009). *Neural Networks and Learning Machines* (3.ª ed.). Pearson.

[4] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[5] Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.

[6] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[7] Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research*, 18(17), 1–5.

[8] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

---

## Anexo — Archivos entregados

| Archivo | Descripción |
|---|---|
| `skin_cancer_RNA.ipynb` | Código fuente completo (Jupyter Notebook) |
| `requirements.txt` | Dependencias Python |
| `download_data.py` | Script para descargar el dataset desde Kaggle |
| `archive/hmnist_28_28_L.csv` | Dataset de entrenamiento (784 features, 28×28 px) |
| `resultados_validacion.csv` | Tabla completa de los 2.504 patrones de validación |
| `curva_aprendizaje.png` | Gráfico de la curva de aprendizaje (v5) |
| `distribucion_clases.png` | Distribución de clases del dataset |
| `matriz_confusion.png` | Matriz de confusión sobre validación |
| `ejemplos_patrones.png` | Visualización de ejemplos por clase |

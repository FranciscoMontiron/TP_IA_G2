# Entrega Parcial — Trabajo Práctico Cuatrimestral
## Redes Neuronales Artificiales — UTN FRLP — Inteligencia Artificial 2026

---

## 1. Descripción del Problema

### Contexto y Relevancia

El cáncer de piel es el tipo de cáncer más frecuente a nivel mundial. Según la Organización Mundial de la Salud (OMS), cada año se diagnostican aproximadamente 1,5 millones de nuevos casos de cáncer de piel no melanoma y más de 325.000 casos de melanoma. La detección temprana es el factor más determinante para la supervivencia del paciente: un melanoma diagnosticado en estadio I tiene una tasa de supervivencia a 5 años superior al 98%, que cae a menos del 25% en estadios avanzados.

El diagnóstico tradicional requiere la evaluación visual por parte de un dermatólogo especializado y, en muchos casos, una biopsia confirmatoria. Sin embargo, la escasez de especialistas en regiones con recursos limitados, sumada a la variabilidad entre observadores, constituye una brecha significativa en la atención médica.

### Problema a Resolver

El problema consiste en **clasificar automáticamente imágenes dermatoscópicas de lesiones cutáneas** en 7 categorías diagnósticas, utilizando los píxeles de la imagen como entrada a una Red Neuronal Artificial con algoritmo de aprendizaje Backpropagation.

Las 7 clases del problema son:

| Etiqueta | Código | Nombre                              | Tipo         |
|----------|--------|--------------------------------------|--------------|
| 0        | akiec  | Queratosis actínica e intraepitelial | Potenc. maligna |
| 1        | bcc    | Carcinoma basocelular                | Maligna      |
| 2        | bkl    | Lesiones tipo queratosis benigna     | Benigna      |
| 3        | df     | Dermatofibroma                       | Benigna      |
| 4        | nv     | Nevos melanocíticos (lunares)        | Benigna      |
| 5        | vasc   | Lesiones vasculares                  | Benigna      |
| 6        | mel    | Melanoma                             | Maligna      |

### Por Qué No Puede Resolverse con Software Tradicional

Un software tradicional (reglas `if-then`, programación estructurada) no puede resolver este problema porque:

- **No existen reglas explícitas** que definan con precisión los límites visuales entre categorías; las diferencias son sutiles y se expresan en combinaciones complejas de color, textura y forma.
- **La variabilidad intraclase es alta**: dos melanomas pueden verse visualmente muy distintos entre sí.
- **Requiere aprendizaje estadístico** a partir de miles de ejemplos, no de reglas codificadas manualmente.
- El problema pertenece al dominio del **reconocimiento de patrones visuales**, donde las RNA superan ampliamente a los algoritmos determinísticos clásicos.

---

## 2. Modelo de RNA a Implementar y Justificación

### Modelo Seleccionado: Perceptrón Multicapa (MLP) con Backpropagation

Se selecciona un **Perceptrón Multicapa (MLP)** entrenado mediante el algoritmo de **Backpropagation (retropropagación del error)** como modelo de RNA.

### Justificación

1. **Problema de clasificación multiclase no lineal**: Las 7 clases de lesiones no son linealmente separables en el espacio de píxeles. El MLP con capas ocultas tiene la capacidad de aprender fronteras de decisión no lineales complejas mediante sus funciones de activación no lineales (ReLU).

2. **Capacidad de aproximación universal**: El teorema de aproximación universal garantiza que un MLP con al menos una capa oculta suficientemente grande puede aproximar cualquier función continua, lo que lo hace apto para mapear patrones de píxeles a etiquetas de diagnóstico.

3. **Adecuación al tipo de entrada**: Las imágenes están preprocesadas como vectores de 64 píxeles (8×8 píxeles en escala de grises), lo que produce un vector de entrada de dimensión fija, formato ideal para la capa de entrada de un MLP.

4. **Aprendizaje supervisado**: El dataset cuenta con etiquetas diagnósticas confirmadas por histopatología, lo que permite el entrenamiento supervisado propio del algoritmo Backpropagation.

5. **Alternativa descartada (Hopfield)**: La red de Hopfield es adecuada para recuperación de patrones y memoria asociativa, pero no para clasificación multiclase con miles de ejemplos distintos, por lo que queda descartada para este problema.

### Arquitectura Propuesta

```
Capa de Entrada:   64 neuronas  (píxeles pixel0000 a pixel0063, normalizados a [0,1])
                        ↓
Capa Oculta 1:    128 neuronas  (activación: ReLU)
                        ↓
Capa Oculta 2:     64 neuronas  (activación: ReLU)
                        ↓
Capa de Salida:     7 neuronas  (activación: Softmax — una por clase diagnóstica)
```

- **Función de pérdida**: Entropía cruzada categórica (Cross-Entropy Loss)
- **Optimizador**: Adam (variante adaptativa de Gradient Descent)
- **Tasa de aprendizaje**: 0.001 (ajustable durante experimentación)
- **Regularización**: Dropout (a definir en la entrega final según resultados)

La arquitectura puede ajustarse durante la experimentación; la configuración definitiva se reportará en la entrega final.

---

## 3. Patrones Utilizados

### Fuente del Dataset

**Dataset**: HAM10000 — *Human Against Machine with 10000 training images*  
**Fuente**: Kaggle / ISIC Archive (International Skin Imaging Collaboration)  
**Referencia**: Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 2018.

### Descripción de los Datos

El dataset en su forma procesada para RNA (`hmnist_8_8_L.csv`) contiene:

| Característica | Valor |
|---|---|
| Total de muestras | 10.015 |
| Características de entrada | 64 (píxeles de imagen 8×8 en escala de grises) |
| Rango de valores de píxel | 0 – 255 (se normalizarán a [0, 1] dividiendo por 255) |
| Variable de salida | Etiqueta numérica 0–6 (clase diagnóstica) |
| Formato | CSV (`hmnist_8_8_L.csv`) |

### Distribución de Clases

| Etiqueta | Clase | Muestras | Porcentaje |
|----------|-------|----------|------------|
| 0 | akiec — Queratosis actínica | 327 | 3,3% |
| 1 | bcc — Carcinoma basocelular | 514 | 5,1% |
| 2 | bkl — Queratosis benigna | 1.099 | 11,0% |
| 3 | df — Dermatofibroma | 115 | 1,1% |
| 4 | nv — Nevo melanocítico | 6.705 | 66,9% |
| 5 | vasc — Lesión vascular | 142 | 1,4% |
| 6 | mel — Melanoma | 1.113 | 11,1% |
| **Total** | | **10.015** | **100%** |

### División Entrenamiento / Validación

Conforme a los requisitos de la cátedra (mínimo 25% para validación):

- **Entrenamiento**: 75% → 7.511 muestras
- **Validación**: 25% → 2.504 muestras

La división se realizará de forma estratificada para mantener la proporción de clases en ambos conjuntos.

### Ejemplos de Patrones de Entrada

Cada patrón consiste en un vector de 64 valores enteros (píxeles de la imagen dermatoscópica redimensionada a 8×8 en escala de grises):

**Ejemplo 1 — akiec (label=0):**
`[85, 177, 193, 182, 179, 192, 121, 40, 25, 115, 152, 155, 153, 165, 148, 69, ...]`

**Ejemplo 2 — bcc (label=1):**
`[144, 236, 227, 217, 220, 233, 226, 177, 157, 234, 226, 220, 222, 234, 237, 197, ...]`

**Ejemplo 3 — bkl (label=2):**
`[172, 182, 191, 183, 180, 181, 165, 164, 173, 192, 201, 172, 176, 188, 179, 167, ...]`

**Ejemplo 4 — nv (label=4):**
`[151, 157, 161, 163, 165, 165, 159, 152, 148, 155, 161, 164, 165, 164, 158, 149, ...]`

**Ejemplo 5 — mel (label=6):**
`[151, 165, 180, 184, 186, 185, 180, 163, 152, 171, 183, 188, 191, 189, 183, 164, ...]`

El vector completo de cada muestra en el CSV tiene la forma:
```
pixel0000, pixel0001, ..., pixel0063, label
172,       182,       ..., 168,       2
```

---

## 4. Herramienta, Lenguaje y Librería Seleccionada

### Lenguaje: Python 3.x

Python es el estándar de la industria para el desarrollo de sistemas de aprendizaje automático y RNA. Ofrece un ecosistema maduro de librerías, amplia documentación y facilidad de experimentación.

### Librerías Principales

| Librería | Versión | Uso en el proyecto |
|---|---|---|
| **scikit-learn** | ≥ 1.3 | Implementación del MLP (`MLPClassifier`), división train/test, métricas de evaluación |
| **pandas** | ≥ 2.0 | Carga y manipulación del dataset CSV |
| **numpy** | ≥ 1.24 | Operaciones matriciales, normalización de píxeles |
| **matplotlib** | ≥ 3.7 | Visualización de curvas de aprendizaje y matriz de confusión |
| **seaborn** | ≥ 0.12 | Visualización avanzada de resultados |

### Justificación

- **`MLPClassifier` de scikit-learn**: Implementación robusta y documentada del Perceptrón Multicapa con Backpropagation. Permite configurar número de capas, neuronas, función de activación, optimizador y tasa de aprendizaje, siendo ideal para experimentación académica.
- **Alternativa considerada (TensorFlow/Keras)**: Si bien Keras ofrece mayor flexibilidad para arquitecturas complejas, para este trabajo scikit-learn es suficiente y reduce la complejidad del entorno de ejecución.

### Entorno de Desarrollo

- **IDE**: Visual Studio Code / Jupyter Notebook
- **Formato de código**: Jupyter Notebook (`.ipynb`) para facilitar la documentación iterativa y visualización inline de resultados

---

*Entrega Parcial — Trabajo Práctico Cuatrimestral RNA — UTN FRLP 2026*

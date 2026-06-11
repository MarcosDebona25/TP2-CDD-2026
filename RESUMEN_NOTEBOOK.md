# Resumen del Notebook TP2 — Clasificación de Intenciones (Intent Detection)

## Objetivo general

Clasificar 77 tipos de intenciones a partir de mensajes de texto en inglés, usando el dataset `intent.csv`. Se comparan múltiples modelos con dos representaciones vectoriales distintas.

---

## Dataset

- **Archivo:** `./Documentos/intent.csv`
- **Columnas:** `text` (mensaje del usuario) y `category` (intención, 77 clases)
- **División:** 80% train / 20% test, estratificada (misma proporción de clases en ambos splits)
- **Desbalanceo:** moderado (hay clases con más muestras que otras)

---

## Etapa 1 — Análisis Exploratorio (EDA)

### Importaciones y configuración

Se importan las siguientes librerías principales:

| Librería | Para qué |
|---|---|
| `pandas`, `numpy` | Manejo de datos |
| `matplotlib`, `seaborn`, `WordCloud` | Visualización |
| `nltk` | Tokenización, POS tagging, lematización, stopwords |
| `sentence_transformers` | Embeddings semánticos (SBERT) |
| `sklearn` | Modelos ML, métricas, preprocesamiento, búsqueda de hiperparámetros |
| `xgboost` | Modelo de boosting |
| `torch` (PyTorch) | Red neuronal MLP |

### Funciones de preprocesamiento de texto

#### `get_wordnet_pos(treebank_tag)`
Convierte las etiquetas POS de NLTK (formato Treebank, ej: `VBZ`) al formato que entiende WordNet (`VERB`, `ADJ`, `ADV`, `NOUN`). Se usa internamente para que la lematización sea más precisa.

#### `clean_text(text)`
Limpia y normaliza un texto en 7 pasos:
1. Todo a minúsculas
2. Elimina URLs (`http://...`, `www...`)
3. Elimina caracteres especiales (deja solo alfanuméricos y espacios)
4. Tokeniza con `word_tokenize()`
5. Hace POS tagging para saber la categoría gramatical de cada token
6. Lematiza cada token usando su POS (ej: "running" → "run")
7. Filtra stopwords en inglés y tokens de menos de 2 caracteres

**Importante:** Esta función se aplica solo para generar la columna `text_clean`, que se usa con TF-IDF. Los embeddings SBERT se alimentan del **texto original** porque el modelo pre-entrenado necesita el contexto completo para funcionar bien.

### Ingeniería de características iniciales
Se crean dos columnas adicionales:
- `word_count`: cantidad de palabras (por `split()`)
- `char_count`: cantidad de caracteres

### Visualizaciones del EDA

| Visualización | Qué muestra |
|---|---|
| Histograma horizontal de clases | Distribución de muestras por categoría, ordenadas de mayor a menor, con línea de promedio |
| WordCloud | Palabras más frecuentes del corpus limpiado |
| Histogramas de `word_count` y `char_count` | Distribución de longitud de mensajes, con media y mediana |
| Boxplot de `word_count` por clase | Si ciertas intenciones tienden a usar mensajes más largos |
| Análisis de outliers | Mensajes muy cortos (≤ 3 palabras) y muy largos (> Q3 + 1.5·IQR) |
| Correlación `word_count` vs `char_count` | Pearson + Spearman + scatterplot con línea de regresión |
| Vocabulario exclusivo por clase | Palabras que aparecen **solo** en esa clase; clases con más vocabulario exclusivo son más fáciles de clasificar |

---

## Etapa 2 — Preprocesamiento y Representaciones Vectoriales

### División del dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

La división se hace **antes** de ajustar cualquier vectorizador para evitar data leakage.

### Representación 1: TF-IDF

```python
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf_train = tfidf.fit_transform(X_train['text_clean'])
X_tfidf_test  = tfidf.transform(X_test['text_clean'])
```

- **Entrada:** `text_clean` (lematizado, sin stopwords)
- **Output:** matriz dispersa, hasta 5000 palabras más frecuentes
- `fit()` solo sobre train, `transform()` sobre test

### Representación 2: Sentence Embeddings (SBERT)

```python
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_embs_train = model_sbert.encode(X_train['text'].tolist(), show_progress_bar=True)
X_embs_test  = model_sbert.encode(X_test['text'].tolist(), show_progress_bar=True)
```

- **Modelo:** `all-MiniLM-L6-v2` — modelo SBERT pre-entrenado, 384 dimensiones de output
- **Entrada:** `text` original (sin limpiar), para preservar el contexto semántico
- **Output:** matriz densa de 384 columnas por muestra
- No requiere `fit()` (es pre-entrenado)
- Captura sinonimia, paráfrasis y relaciones semánticas que TF-IDF no puede

### Codificación del target

```python
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded  = encoder.transform(y_test)
class_names = encoder.classes_
```

Convierte los nombres de categoría (strings) en enteros 0–76. `class_names` se guarda para reportes.

---

## Etapa 3 — Modelos

### Estrategia general de evaluación

- **Búsqueda de hiperparámetros:** `GridSearchCV` o `RandomizedSearchCV` usando **solo el conjunto de train** con `StratifiedKFold(n_splits=5)`
- **Métrica de optimización:** F1-macro (trata igual a todas las clases, ideal para desbalanceo)
- **Evaluación final:** una sola vez sobre el conjunto de TEST (sin refitting)
- **Balance de clases:** todos los modelos usan `class_weight='balanced'` o pesos por muestra

### Función de evaluación

```python
evaluar_modelo(nombre, y_true_enc, y_pred_enc, mostrar_cm=False, mostrar_reporte=True)
```

Calcula y reporta:
- **Accuracy**
- **F1-macro:** promedio sin ponderar (da igual peso a clases minoritarias)
- **F1-weighted:** promedio ponderado por frecuencia de clase
- Opcionalmente: matriz de confusión y classification report por clase

---

### Modelo 1: Regresión Logística + TF-IDF

**Representación:** TF-IDF (matriz dispersa)

**Configuración base:**
```python
LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
```

**Búsqueda con GridSearchCV:**
| Hiperparámetro | Valores buscados |
|---|---|
| `C` (regularización) | [0.01, 0.1, 1, 5, 10, 100] |
| `solver` | ['lbfgs', 'saga'] |
| `class_weight` | [None, 'balanced'] |

- Total: 24 combinaciones × 5 folds = **120 entrenamientos**

---

### Modelo 2: Regresión Logística + SBERT

**Representación:** Embeddings SBERT (384 dimensiones, denso)

**Configuración:** Idéntica al Modelo 1, solo cambia la entrada.

Se muestra una comparativa directa entre Modelo 1 y Modelo 2 para ver el impacto de la representación vectorial (TF-IDF vs embeddings semánticos).

---

### Modelo 3: XGBoost + SBERT

**Representación:** Embeddings SBERT

**Configuración base:**
```python
XGBClassifier(tree_method='hist', random_state=42)
```

**Búsqueda con RandomizedSearchCV:**
| Hiperparámetro | Valores buscados |
|---|---|
| `n_estimators` | [100, 200] |
| `max_depth` | [4, 6, 8] |
| `learning_rate` | [0.1, 0.2] |
| `subsample` | [0.8, 1.0] |

- Total: 5 iteraciones aleatorias × 3 folds = **15 entrenamientos**

**Balance de clases:** se calculan pesos de muestra con `compute_sample_weight` y se pasan al `fit()`.

---

### Modelo 4: SVM Lineal + TF-IDF

**Representación:** TF-IDF

**Configuración base:**
```python
LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
```

**Búsqueda con GridSearchCV:**
| Hiperparámetro | Valores buscados |
|---|---|
| `C` | [0.5, 1, 5] |

- Total: 3 combinaciones × 3 folds = **9 entrenamientos**

`LinearSVC` es más eficiente que `SVC` con kernel RBF en espacios dispersos de alta dimensión como TF-IDF.

---

### Modelo 5: MLP (Red Neuronal) con PyTorch + SBERT

**Representación:** Embeddings SBERT (input de 384 dimensiones)

**Arquitectura:**
```
Input (384) → Linear(384→256) → ReLU → Dropout(0.3)
           → Linear(256→128) → ReLU → Dropout(0.3)
           → Linear(128→77)
```

**Clase PyTorch:**
```python
class IntentMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        self.red = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
```

**Detalles de entrenamiento:**
- **Función de pérdida:** `CrossEntropyLoss` con pesos de clase balanceados
- **Optimizador:** Adam
- **Batch size:** 32
- **Validación interna:** 85% train / 15% validación (solo dentro del train set)
- **Early stopping:** basado en la métrica de validación interna
- **TEST:** reservado completamente para evaluación final

---

## Etapa 4 — Comparativa de Resultados

### Tabla de resultados

Se construye un DataFrame con los 5 modelos, ordenados por F1-macro descendente:

```python
df_resultados = pd.DataFrame(resultados).T
                 .sort_values('F1 macro', ascending=False)
                 .round(4)
```

Columnas: **Accuracy**, **F1-macro**, **F1-weighted**

### Visualizaciones de comparativa

1. **Barplot agrupado:** 3 métricas (azul/naranja/verde) para cada modelo, con valores etiquetados
2. **Heatmap:** modelos × métricas, escala de color RdYlGn (rojo=bajo, verde=alto)
3. **Lollipop chart:** ranking de modelos por F1-macro con gradiente de colores

---

## Resumen de representaciones y modelos

| Modelo | Representación | Método búsqueda HP | Folds CV |
|---|---|---|---|
| Regresión Logística | TF-IDF | GridSearchCV | 5 |
| Regresión Logística | SBERT | GridSearchCV | 5 |
| XGBoost | SBERT | RandomizedSearchCV | 3 |
| SVM Lineal | TF-IDF | GridSearchCV | 3 |
| MLP (PyTorch) | SBERT | Arquitectura fija + early stopping | — |

---

## Posibles gaps / partes incompletas

Basado en el análisis de la notebook, estas son las secciones que podrían estar incompletas o que no existen aún:

1. **Loop de entrenamiento del MLP:** el código de la celda 49 puede estar truncado; posiblemente faltan las épocas de entrenamiento y la evaluación sobre TEST
2. **Curvas de aprendizaje:** no hay gráficos de train vs val loss por época para el MLP
3. **Análisis de errores:** no hay análisis de qué clases se confunden entre sí (pares de clases con mayor error)
4. **Tiempos de entrenamiento/inferencia:** no se reportan
5. **Selección y guardado del mejor modelo:** no hay serialización del modelo ganador
6. **Siguiente pasos o conclusiones:** no hay recomendaciones de mejora documentadas

---

## Flujo general del código

```
Carga del dataset
      ↓
EDA (distribución, longitud, vocabulario, outliers)
      ↓
Preprocesamiento: clean_text() → text_clean
      ↓
División 80/20 estratificada (ANTES de vectorizar)
      ↓
         ┌──────────────────────────────────────────────┐
         │                                              │
    TF-IDF (text_clean)                     SBERT (text original)
    max_features=5000                       all-MiniLM-L6-v2 → 384 dims
         │                                              │
    ┌────┴────┐                         ┌───────────────┼────────────────┐
    │         │                         │               │                │
  LogReg   LinearSVC                 LogReg          XGBoost           MLP
  GridCV   GridCV                    GridCV        RandomCV          PyTorch
    │         │                         │               │                │
    └────┬────┘                         └───────────────┼────────────────┘
         │                                              │
         └──────────────────┬───────────────────────────┘
                            ↓
              evaluar_modelo() sobre TEST
                            ↓
              Tabla comparativa + visualizaciones
```

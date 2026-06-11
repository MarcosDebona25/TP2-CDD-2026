# ── 3.3  SVM lineal (LinearSVC) + TF-IDF enriquecido ──────────────────────────
# LinearSVC escala mucho mejor que SVC(kernel='rbf') en espacios dispersos de
# alta dimensión como TF-IDF.
#
# BÚSQUEDA DE HIPERPARÁMETROS CON RandomizedSearchCV:
# Metemos el TF-IDF dentro de un Pipeline para optimizar la VECTORIZACIÓN junto
# con el clasificador, no solo C. Eso agranda mucho el espacio de búsqueda
# (max_features × ngram_range × min_df × C), así que usamos búsqueda aleatoria
# en lugar de GridSearchCV:
#   1) Eficiencia: el grid completo serían 4×3×3×(continuo) combinaciones; con
#      n_iter=25 exploramos una muestra representativa en ~1/5 del tiempo.
#   2) C continuo: loguniform muestrea cualquier valor en escala log (la escala
#      natural de la regularización), en vez de 5 puntos fijos de una grilla.
#   3) Bergstra & Bengio (2012): si pocos hiperparámetros importan de verdad
#      (típicamente C y ngram_range), el muestreo aleatorio halla buenas zonas
#      con muchas menos evaluaciones que la búsqueda exhaustiva.
#
# El Pipeline vectoriza internamente con fit SOLO en train (cada fold de CV
# ajusta su propio TF-IDF), evitando data leakage de forma automática.

from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

pipe_svm = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),   # 1 + log(tf): suaviza repeticiones
    ('clf', LinearSVC(
        class_weight='balanced',   # compensa el desbalanceo de clases
        random_state=42,
        max_iter=10000,
        dual=False,                # n_muestras > n_features -> formulación primal
    )),
])

# Distribuciones/listas a muestrear (no se prueban todas: se sortean n_iter)
param_dist_svm = {
    'tfidf__max_features': [20000, 30000, 50000, None],  # None = vocabulario completo
    'tfidf__ngram_range':  [(1, 1), (1, 2), (1, 3)],     # uni / bi / trigramas
    'tfidf__min_df':       [2, 3, 5],                    # filtra términos raros (ruido)
    'clf__C':              loguniform(1e-2, 1e1),        # C continuo en escala log
}

rand_svm = RandomizedSearchCV(
    pipe_svm,
    param_distributions=param_dist_svm,
    n_iter=25,                 # 25 combinaciones sorteadas (vs. grilla exhaustiva)
    scoring='f1_macro',        # métrica robusta al desbalanceo
    cv=cv,                     # StratifiedKFold(3) reutilizado en todo el TP
    n_jobs=-1,
    verbose=1,
    random_state=42,           # reproducibilidad del sorteo
    refit=True,                # reentrena el mejor sobre todo el train
)
rand_svm.fit(X_train['text_clean'], y_train_encoded)   # texto crudo: el pipe vectoriza

print(f"Mejores hiperparámetros (SVM): {rand_svm.best_params_}")
print(f"Mejor F1-macro en CV         : {rand_svm.best_score_:.4f}\n")

# Evaluación final sobre el TEST (una sola vez)
best_svm = rand_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test['text_clean'])
evaluar_modelo("SVM lineal (TF-IDF)", y_test_encoded, y_pred_svm)

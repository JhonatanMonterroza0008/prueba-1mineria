# Iris species classification

Proyecto de clasificación de flores Iris para la asignatura de Minería de Datos.  
La idea es entrenar un modelo de Machine Learning que reciba cuatro medidas de la flor y devuelva la especie probable.

## Dataset

Se usa el dataset clásico **Iris** (150 registros y 3 especies: *Iris-setosa*, *Iris-versicolor* e *Iris-virginica*).  
Cada muestra tiene estas variables numéricas:

- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species` (etiqueta de salida)

El archivo se llama `Iris.csv` y debe estar en la misma carpeta del proyecto.

## Flujo de trabajo

El flujo que se implementa en `main.py` sigue la idea vista en clase:

1. **Entendimiento y exploración**
   - Carga del dataset desde `Iris.csv`.
   - Vista rápida de las primeras filas.
   - Conteo de registros por especie.
   - Cálculo de correlación entre las variables numéricas.
   - Histogramas por característica separados por especie.

2. **Preprocesamiento**
   - Selección de las 4 variables numéricas como entradas del modelo.
   - Eliminación de la columna `Id` si está presente.
   - Escalado de las variables de entrada con `StandardScaler`.

3. **División entrenamiento / prueba**
   - Separación en entrenamiento y prueba (80% / 20%).
   - Estratificación para mantener el balance entre especies.

4. **Modelo de clasificación**
   - Se usa un **Random Forest** como modelo principal.
   - El modelo se construye dentro de un `Pipeline` junto con el escalador.

5. **Búsqueda de hiperparámetros y validación**
   - Se define una rejilla pequeña de hiperparámetros (`n_estimators`, `max_depth`, `min_samples_split`).
   - Se aplica `GridSearchCV` con validación cruzada de 5 folds sobre el conjunto de entrenamiento.
   - Se guarda el mejor modelo encontrado.

6. **Evaluación**
   - Predicciones sobre el conjunto de prueba.
   - Cálculo de las métricas: **Accuracy**, **Precision**, **Recall** y **F1-score** con promedio ponderado.
   - Cálculo y visualización de la matriz de confusión.

7. **Dashboard interactivo (Streamlit)**
   - Pestaña de datos: tabla, distribución de especies, mapa de correlación e histogramas.
   - Pestaña de modelo: descripción breve del modelo, métricas y matriz de confusión.
   - Pestaña de predicción:
     - Panel para ingresar manualmente las 4 medidas de la flor.
     - Predicción de la especie y gráfico de barras con las probabilidades.
     - Gráfico 3D con el dataset original y el nuevo punto marcado para ver su posición relativa.

## Cómo ejecutar el proyecto

1. Clonar o descargar este repositorio.
2. Crear (opcional) un entorno virtual de Python.
3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Asegurarse de que el archivo `Iris.csv` esté en la misma carpeta que `main.py`.
5. Ejecutar la aplicación de Streamlit:

   ```bash
   streamlit run main.py
   ```

6. Abrir el enlace que muestra la consola (normalmente `http://localhost:8501`) y explorar el dashboard.





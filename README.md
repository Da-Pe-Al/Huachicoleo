# Pipeline robusto para detección de robo de hidrocarburos

Este repositorio contiene un flujo completo para entrenar y validar un autoencoder LSTM que detecta eventos anómalos en operaciones de rebombeo. El objetivo es ofrecer una guía reproducible que cubra desde la preparación de datos hasta la calibración del umbral de alerta y la interpretación de métricas clave.

## Requerimientos

- Python 3.10+
- Dependencias principales:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Instala el entorno recomendado ejecutando (o usa `requirements.txt` para crear un entorno reproducible):

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Archivos principales

- **`rebombeo_huachicoleo.csv`**: dataset crudo con mediciones de operación y etiqueta binaria (`label`) que identifica anomalías confirmadas.
- **`Red_Huachicoleo.ipynb`**: cuaderno con el pipeline robustecido para entrenamiento, calibración y evaluación del autoencoder LSTM.

## Pasos para ejecutar el análisis

1. **Preparar el entorno**
   - Clona el repositorio y activa el entorno virtual con las dependencias listadas.
   - Verifica que el archivo `rebombeo_huachicoleo.csv` esté en la raíz del proyecto.

2. **Ejecutar el pipeline**
   - **Opción cuaderno:** ejecuta `jupyter notebook` o `jupyter lab`, abre `Red_Huachicoleo.ipynb` y corre las celdas en orden.
codex/review-hydrocarbon-theft-analysis-code-x3whc0
   - **Opción script reproducible:** ejecuta `python run_pipeline.py --output artefactos/resultados.json --report artefactos/resumen.md --advanced-models` para entrenar el modelo, evaluar detectores complementarios y generar un reporte JSON + Markdown con las métricas, el umbral calibrado y los errores de reconstrucción del conjunto de prueba.


3. **Comprender el pipeline**
   - **Carga y ordenamiento de datos**: el dataset se ordena cronológicamente para preservar dependencias temporales.
   - **Generación de ventanas**: se crean secuencias de 30 minutos con un paso de 5 minutos. Cada ventana hereda la etiqueta positiva si contiene al menos una muestra anómala.
   - **División temporal**: se separan los datos en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%) sin mezclar periodos.
   - **Escalamiento sin fuga de información**: el `StandardScaler` se ajusta exclusivamente con observaciones normales del segmento de entrenamiento y se aplica al resto de los datos.
codex/review-hydrocarbon-theft-analysis-code-x3whc0
   - **Entrenamiento del autoencoder**: el modelo LSTM configurable se entrena solo con ventanas normales y utiliza `EarlyStopping` para evitar sobreajuste.
   - **Calibración del umbral**: se calculan los errores de reconstrucción en validación, se experimenta con percentiles, MAD e IQR y se selecciona automáticamente el percentil que maximiza el `F1-score`, equilibrando precisión y recall.
   - **Evaluación final**: en el conjunto de prueba se reportan métricas completas (`precision`, `recall`, `f1`, `ROC AUC`, `Average Precision`) y se generan visualizaciones de errores y matrices de confusión.

## Cómo implementar las acciones recomendadas

Las acciones sugeridas en el resumen pueden llevarse a cabo de la siguiente forma. Los pasos están pensados para ejecutarse tanto desde el cuaderno como desde `run_pipeline.py`.

### 1. Pipeline reproducible con separación temporal y escalado seguro

1. **Ordenar y dividir cronológicamente**: mantén el orden temporal al cargar el CSV. En el script la función `temporal_split` utiliza proporciones (`--train-ratio`, `--val-ratio`) para crear particiones consecutivas, lo que evita fuga de información entre periodos.
2. **Generar ventanas consistentes**: controla el tamaño y paso con los flags `--window` y `--step`. Esto garantiza que toda ejecución utilice la misma estructura de secuencias.
3. **Escalar solo con datos normales de entrenamiento**: el pipeline identifica filas con `label == 0` dentro del tramo de entrenamiento y ajusta el `StandardScaler` únicamente con ellas. Así se previene que patrones anómalos contaminen el escalado.
4. **Persistir artefactos**: después de entrenar, guarda el scaler y el modelo (`model.save('artefactos/modelo')`) para reproducir inferencias futuras bajo las mismas transformaciones.

### 2. Calibración basada en validación para elegir el umbral

1. **Calcular errores en validación**: tras el entrenamiento se reconstruyen las ventanas de validación y se calculan sus errores medios cuadrados.
2. **Buscar percentiles candidatos**: el script recorre percentiles entre 80 y 99 sobre los errores de ventanas normales y calcula `precision`, `recall` y `f1` para cada umbral.
3. **Explorar estrategias alternativas**: el script genera métricas para percentiles específicos (85, 90, 95, 97.5, 99), el enfoque basado en IQR y la regla MAD (`median absolute deviation`) para comparar sensibilidad vs. precisión. Los resultados se imprimen en consola y se almacenan en el JSON/Markdown.
4. **Seleccionar el mejor F1**: se escoge automáticamente el percentil que maximiza `F1`, almacenando el umbral elegido (`threshold`) y la métrica alcanzada.
5. **Documentar el valor**: el JSON de salida incluye el percentil y el umbral, permitiendo auditar cómo se definió. En producción, repite la calibración cuando haya nuevos datos etiquetados.

### 3. Métricas auditables y reportes

1. **Generar reporte automático**: ejecuta `python run_pipeline.py --epochs 50 --output artefactos/resultados.json --report artefactos/resumen.md --advanced-models`. Los artefactos incluyen la matriz de confusión, el `classification_report`, `ROC AUC`, `Average Precision`, la comparación de umbrales y los resultados de detectores avanzados.
2. **Visualizar desde el cuaderno**: el notebook produce gráficos de distribuciones de error, curvas ROC/PR y reconstrucciones. Exporta estas figuras en formato PNG para anexarlas a informes.
3. **Registrar ejecuciones**: versiona el JSON y los gráficos en un repositorio o herramienta de experiment tracking (MLflow, Weights & Biases) para comparar iteraciones.
4. **Conectar con métricas de negocio**: complementa las métricas de clasificación con indicadores operativos (volumen recuperado, tiempo de respuesta) añadidos manualmente en tus reportes ejecutivos.


4. **Interpretar resultados**
   - Revisa la tabla de métricas y la matriz de confusión (en el cuaderno) o el archivo JSON generado por `run_pipeline.py` para entender el desempeño del detector.
   - Observa las curvas de entrenamiento, densidades de error y trazas temporales para validar la separación entre comportamientos normales y anómalos.
   - Ajusta los hiperparámetros (tamaño de ventana, percentiles, arquitectura) si necesitas priorizar métricas específicas del negocio. En el script se pueden parametrizar con flags (`--window`, `--step`, `--epochs`, etc.).

codex/review-hydrocarbon-theft-analysis-code-x3whc0
### 4. Optimización de ventana, paso, arquitectura y entrenamiento

El script permite experimentar con configuraciones clave sin editar código:

1. **Cambiar hiperparámetros puntuales**: usa los flags `--window`, `--step`, `--encoder-units`, `--decoder-units`, `--dropout`, `--learning-rate`, `--batch-size`, `--epochs`, `--validation-split` y `--patience` para ejecutar una corrida específica.
2. **Lanzar una búsqueda sistemática**: habilita `--sweep` para recorrer automáticamente combinaciones de ventana/paso, arquitectura y parámetros de entrenamiento. Por defecto, la búsqueda examina:
   - Ventanas `[24, 36, 48]` minutos y pasos `[1, 5, 10]` minutos.
   - Arquitecturas de codificador/decodificador `[(128, 64), (64, 32)]` ↔ `[(64, 128), (32, 64)]` y dropout `[0.1, 0.2]`.
   - Tasas de aprendizaje `[1e-3, 5e-4]`, tamaños de lote `[32, 64]`, épocas `[60, 100]`, paciencia `[5, 10]` y `validation_split` `[0.1, 0.2]`.
   - Personaliza los espacios de búsqueda con `--window-grid`, `--step-grid`, `--encoder-grid`, `--decoder-grid`, `--dropout-grid`, `--lr-grid`, `--batch-grid`, `--epochs-grid`, `--patience-grid` y `--val-split-grid`.
3. **Interpretar la salida**: el modo `--sweep` imprime un JSON ordenado por `F1` sobre anomalías que resume los mejores hiperparámetros, métricas y umbrales encontrados.

### 5. Técnicas avanzadas de detección de anomalías

Para comparar el autoencoder con métodos no secuenciales:

1. Ejecuta `python run_pipeline.py --advanced-models` para entrenar el autoencoder y, adicionalmente, `IsolationForest` y `LocalOutlierFactor` con las mismas ventanas. 
2. El resumen en consola y los artefactos generados mostrarán precisión, recall, F1, ROC-AUC y `Average Precision` de cada detector, facilitando la selección del método más robusto para producción o como modelo de respaldo.

### 6. Presentación de resultados y siguientes pasos

- Exporta los reportes generados (`artefactos/resultados.json` y `artefactos/resumen.md`) para incluirlos en reuniones ejecutivas o tableros de seguimiento. El Markdown resume configuración, calibración, métricas y benchmarking de umbrales/modelos.
- Complementa con las gráficas del cuaderno para visualizar reconstrucciones, distribuciones de error y matrices de confusión.
- Programa ejecuciones periódicas con `--report` y `--output` para generar historiales comparables y alimentar tableros de BI.
- Exporta el `StandardScaler` y los pesos del autoencoder (`model.save()`) para integrar el flujo en servicios de inferencia en tiempo real.
- Considera automatizar la corrida con orquestadores (Prefect, Airflow) y añadir validación `walk-forward` más variables contextuales (válvulas, mantenimiento, clima) para robustecer la generalización.


## Presentación profesional de hallazgos

- Genera reportes ejecutivos con las métricas clave del cuaderno y capturas de las visualizaciones.
- Construye dashboards (Power BI, Tableau o herramientas web) que muestren el error de reconstrucción en tiempo real y el umbral calibrado.
- Documenta supuestos, limitaciones y procedimientos de recalibración en una ficha técnica que acompañe al despliegue del modelo.

Siguiendo estos pasos, obtendrás un detector de anomalías preparado para operar en entornos industriales y una guía clara para su adopción dentro de la organización.

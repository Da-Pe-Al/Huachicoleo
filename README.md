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
   - **Opción script reproducible:** ejecuta `python run_pipeline.py --output artefactos/resultados.json` para entrenar el modelo y generar un reporte JSON con las métricas, el umbral calibrado y los errores de reconstrucción del conjunto de prueba.

3. **Comprender el pipeline**
   - **Carga y ordenamiento de datos**: el dataset se ordena cronológicamente para preservar dependencias temporales.
   - **Generación de ventanas**: se crean secuencias de 30 minutos con un paso de 5 minutos. Cada ventana hereda la etiqueta positiva si contiene al menos una muestra anómala.
   - **División temporal**: se separan los datos en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%) sin mezclar periodos.
   - **Escalamiento sin fuga de información**: el `StandardScaler` se ajusta exclusivamente con observaciones normales del segmento de entrenamiento y se aplica al resto de los datos.
   - **Entrenamiento del autoencoder**: el modelo LSTM con dropout se entrena solo con ventanas normales y utiliza `EarlyStopping` para evitar sobreajuste.
   - **Calibración del umbral**: se calculan los errores de reconstrucción en validación y se selecciona el percentil que maximiza el `F1-score`, equilibrando precisión y recall.
   - **Evaluación final**: en el conjunto de prueba se reportan métricas completas (`precision`, `recall`, `f1`, `ROC AUC`, `Average Precision`) y se generan visualizaciones de errores y matrices de confusión.

4. **Interpretar resultados**
   - Revisa la tabla de métricas y la matriz de confusión (en el cuaderno) o el archivo JSON generado por `run_pipeline.py` para entender el desempeño del detector.
   - Observa las curvas de entrenamiento, densidades de error y trazas temporales para validar la separación entre comportamientos normales y anómalos.
   - Ajusta los hiperparámetros (tamaño de ventana, percentiles, arquitectura) si necesitas priorizar métricas específicas del negocio. En el script se pueden parametrizar con flags (`--window`, `--step`, `--epochs`, etc.).

5. **Siguientes pasos sugeridos**
   - Exportar el `StandardScaler` y los pesos del modelo (`model.save()`) para integrarlos en un servicio de inferencia.
   - Automatizar la ejecución en pipelines (por ejemplo, `mlflow`, `Prefect` o `Airflow`) y programar recalibraciones periódicas del umbral.
   - Incorporar nuevas variables contextuales (válvulas, mantenimiento, clima) y experimentos de validación walk-forward para robustecer la generalización.

## Presentación profesional de hallazgos

- Genera reportes ejecutivos con las métricas clave del cuaderno y capturas de las visualizaciones.
- Construye dashboards (Power BI, Tableau o herramientas web) que muestren el error de reconstrucción en tiempo real y el umbral calibrado.
- Documenta supuestos, limitaciones y procedimientos de recalibración en una ficha técnica que acompañe al despliegue del modelo.

Siguiendo estos pasos, obtendrás un detector de anomalías preparado para operar en entornos industriales y una guía clara para su adopción dentro de la organización.

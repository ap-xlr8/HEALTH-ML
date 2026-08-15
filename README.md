# Health OS - Módulo de Machine Learning (ML)

Este repositorio contiene la arquitectura de Machine Learning para Health OS. Su objetivo exclusivo es **entrenar, evaluar y exportar** modelos de detección de anomalías y clasificación de salud. 

> [!WARNING]
> **Responsabilidades Clave:**
> - **SÍ:** Entrenar y evaluar modelos (Isolation Forest, Random Forest, Redes Neuronales).
> - **SÍ:** Exportar a TFLite (on-device/Android) y ONNX (backend/Go).
> - **SÍ:** Monitorear drift de datos en producción.
> - **NO:** Servir tráfico en producción (no hay APIs Flask/FastAPI sirviendo inferencia aquí).
> - **NO:** Realizar diagnósticos médicos (los modelos **detectan anomalías**, el equipo médico diagnostica).

---

## 🛠 Tecnologías y Stack

- **Lenguaje:** Python 3.11+
- **Gestor de Paquetes:** `uv` (extremadamente rápido, reemplaza pip/conda)
- **Modelado:** `scikit-learn` (baselines), `TensorFlow` / `PyTorch` (avanzados)
- **Exportación:** `TensorFlow Lite` (Android), `ONNX` + `onnxruntime` (Backend Go)
- **MLOps:** `DVC` (Versionado de datos), `MLflow` (Tracking de experimentos)
- **Calidad de Datos:** `Great Expectations` (Validación de esquemas y rangos)
- **Monitoreo:** `Evidently AI` (Data y Model Drift)
- **DevSecOps:** `pytest` (Testing), `ruff` + `mypy` + `black` + `bandit` (Linting, Tipado, Formato, SAST)

---

## 🚀 Setup Local y Entrenamiento

Sigue estos pasos exactamente para configurar tu entorno local.

```bash
# 1. Instalar uv (Package Manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar repo y crear entorno virtual
git clone git@github.com:healthos/health-ml.git
cd health-ml
uv venv
source .venv/bin/activate

# 3. Instalar dependencias exactas
uv pip install -r requirements.txt

# 4. Configurar DVC y descargar datasets
dvc remote add origin s3://healthos-ml-artifacts
# Nota: Requiere credenciales AWS configuradas localmente o en vars
dvc pull 

# 5. Configurar e Iniciar MLflow local
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME="health_os_baselines"
mlflow ui & # Abrir http://localhost:5000 en el navegador

# 6. Correr entrenamiento de prueba
python training/anomaly_detection/heart_rate/train.py
```

---

## ⚙️ Variables de Entorno

Para operar los pipelines, necesitas el archivo `.env` configurado. Pide las claves de acceso a los administradores del sistema.

| Variable | Descripción | Ejemplo | Requerido |
|----------|-------------|---------|-----------|
| `DVC_ACCESS_KEY_ID` | Access key para S3 donde residen datasets DVC | `AKIAIOSFODNN7EXAMPLE` | **Sí** |
| `DVC_SECRET_ACCESS_KEY`| Secret key para el bucket S3 | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | **Sí** |
| `MLFLOW_TRACKING_URI` | URL de tu servidor MLflow remoto o local | `http://mlflow.healthos.internal:5000` | **Sí** |
| `MLFLOW_EXPERIMENT_NAME`| Nombre del experimento para agrupar runs | `heart_rate_anomaly_v1` | Opcional |
| `DATA_ENCRYPTION_KEY` | Clave AES-256 para descifrar datasets RAW y anonimizados | `b14ca5898a4e4133bbce2ea2315a1916` | **Sí** |
| `AUDIT_API_URL` | Webhook para registrar uso de datos (Compliance) | `https://api.healthos.com/v1/audit/ml` | **Sí** |
| `AUDIT_API_TOKEN` | Token de autenticación para Audit log | `h-os-aud-8899aabbccddeeff` | **Sí** |

---

## 📁 Estructura del Proyecto

```text
health-ml/
├── datasets/
│   ├── raw/                    # (Ignorado en git) Datos crudos descifrados.
│   ├── processed/              # (Ignorado en git) Features listas. Trackeado en DVC.
│   ├── splits/                 # Train(70%) / Val(15%) / Test(15%) — Separado rígidamente por patientId.
│   └── expectations/           # Configuración de Great Expectations.
├── preprocessing/
│   ├── anonymizer.py           # Elimina PII y valida 'consent_for_ml_training'
│   ├── feature_extractor.py    # Extrae variabilidad, medias móviles, etc.
│   ├── normalizer.py           # Z-Score y MinMax scalars.
│   └── time_sync.py            # Alineación de series temporales (clock drift de wearables).
├── training/
│   ├── anomaly_detection/
│   │   ├── heart_rate/train.py # Isolation Forest
│   │   ├── spo2/train.py       # Reglas + Heurística Contextual
│   │   └── combined/train.py   # Multivariante AutoEncoder
│   ├── sleep_quality/train.py  # Clasificación Random Forest
│   ├── activity_recognition/   # Clasificación de actividad física
│   └── risk_scoring/train.py   # Modelo ensamblado predictivo. Cloud only.
├── evaluation/
│   ├── metrics.py              # Cálculo de Recall, FPR, Latencia.
│   ├── clinical_thresholds.py  # Validadores automáticos contra reglas médicas.
│   └── reports/                # Salida de matrices de confusión.
├── model-registry/
│   ├── registry.json           # Catálogo versionado con metadata.
│   └── models/                 # Modelos serializados .pkl / .h5 (trackeados por DVC).
├── export/
│   ├── to_tflite.py            # Conversor a TensorFlow Lite para Android.
│   ├── to_onnx.py              # Conversor para ONNX (Backend).
│   └── quantize.py             # Cuantización Int8/FP16.
├── monitoring/
│   ├── drift_check.py          # Evidently AI script para chequeos semanales.
│   └── alerts.py               # Envío de alertas a Slack/PagerDuty.
└── .github/workflows/          # Pipelines de CI/CD.
```

---

## 🛡️ Privacidad, Consentimiento y Auditoría (Compliance Médico)

La privacidad es una regla estricta (HIPAA/GDPR).

1. **Consentimiento Explícito:** Ningún dato entra al pipeline sin `consent_for_ml_training=true`.
2. **Anonimización:** `userId` es destruido y reemplazado con un UUID efímero por sesión de entrenamiento.
3. **Auditoría:** Cada lectura de un dataset crudo dispara un evento HTTP al backend.

### Código de ejemplo (`preprocessing/anonymizer.py`):
```python
import os
import requests
import pandas as pd
from uuid import uuid4

def process_and_anonymize(raw_data_path: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_parquet(raw_data_path)
    
    # 1. Filtro estricto de consentimiento
    if 'consent_for_ml_training' not in df.columns:
        raise ValueError("El dataset carece del flag de consentimiento.")
    
    df_consented = df[df['consent_for_ml_training'] == True].copy()
    
    # 2. Eliminar PII y tokenizar ID
    columns_to_drop = ['email', 'name', 'ip_address', 'device_mac']
    df_consented.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Substituir patientId con tokens efímeros para no mezclar records del mismo paciente en train/test
    patient_map = {pid: str(uuid4()) for pid in df_consented['patient_id'].unique()}
    df_consented['patient_id'] = df_consented['patient_id'].map(patient_map)
    
    # 3. Registrar auditoría (obligatorio)
    requests.post(
        os.environ['AUDIT_API_URL'],
        headers={"Authorization": f"Bearer {os.environ['AUDIT_API_TOKEN']}"},
        json={
            "action": "ml_dataset_read",
            "records_processed": len(df_consented),
            "dataset": dataset_name,
            "purpose": "model_training"
        }
    )
    
    return df_consented
```

---

## 🤖 Catálogo de Modelos

| Modelo | Tipo/Algoritmo | Consumidor | Input Features | Output |
|--------|----------------|------------|----------------|--------|
| `heart_rate_anomaly` | Isolation Forest | Android (TFLite) + Backend (ONNX) | `hr_bpm`, `hrv_ms`, `activity_intensity` | `is_anomaly` (bool), `score` (float) |
| `spo2_critical` | Árbol de Reglas + ML | Android (TFLite) | `spo2_percent`, `elevation_meters`, `is_sleeping` | `critical_drop` (bool) |
| `combined_vitals` | AutoEncoder (Deep Learning) | Backend (ONNX) | `hr_bpm`, `spo2`, `resp_rate`, `temp_c`, `bp_sys`, `bp_dia` | `risk_probability` (0.0-1.0) |
| `sleep_quality` | Random Forest | Android (TFLite) | `movement_variance`, `hr_avg`, `hrv_avg`, `duration_mins` | `sleep_stage` (Int: 1-4) |
| `activity_recognition` | CNN 1D / LSTM | Android (TFLite) | Accel X, Y, Z (50Hz), Gyro X, Y, Z | `activity_class` (Enum) |
| `risk_score` | Gradient Boosting (XGBoost) | Backend (Cloud Only) | Features vitales (históricas 30 días), demografía agregada | `30_day_risk_score` (0.0-100.0) |

---

## 📈 Pipeline de Entrenamiento (Ejemplo End-to-End)

El pipeline automatizado sigue estas métricas clínicas mínimas antes de permitir el export.

### Métricas de Aprobación (Automáticas)
| Métrica | Umbral | Razón Clínica |
|---|---|---|
| **Recall (Sensibilidad)** | `> 95%` | Falsos negativos pueden ser letales (ignorar una anomalía real). |
| **FPR (Tasa Falsos Positivos)** | `< 5%` | Evitar la *fatiga de alarmas* en pacientes y médicos. |
| **Precision** | `> 80%` | Las alertas deben ser accionables y confiables. |
| **Latencia Inferencia** | `< 5ms` | Para TFLite on-device, no bloquear el hilo BLE. |
| **Tamaño en Disco** | `< 2MB` | Para TFLite on-device, respetar almacenamiento del móvil. |

### 1. Validación de Datos (Great Expectations)
Antes de entrenar, verificamos que los datos tengan sentido. (`datasets/expectations/suite.py`):
```python
import great_expectations as ge

def validate_heart_rate_data(df_path: str):
    df = ge.read_parquet(df_path)
    
    # Reglas clínicas duras
    df.expect_column_values_to_be_between("hr_bpm", min_value=30, max_value=220)
    df.expect_column_values_to_not_be_null("patient_id")
    df.expect_column_values_to_be_in_set("activity_intensity", [0, 1, 2, 3])
    
    results = df.validate()
    if not results["success"]:
        raise ValueError(f"Fallo en calidad de datos: {results}")
```

### 2. Entrenamiento con Tracking (MLflow)
Ejemplo de `training/anomaly_detection/heart_rate/train.py`:
```python
import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

def train_hr_anomaly_model(X_train, X_test, y_test):
    with mlflow.start_run(run_name="isolation_forest_v1"):
        # Parámetros
        contamination = 0.05
        n_estimators = 100
        
        mlflow.log_params({"contamination": contamination, "n_estimators": n_estimators})
        
        # Entrenamiento
        model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        model.fit(X_train)
        
        # Evaluación ficticia (reemplazar con metrics.py reales)
        y_pred = model.predict(X_test) # -1 anomalía, 1 normal
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # Calcular recall (ejemplo)
        recall = 0.96 # Asumamos calculado
        fpr = 0.03
        
        mlflow.log_metrics({"recall": recall, "fpr": fpr})
        
        # Validación Clínica Automática
        if recall < 0.95 or fpr > 0.05:
            print("El modelo NO supera los umbrales clínicos.")
            mlflow.log_tag("status", "rejected")
        else:
            joblib.dump(model, "model-registry/models/hr_model.pkl")
            mlflow.sklearn.log_model(model, "isolation_forest")
            mlflow.log_tag("status", "approved_for_export")
            print("Modelo aprobado.")
```

### 3. Exportación a ONNX y TFLite
Archivos bajo `/export/`.

**Export a ONNX (`export/to_onnx.py`):**
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

model = joblib.load("model-registry/models/hr_model.pkl")
# Definimos el input: 3 features continuas
initial_type = [('float_input', FloatTensorType([None, 3]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model-registry/models/hr_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Export a TFLite (`export/to_tflite.py`):**
*(Nota: Scikit-Learn requiere conversión extra. Para redes neuronales en TF/Keras se usa el nativo)*
```python
import tensorflow as tf

# Asumiendo un modelo Keras previamente guardado
converter = tf.lite.TFLiteConverter.from_saved_model("model-registry/models/activity_keras")
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Cuantización (reduce tamaño < 2MB)
tflite_model = converter.convert()

with open("model-registry/models/activity.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 📡 Integración en Producción (Consumo de los Modelos)

El equipo de ML **no sirve los modelos**, exporta artefactos que otros repositorios consumen.

### 1. Integración Backend (Go)
El microservicio en Go consume los modelos `.onnx` para las inferencias de la plataforma en la nube:
```go
// En el repo de backend: internal/ml/model_loader.go
package ml

import "github.com/yalue/onnxruntime_go"

func EvaluateHeartRate(hr float32, hrv float32, activity float32) (bool, error) {
    ort.InitializeEnvironment()
    defer ort.DestroyEnvironment()
    
    session, _ := ort.NewSession("models/hr_model.onnx")
    defer session.Destroy()
    
    // Preparar tensores de entrada y llamar a predict...
    // Inferencia ultrarrápida en memoria en el servidor.
}
```

### 2. Integración App Móvil (Kotlin / Android)
La aplicación móvil descarga modelos `.tflite` vía WiFi y corre la inferencia localmente sin internet para proteger al paciente offline.
```kotlin
// En el repo Android: ml-runtime/ModelRegistry.kt
val modelRegistry = ModelRegistry(context)

// En background (WorkManager):
modelRegistry.downloadLatestModel("heart_rate_anomaly_v1")

// En runtime (cuando llega un dato BLE del wearable):
val anomalyDetector = AnomalyDetector(modelRegistry.getModelPath("heart_rate_anomaly_v1"))
val isAnomaly = anomalyDetector.evaluate(measurement.hr, measurement.hrv, measurement.activity)

if (isAnomaly) {
    NotificationManager.fireCriticalAlert()
}
```

---

## 🕵️ Monitoreo de Data Drift (Evidently AI)

Para detectar cuándo un modelo se degrada en producción (ej: nuevo wearable lanza datos distintos), usamos `Evidently`. Corremos esto semanalmente vía Airflow/Cron.

**`monitoring/drift_check.py`:**
```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference_data = pd.read_parquet("datasets/splits/train_data.parquet")
# Datos de producción de la última semana (agregados, anónimos)
current_data = pd.read_parquet("datasets/raw/prod_last_week.parquet") 

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data)
drift_report.save_html("monitoring/reports/drift_report.html")

summary = drift_report.as_dict()
if summary["metrics"][0]["result"]["dataset_drift"]:
    print("ALERTA: Data drift detectado. Requiere re-entrenamiento.")
    # Código para alertar a Slack/PagerDuty aquí...
```

---

## 🛡️ DevSecOps y CI/CD Pipeline

En cada PR se ejecuta un pipeline estricto en GitHub Actions (`.github/workflows/ml-ci.yml`).

1. **Linting & Code Quality:**
   `ruff check . && black --check . && mypy .`
2. **SAST Security Scanning:**
   `bandit -r . -c pyproject.toml`
3. **Data Validation:** 
   Corre Great Expectations sobre los datasets versionados en DVC.
4. **Unit Tests:**
   `pytest tests/ --cov=preprocessing --cov=evaluation`
5. **Training Run (solo en master/main, en runner con GPU):**
   Descarga datos de DVC -> Ejecuta MLflow pipeline -> Genera ONNX/TFLite.
6. **Upload to Registry:**
   Sube artefactos empaquetados si pasan los umbrales de métricas (Recall > 95%).

---

## ✅ Checklist de Tareas del Equipo (Nuevos Ingresos)

Copia este checklist en tus issues para garantizar el cumplimiento del proceso:

- [ ] Verificar que tengo acceso al bucket S3 de DVC (`dvc pull` funciona).
- [ ] Verificar que puedo acceder a la UI del servidor MLflow remoto.
- [ ] Revisar el script `preprocessing/anonymizer.py` y confirmar que no hay Data Leakage (mismo `patient_id` en train y test).
- [ ] Correr tests localmente (`pytest`) antes de crear el PR.
- [ ] Ejecutar validaciones estáticas (`ruff`, `mypy`, `bandit`).
- [ ] Asegurar que el modelo propuesto registre su hiperparámetros en MLflow.
- [ ] Verificar que las exportaciones ONNX y TFLite pasan la validación de tamaño (< 2MB para móvil).
- [ ] Validar que el pipeline de predicción cumple Latencia < 5ms (medir con CPU estrangulada).
- [ ] Solicitar revisión de par a un Data Scientist Senior y a un Médico Asesor.

---

## ⚠️ NOTAS DE COORDINACION DE EQUIPO — LEER ANTES DE ARRANCAR

> El equipo de ML es el mas independiente del proyecto, pero tiene dependencias criticas de datos y de entrega de artefactos.

---

### [DEPENDENCIA CRITICA] Sin datos reales no hay training — coordina el acceso desde el dia 1

El modelo no se puede entrenar sin datos de mediciones biometricas. Los datos vienen de la base de datos del backend (MongoDB Atlas).

**Lo que necesitas del equipo de Backend antes de empezar training:**
1. Un dump anonimizado de la coleccion `health_measurements` (con `consent_for_ml_training=true`)
2. El schema exacto de esa coleccion (nombres de campos, tipos, unidades de cada metrica)
3. Acceso de solo lectura a la instancia de staging de MongoDB (para extraer datos periodicamente)

**Mientras no tengas datos reales:**
- Genera datos sinteticos con la forma del schema real (ver `datasets/synthetic_generator.py`)
- Los datos sinteticos te permiten desarrollar y probar el pipeline completo
- NO entrenes el modelo final con datos sinteticos — los umbrales clinicos no seran confiables

**Cuando el backend tenga usuarios reales en staging:**
- Pide el dump anonimizado al equipo de Backend
- Verificar que `consent_for_ml_training=true` antes de usar cualquier registro
- Documentar en el audit log que se usaron esos datos (ver seccion de privacidad)

---

### [ENTREGA A OTROS EQUIPOS] Como entregas los modelos — esto es lo mas importante que produces

Tu trabajo no termina cuando el modelo entrena bien. Termina cuando el modelo esta integrado y funcionando en Mobile y Backend.

**Entrega al equipo de Mobile (Android):**
- Archivo: `model-registry/models/[modelo]/[version]/model.tflite`
- Metadata: `model-registry/models/[modelo]/[version]/metadata.json`
- El metadata DEBE incluir: version, feature schema version, threshold, metricas de evaluacion
- Cuando tengas un nuevo modelo listo: avisa al equipo de Mobile con la version y el archivo
- Mobile descargara el modelo automaticamente si lo subes al registry — pero deben saber que existe

**Entrega al equipo de Backend (Go/ONNX):**
- Archivo: `model-registry/models/[modelo]/[version]/model.onnx`
- El backend lo carga al arrancar el servidor
- Cuando actualices el modelo: el backend necesita reiniciar para cargar la nueva version
- Coordina el deploy del nuevo modelo con el deploy del backend para que coincidan

**Regla de versiones:**
```
Si cambias el feature schema (los campos de entrada del modelo):
  - La version MAYOR sube (1.x.x -> 2.0.0)
  - El equipo de Mobile y Backend DEBEN actualizar su extractor de features
  - Avisa con minimo 3 dias de anticipacion

Si solo mejoras las metricas sin cambiar el schema:
  - La version MENOR sube (1.2.x -> 1.3.0)
  - Mobile y Backend pueden actualizar sin cambios de codigo
  - Solo avisa cuando el modelo este listo
```

---

### [ACCION UNICA] URL de staging de la API para el audit log

El pipeline de training necesita registrar en el audit log del backend que se usaron ciertos datos. Para eso necesita la URL del backend de staging y un token de servicio.

Pide al equipo de Backend:
```
AUDIT_API_URL=https://[nombre-real].onrender.com
AUDIT_API_TOKEN=[token de servicio para ML]
```

Guardalo en tus secretos de GitHub y en tu `.env.local`. No lo vuelvas a pedir.

---

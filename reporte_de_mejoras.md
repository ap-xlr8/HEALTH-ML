# 🧠 Reporte de Mejoras — Machine Learning (`ML`)

**Proyecto:** Health OS  
**Módulo:** `health-ml` (Python 3.11 / scikit-learn / TFLite + ONNX)  
**Fecha de Auditoría:** 2026-08-16  
**Auditor:** Agente de Diagnóstico Automatizado  
**Versión del Reporte:** 1.0.0

---

## 1. Auditoría de DevSecOps y Fuga de Información

### 1.1 Detección de Secretos y Datos Sensibles Hardcodeados

| Hallazgo | Severidad | Archivo | Detalle |
|:---|:---:|:---|:---|
| **Credenciales de ejemplo en `.env.example`** | ✅ OK | `.env.example` L2-3 | `DVC_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE` y `DVC_SECRET_ACCESS_KEY=wJalrXUtnFEMI/...EXAMPLEKEY` — Credenciales de ejemplo de AWS documentation. No son reales. |
| **Fallback de `DATA_ENCRYPTION_KEY`** | 🟠 ALTO | `train_all.py` L36 | `enc_key = os.getenv("DATA_ENCRYPTION_KEY") or secrets.token_hex(32)` — Si la variable de entorno no está configurada, genera una clave efímera aleatoria. Esto **NO** es un secreto hardcodeado (fue corregido en v1.0.0 según CHANGELOG). Sin embargo, en entornos sin la variable configurada, los datos se cifran con clave irrecuperable. |
| **Rutas locales de artefactos** | 🟡 MEDIO | `model-registry/registry.json` | Paths locales del sistema de archivos (`C:/Users/alexi/AppData/Local/Temp/pytest-of-alexi/...`). No son secretos, pero exponen estructura del filesystem del desarrollador. | 
| **Sin archivo `.env` real encontrado** | ✅ OK | N/A | No se detectó `.env` con credenciales reales en el directorio. `.gitignore` excluye `.env` y `.env.local`. |

> [!NOTE]
> La remediación del CHANGELOG v1.0.0 ("Eliminación de clave AES fallback hardcodeada") fue correctamente implementada. El patrón actual `secrets.token_hex(32)` genera claves efímeras seguras pero irrecuperables — apropiado para tests, pero en producción `DATA_ENCRYPTION_KEY` debe ser obligatorio (fail-closed).

### 1.2 Análisis de Prácticas DevSecOps en Pipeline CI/CD

| Práctica | Estado | Evidencia |
|:---|:---:|:---|
| Secret Scanning (Gitleaks) | ✅ Implementado | `ml-ci.yml` L10-22 |
| SAST (Bandit) | ✅ Implementado | `ml-ci.yml` L56-58 — Escaneo de todos los módulos con `-ll` (medium+). |
| SCA (pip-audit) | ✅ Implementado | `ml-ci.yml` L43-45 — Auditoría de dependencias con descripción. |
| Linting (Ruff + Black) | ✅ Implementado | `ml-ci.yml` L47-50 — Ruff check + Black format check. |
| Type Checking (Mypy) | ✅ Implementado | `ml-ci.yml` L52-54 — Verificación estricta de tipos. |
| Tests + Coverage Gate (≥80%) | ✅ Implementado | `ml-ci.yml` L60-62 — pytest con `--cov-fail-under=80`. |
| Training Pipeline en CI | ✅ Implementado | `ml-ci.yml` L64-88 — Ejecución completa de `train_all.py` + validación del registry. |
| Model Validation Gate | ✅ Implementado | `ml-ci.yml` L86-88 — Verifica que el registry contenga exactamente 7 modelos. |
| SBOM / Dependency Lock | ✅ Implementado | `requirements.lock` presente (405 bytes) — Dependencias pinned determinísticamente. |

### 1.3 Gestión de Variables de Entorno

| Criterio | Estado | Observación |
|:---|:---:|:---|
| `.env` en `.gitignore` | ✅ | `.gitignore` L39-40 excluye `.env` y `.env.local`. |
| `.env.example` con placeholders | ✅ | Contiene marcadores `EXAMPLE`/`GENERATE_SECURE_*` — nunca valores reales. |
| Datasets raw excluidos | ✅ | `.gitignore` L18 excluye `datasets/raw/` (archivos grandes y potencialmente sensibles). |
| Evaluation reports excluidos | ✅ | `.gitignore` L36 excluye `evaluation/reports/*.json` — generados dinámicamente. |
| DVC cache excluido | ✅ | `.gitignore` L26-27 excluye `.dvc/cache` y `.dvc/tmp`. |

---

## 2. Checklist de Madurez Técnica (12 Ejes)

| # | Eje | Estatus | Observaciones |
|:---:|:---|:---:|:---|
| 1 | **Requerimientos y Arquitectura** | ✅ Cumple | Pipeline end-to-end: `datasets/` → `preprocessing/` → `training/` → `evaluation/` → `export/` → `model-registry/` → `monitoring/`. 7 modelos en catálogo, exportación multi-formato (PKL, ONNX, TFLite). |
| 2 | **Desarrollo y Estándares de Código** | ✅ Cumple | Ruff (line-length 100), Black, Mypy strict, Bandit SAST. `pyproject.toml` bien configurado con metadata formal. |
| 3 | **Git y Control de Versiones** | ✅ Cumple | `.gitignore` exhaustivo para Python, datasets, DVC, MLflow, env vars. |
| 4 | **CI/CD** | ✅ Cumple | Pipeline completo: Secret Scan → pip-audit → Ruff/Black → Mypy → Bandit → Tests (≥80%) → Training → Registry Validation. |
| 5 | **Testing y QA** | ✅ Cumple | 6 test files: `test_anonymizer`, `test_clinical_thresholds`, `test_exports`, `test_feature_extractor`, `test_models_training`, `test_synthetic_data`. Coverage gate ≥80% enforced en CI. |
| 6 | **DevSecOps** | ✅ Cumple | Gitleaks, pip-audit, Bandit, cobertura enforced. Mejor pipeline DevSecOps del ecosistema para un módulo ML. |
| 7 | **Seguridad de Aplicación y Datos** | ✅ Cumple | `anonymizer.py` con: verificación de consentimiento obligatoria, eliminación de PII (HIPAA Safe Harbor), tokenización de `patient_id` con UUID efímeros, audit log emitido a API centralizada. `split_by_patient` previene data leakage. |
| 8 | **Datos y BD** | ⚠️ Requiere ajuste | Datasets sintéticos para training en CI. **Observación:** todos los modelos en el registry tienen `status: "rejected"` excepto `activity_recognition` (`"production"`). Los modelos rechazados no cumplen umbrales clínicos (ej. `heart_rate_anomaly`: Recall 70% < 95% requerido; `combined_vitals`: Recall 30%, AUC 0.61). |
| 9 | **Observabilidad y Monitoreo** | ✅ Cumple | `DriftDetector` con KS-Test + PSI implementado. Reporte JSON automatizado. Alertas de drift configuradas. Webhook Slack/PagerDuty en `.env.example`. |
| 10 | **Resiliencia, Backups y DR** | ⚠️ Requiere ajuste | Model Registry local (`registry.json`). **Falta:** registro en sistema externo (MLflow/W&B) para versionado durable. DVC referenciado en `.gitignore` pero sin configuración activa visible. |
| 11 | **Compliance y Auditoría Médica** | ✅ Cumple | Consentimiento verificado antes de usar datos. PII eliminada. Audit log emitido. Modelo con disclaimer: "ML nunca emite diagnósticos" (ADR formal). Cifrado en reposo de datasets con AES. |
| 12 | **Operación, Incidentes y Mejora Continua** | ⚠️ Requiere ajuste | Drift check semanal configurado. **Falta:** scheduling automatizado (cron job o CI scheduled) para ejecutar `drift_check.py` periódicamente. Alertas de drift documentadas en Runbook pero sin webhook real configurado. |

---

## 3. Plan de Nuevas Funcionalidades — Pipeline de Estimaciones Biométricas

### 3.1 Catálogo Actual de Modelos

| # | Modelo | Algoritmo | Target | Deploy | Status Registry | Métricas Clave |
|:---:|:---|:---|:---|:---|:---:|:---|
| 1 | `heart_rate_anomaly` | IsolationForest | Anomalías HR | Android + Backend | ❌ Rejected | Recall: 70%, F1: 0.68, AUC: 0.975 |
| 2 | `spo2_critical` | RandomForest Weighted | SpO2 crítico | Android | ❌ Rejected | Recall: 100%, F1: 1.0, AUC: 1.0 |
| 3 | `combined_vitals` | MLP AutoEncoder | Multi-vital anomaly | Backend | ❌ Rejected | Recall: 30%, F1: 0.17, AUC: 0.61 |
| 4 | `sleep_quality` | RandomForest | Sleep stage | Android | ❌ Rejected | Accuracy: 82%, 128 val samples |
| 5 | `activity_recognition` | MLP Classifier | Activity class | Android | ✅ Production | Accuracy: 100%, 480 val samples |
| 6 | `glucose_patterns` | GradientBoosting | Metabolic flag | Android + Backend | ❌ Rejected | Recall: 65%, F1: 0.74, AUC: 0.99 |
| 7 | `risk_score` | HistGradientBoosting | 30-day risk | Backend | ❌ Rejected | R²: 0.94, RMSE: 3.78 |

> [!WARNING]
> **6 de 7 modelos están en estado `rejected`**. Los modelos rechazados no cumplen los quality gates clínicos (Recall ≥ 95%, FPR ≤ 5%). Se requiere reentrenamiento con datasets más grandes y representativos. El `spo2_critical` muestra métricas perfectas (AUC=1.0) que sugieren overfitting en datos sintéticos — requiere validación con datos reales.

### 3.2 Nuevos Modelos Requeridos para Estimaciones Biométricas

| # | Condición / Métrica | Sensores | Mecanismo Fisiológico | Precisión Esperada | Estado | Acción |
|:---:|:---|:---|:---|:---:|:---:|:---|
| 8 | **Picos de glucosa / Riesgo metabólico** | PPG (HR+HRV) + EDA + Acelerómetro | Activación simpática postprandial: caída de HRV, aumento de HR en reposo, respuesta galvánica. | Experimental | ⚠️ Parcial (`glucose_patterns` existe) | **Mejorar:** agregar features de EDA y HRV (rMSSD, SDNN). Actual solo usa `hr_bpm, hrv_ms, movement_variance, temp_c`. Necesita features de conductancia electrodérmica. |
| 9 | **Arritmias (FA, Taquicardia/Bradicardia)** | PPG continuo + ECG bajo demanda | Detección de intervalos R-R irregulares en PPG; confirmación con ECG de 1 derivación. | Alta (estándar clínico) | ⚠️ Parcial (`heart_rate_anomaly` existe) | **Mejorar:** cambiar de IsolationForest a modelo secuencial (LSTM/1D-CNN) sobre ventanas de R-R intervals. Agregar clasificación multi-clase: Normal, FA, Taquicardia, Bradicardia. Recall actual (70%) insuficiente para uso clínico. |
| 10 | **Apnea del sueño / Hipoxemia nocturna** | Oxímetro (SpO₂) + Acelerómetro | Caídas transitorias de SpO₂ < 90% con microdespertares y cese de movimiento. | Media-Alta (cribado) | ⚠️ Parcial (`spo2_critical` + `sleep_quality`) | **Crear modelo dedicado:** `sleep_apnea_screening` que combine features de SpO₂ nocturno + acelerometría. Métrica clave: AHI (Apnea-Hypopnea Index) estimado. |
| 11 | **Estrés crónico y fatiga del SNC** | HRV (rMSSD, SDNN) + EDA + Temp | Predominio simpático: descenso de HRV matutino y picos de conductancia electrodérmica. | Alta (tendencias) | ❌ No existe | **Crear:** `stress_fatigue_cns` model. Features: `hrv_rmssd_morning, hrv_sdnn_24h, eda_peak_count, eda_tonic_level, skin_temp_nocturnal`. Algoritmo sugerido: Gradient Boosting con ventanas temporales de 7 días. |
| 12 | **Detección temprana infecciones / Fiebre** | Temp cutánea + HR en reposo | Elevación de temperatura periférica basal nocturna con taquicardia en reposo (+8 a 10 lpm por +1°C). | Alta (pre-síntomas) | ❌ No existe | **Crear:** `early_infection_detection` model. Features: `temp_basal_nocturnal_delta, resting_hr_delta, hrv_suppression_index`. Algoritmo: RandomForest con umbral calibrado para alta sensibilidad. |
| 13 | **Riesgo de hipertensión / Rigidez arterial** | PPG (morfología de onda) + PTT | Medición de velocidad del pulso a la periferia (PTT); correlación con resistencia vascular. | Media (tendencias) | ❌ No existe | **Crear:** `hypertension_risk` model. Features: `ptt_ms, ppg_augmentation_index, ppg_stiffness_index, age, bmi`. Algoritmo: Ridge Regression para estimación continua de presión arterial media. |
| 14 | **Riesgo cardiovascular global / VO₂máx** | PPG + Acelerómetro + GPS | Estimación de VO₂máx y velocidad de recuperación cardíaca post-esfuerzo (minutos 1 y 2). | Alta (seguimiento funcional) | ❌ No existe | **Crear:** `vo2max_estimation` model. Features: `recovery_hr_1min, recovery_hr_2min, max_hr_during_exercise, exercise_duration_s, distance_m, pace_min_km, age, weight_kg`. Algoritmo: Gradient Boosting Regressor calibrado contra VO₂máx de referencia. |

### 3.3 Diseño Técnico de Nuevos Pipelines

```
Para cada nuevo modelo:

1. datasets/synthetic_generator.py
   └── Agregar generador de features sintéticas específicas del modelo

2. preprocessing/feature_extractor.py
   └── Agregar extractor de features del raw measurement stream

3. training/<model_name>/train.py
   └── Pipeline: load → preprocess → split_by_patient → train → evaluate → save

4. evaluation/clinical_thresholds.py
   └── Definir quality gates clínicos específicos:
       - Anomaly detection: Recall ≥ 95%, FPR ≤ 5%
       - Classification: Accuracy ≥ 90%, per-class Recall ≥ 85%
       - Regression: R² ≥ 0.80, RMSE dentro de rango clínico aceptable

5. export/to_tflite.py + to_onnx.py
   └── Exportar a formato de deployment target

6. model-registry/
   └── Registrar automáticamente con metadata completa

7. monitoring/drift_check.py
   └── Agregar features del nuevo modelo al drift monitoring
```

### 3.4 Mejoras Transversales del Pipeline

| Área | Estado Actual | Acción Requerida | Prioridad |
|:---|:---|:---|:---:|
| Datasets — solo sintéticos | ⚠️ 30 pacientes × 120 registros | Escalar a ≥ 500 pacientes con distribuciones realistas. Incorporar datos reales anonimizados cuando estén disponibles vía `anonymizer.py`. | P0 |
| Versionado de modelos (MLflow/W&B) | ❌ Solo registry JSON local | Integrar MLflow para experiment tracking y model registry durable. Registrar hiperparámetros, métricas, artefactos. | P1 |
| DVC para datasets | ⚠️ Referenciado en `.gitignore` pero sin `.dvc/` config | Activar DVC con storage backend (S3/GCS/Azure) para versionado reproducible de datasets. | P1 |
| Great Expectations (Data Quality) | ❌ No implementado | Integrar Great Expectations para validación de esquemas de datos pre-entrenamiento. Verificar: completitud, rangos fisiológicos, distribuciones esperadas. | P1 |
| Evidently AI (Drift continuo) | ⚠️ Custom `DriftDetector` implementado | Evaluar si migrar a Evidently AI para reportes de drift más ricos con visualizaciones HTML y integraciones nativas. | P2 |
| Scheduled drift check | ❌ Solo manual | Configurar GitHub Actions scheduled workflow (`cron: '0 6 * * 1'`) para ejecutar `drift_check.py` semanalmente con webhook de alerta. | P1 |
| Modelos rechazados — reentrenamiento | 6/7 modelos `rejected` | Priorizar reentrenamiento de `heart_rate_anomaly` (mejorar Recall de 70% a ≥95%) y `combined_vitals` (mejorar AUC de 0.61 a ≥0.85). Considerar cambio de algoritmo. | P0 |
| Validación con datos clínicos reales | ❌ Solo datos sintéticos | Establecer protocolo de validación clínica: dataset de validación con ground truth de dispositivos certificados FDA/CE. | P0 |

---

## 4. Acciones Inmediatas (P0)

> [!CAUTION]
> Acciones bloqueantes para el módulo ML:

1. **Reentrenar modelos rechazados** — 6 de 7 modelos no cumplen umbrales clínicos. Prioridad: `heart_rate_anomaly` (Recall 70% → ≥95%) y `combined_vitals` (AUC 0.61 → ≥0.85).
2. **Validar `spo2_critical` contra overfitting** — Métricas perfectas (AUC=1.0, F1=1.0) en datos sintéticos son sospechosas. Requiere validación con datos reales.
3. **Escalar datasets** — 30 pacientes sintéticos son insuficientes para modelos clínicos. Escalar a ≥500 pacientes.
4. **Hacer `DATA_ENCRYPTION_KEY` obligatorio (fail-closed)** — En `train_all.py` L36, el fallback a clave efímera debe lanzar excepción si no está configurada en entornos no-dev.
5. **Activar DVC** para versionado de datasets — Actualmente solo referenciado en `.gitignore` sin configuración real.
6. **Crear los 5 modelos faltantes** — Stress/CNS fatigue, Early infection, Hypertension risk, VO₂max, Sleep apnea screening.

---

## 5. Resumen Ejecutivo

| Categoría | Nota |
|:---|:---:|
| DevSecOps Pipeline | **A+** (10/10) — Mejor pipeline del ecosistema: Gitleaks, pip-audit, Bandit, Mypy, Ruff, Black, Coverage ≥80%, Training + Registry validation |
| Seguridad de Datos / HIPAA | **A** (9/10) — Anonymizer con consentimiento, PII removal, tokenización, audit log, cifrado en reposo |
| Testing y QA | **A-** (8.5/10) — 6 test files, coverage enforced, training end-to-end en CI |
| Arquitectura y Código | **A** (9/10) — Pipeline bien modularizado, separación clara de responsabilidades |
| Calidad de Modelos | **D** (3/10) — 6/7 modelos rechazados, solo `activity_recognition` en producción |
| Observabilidad ML | **B** (7/10) — Drift detection implementado, falta scheduling y alertas activas |
| Datos de Entrenamiento | **D** (3.5/10) — Solo 30 pacientes sintéticos, sin datos reales, DVC inactivo |
| Cobertura de Estimaciones | **C** (5/10) — 7 modelos existentes de 12 requeridos, 5 nuevos por crear |

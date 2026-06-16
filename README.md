# Student Performance Predictor API

**Taller 4 - Servicio de Acceso a Modelos de Aprendizaje Automático en la Nube**

Maestría CODING — Universidad del Valle
**Nestor Cardona** (202502968) | **Carlos Tovar** (202503011) | **Yolima Guadir** (202502581)

---

## 📋 Descripción

API REST desarrollada con Flask para predecir el rendimiento estudiantil utilizando Machine Learning. El sistema implementa mejoras propuestas del **Taller 3**, incluyendo **Ensemble Learning** con Voting Regressor y optimización avanzada con Random Search.

### ✨ Características Principales

- ✅ **3 Modelos ML Optimizados** - SVR, Gradient Boosting, Random Forest (R² > 0.82)
- ✅ **Selección de Modelo en Tiempo Real** - Elige el algoritmo desde la interfaz o API
- ✅ **Ensemble Learning** combinando SVR + Gradient Boosting + Random Forest
- ✅ **Feature Engineering Avanzado** - Eficiencia cognitiva, compromiso académico, cuartiles
- ✅ **API REST** documentada con endpoints JSON
- ✅ **Interfaz Web** moderna con Tailwind CSS
- ✅ **Serialización** de modelos con pickle
- ✅ **Deploy** en Render.com
- ✅ **Predicciones** individuales y por lotes

---

## 🚀 Demo en Vivo

🌐 **URL de la API**: https://your-app.onrender.com
📚 **Documentación**: https://your-app.onrender.com/docs

---

## 🏗️ Arquitectura del Proyecto

```
StudentPerformanceAPI/
├── app.py                                    # Servidor Flask principal (carga múltiples modelos)
├── model_training.py                         # Entrenamiento SVR + Ensemble
├── model_training_gradient_boosting.py       # Entrenamiento Gradient Boosting
├── model_training_random_forest.py           # Entrenamiento Random Forest
├── test_api.py                               # Suite de pruebas
├── requirements.txt                          # Dependencias Python
├── Procfile                                  # Configuración Render/Heroku
├── runtime.txt                               # Versión de Python
├── README.md                                 # Documentación
├── .gitignore                                # Archivos ignorados
├── data/
│   └── student_dataset_10000_rows.csv        # Dataset (10,000 registros)
├── models/                                   # Modelos entrenados (generados por scripts)
│   ├── student_performance_model.pkl         # SVR + Ensemble (por defecto)
│   ├── gradient_boosting_model.pkl           # Gradient Boosting optimizado
│   └── random_forest_model.pkl               # Random Forest optimizado
└── templates/
    ├── index.html                            # Interfaz con selector de modelo
    └── api_docs.html                         # Documentación API completa
```

---

## 🧠 Modelos de Machine Learning

### 🎯 Modelos Disponibles

La API soporta **3 modelos de Machine Learning** que se pueden seleccionar en tiempo real:

| Modelo | R² | RMSE | MAE | Características | Uso Recomendado |
|--------|-----|------|-----|-----------------|-----------------|
| **SVR_Optimized** | 0.8350 | 6.05 | 4.51 | **Mejor modelo**. Ensemble Voting (SVR+GB+RF) | Predicciones generales balanceadas |
| **GradientBoosting_Optimized** | 0.8229 | 6.31 | 4.83 | Boosting iterativo, patrones no lineales | Datasets con relaciones complejas |
| **RandomForest_Optimized** | 0.8183 | 6.39 | 4.73 | Ensemble de árboles, robusto | Datos con ruido o valores atípicos |

**Cómo seleccionar modelo:**
- **Interfaz Web:** Selector desplegable en "Modelo de Predicción"
- **API:** Parámetro `model_type` en el JSON (ej: `"model_type": "GradientBoosting_Optimized"`)

---

### Mejoras Implementadas del Taller 3

#### 1. **Feature Engineering Avanzado (NUEVO) ⭐**
```python
# 1. Eficiencia Cognitiva
cognitive_efficiency = study_hours / (sleep_hours + 1e-5)

# 2. Índice de Compromiso Académico
academic_engagement = (assignments_completed × attendance) / 100

# 3. Cuartiles de Previous Score (captura umbrales no lineales)
previous_score_quartile = pd.qcut(previous_score, q=4,
    labels=['Q1_Low', 'Q2_Medium', 'Q3_Good', 'Q4_Excellent'])
```

**Justificación:**
- **Eficiencia Cognitiva**: Captura la relación entre horas de estudio y descanso
- **Compromiso Académico**: Combina dos indicadores clave de dedicación
- **Cuartiles**: Modela transiciones críticas en el rendimiento previo

#### 2. **SVR con Optimización Ampliada**
```python
# Espacio de búsqueda ampliado
param_distributions = {
    'C': uniform(1, 20),          # vs 0.1-10 en Taller 3
    'epsilon': uniform(0.01, 0.2), # vs 0.01-0.5
    'n_iter': 100                  # vs 50
}
```

#### 3. **Ensemble Learning (NUEVO)**
```python
# Voting Regressor combinando los 3 mejores modelos
estimators = [
    ('svr', SVR_optimized),
    ('gb', GradientBoosting),
    ('rf', RandomForest)
]
voting_model = VotingRegressor(estimators=estimators)
```

### Métricas de Rendimiento

**Modelos Individuales (entrenados por separado):**

| Modelo | R² Test | RMSE | MAE | Archivo |
|--------|---------|------|-----|---------|
| **Gradient Boosting Optimizado** | **0.8229** | **6.3126** | **4.8338** | `gradient_boosting_model.pkl` |
| **Random Forest Optimizado** | **0.8183** | **6.3943** | **4.7309** | `random_forest_model.pkl` |

**Modelos en SVR + Ensemble (archivo principal):**

| Modelo | R² Test | RMSE | MAE | Observación |
|--------|---------|------|-----|-------------|
| SVR Optimizado | 0.8293 | 6.20 | 4.66 | Baseline del Ensemble |
| Gradient Boosting | 0.8200 | 6.37 | 4.90 | Componente del Ensemble |
| Random Forest | 0.8172 | 6.41 | 4.78 | Componente del Ensemble |
| **Ensemble Voting** | **0.8350** | **6.05** | **4.51** | **Mejor modelo (+0.7%)** ✅ |

**Nota:** Los modelos individuales (GB y RF entrenados por separado) tienen métricas ligeramente diferentes debido a diferencias en la optimización de hiperparámetros (Random Search con más iteraciones).

---

## 📦 Instalación Local

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/student-performance-api.git
cd StudentPerformanceAPI
```

### 2. Crear Entorno Virtual

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar los Modelos

**IMPORTANTE:** Se deben entrenar los 3 modelos por separado antes del despliegue.

```bash
# 1. Entrenar SVR Optimizado (modelo por defecto con Ensemble)
python model_training.py

# 2. Entrenar Gradient Boosting Optimizado
python model_training_gradient_boosting.py

# 3. Entrenar Random Forest Optimizado
python model_training_random_forest.py
```

**Nota:** Cada script tarda entre 5-10 minutos en completar el entrenamiento dependiendo del hardware.

#### Salida esperada de `model_training.py`:
```
================================================================================
🚀 INICIANDO ENTRENAMIENTO DE MODELOS
================================================================================
📥 Cargando dataset desde data/student_dataset_10000_rows.csv...
✓ Dataset cargado: 10000 registros, 8 columnas

🔧 Preprocesando datos...
  🎨 Creando features engineered...
    ✓ cognitive_efficiency: study/sleep ratio
    ✓ academic_engagement: (assignments × attendance)/100
    ✓ previous_score_quartile: cuartiles Q1-Q4
✓ Preprocesamiento completado
  - Train set: 7000 registros
  - Test set: 3000 registros
  - Features originales: 7
  - Features engineered: 3 nuevas
  - Total features: 15

🔬 Entrenando SVR con Random Search...
✓ SVR entrenado
  - R² Test: 0.8293
  - RMSE: 6.1981

🎯 Entrenando Ensemble (Voting Regressor)...
✓ Ensemble entrenado
  - R² Test: 0.8350
  - RMSE: 6.0521

💾 Guardando modelo en student_performance_model.pkl...
✓ Modelo guardado exitosamente
```

### 5. Ejecutar el Servidor

```bash
# Desarrollo
python app.py

# Producción (con gunicorn)
gunicorn app:app
```

El servidor estará disponible en: **http://127.0.0.1:5000**

---

## 🌐 Deploy en Render.com

### Paso 1: Preparar el Repositorio

**IMPORTANTE:** Entrenar los 3 modelos antes de hacer el deploy.

Asegúrate de tener estos archivos en la raíz:

- ✅ `requirements.txt`
- ✅ `Procfile`
- ✅ `runtime.txt`
- ✅ `app.py`
- ✅ `data/student_dataset_10000_rows.csv` (dataset)
- ✅ **Modelos entrenados (requeridos):**
  - `models/student_performance_model.pkl` (SVR + Ensemble)
  - `models/gradient_boosting_model.pkl` (Gradient Boosting)
  - `models/random_forest_model.pkl` (Random Forest)

**Nota:** Si no se incluyen los 3 modelos, la API solo cargará los modelos disponibles. Se recomienda incluir los 3 para tener todas las opciones de predicción.

### Paso 2: Crear Nuevo Web Service en Render

1. **Ir a Render Dashboard**: https://dashboard.render.com/
2. **New > Web Service**
3. **Conectar repositorio de GitHub**

### Paso 3: Configurar el Servicio

```yaml
Name: student-performance-api
Environment: Python 3
Region: Oregon (US West) o más cercana
Branch: main
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Instance Type: Free (o Starter si necesitas más recursos)
```

### Paso 4: Variables de Entorno (Opcional)

```bash
PYTHON_VERSION=3.9.20
DEBUG=False
```

### Paso 5: Deploy

Hacer clic en **Create Web Service**. El deploy toma ~5-10 minutos.

(https://studentperformanceapi.onrender.com)

### Verificar Deploy

```bash
# Health check
curl https://your-app.onrender.com/api/health

# Respuesta esperada:
{
  "status": "ok",
  "model_name": "Ensemble_Voting",
  "model_metrics": {
    "r2": 0.835,
    "rmse": 6.05,
    "mae": 4.51
  }
}
```

---

## 📚 Documentación de la API

### Endpoints Disponibles

#### 1. **GET** `/api/health`
Verifica el estado del servidor y modelo.

**Respuesta:**
```json
{
  "status": "ok",
  "model_name": "Ensemble_Voting",
  "model_metrics": {...},
  "trained_date": "2025-01-15T10:30:00"
}
```

#### 2. **GET** `/api/model/info`
Obtiene información detallada del modelo por defecto.

**Respuesta:**
```json
{
  "best_model": "Ensemble_Voting",
  "all_models_metrics": {...},
  "feature_names": ["study_hours", ...],
  "num_features": 28
}
```

#### 3. **GET** `/api/models/available` ⭐ NUEVO
Lista todos los modelos de ML disponibles con sus métricas.

**Respuesta:**
```json
{
  "available_models": {
    "SVR_Optimized": {
      "metrics": {"r2": 0.8350, "rmse": 6.05, "mae": 4.51},
      "trained_date": "2026-06-16T10:30:00"
    },
    "GradientBoosting_Optimized": {
      "metrics": {"r2": 0.8229, "rmse": 6.3126, "mae": 4.8338},
      "trained_date": "2026-06-16T11:00:00"
    },
    "RandomForest_Optimized": {
      "metrics": {"r2": 0.8183, "rmse": 6.3943, "mae": 4.7309},
      "trained_date": "2026-06-16T11:30:00"
    }
  },
  "default_model": "SVR_Optimized"
}
```

#### 4. **POST** `/api/predict`
Realiza una predicción individual.

**Request:**
```json
{
  "study_hours": 6,
  "attendance": 90,
  "sleep_hours": 8,
  "internet_usage": 3,
  "assignments_completed": 10,
  "previous_score": 75,
  "placement_status": "Placed",
  "model_type": "GradientBoosting_Optimized"  // Opcional: SVR_Optimized (defecto), GradientBoosting_Optimized, RandomForest_Optimized
}
```

**Parámetros:**
- `model_type` (opcional): Selecciona el modelo a usar. Por defecto: `SVR_Optimized`

**Nota:** Las siguientes features se calculan automáticamente:
- `cognitive_efficiency = study_hours / sleep_hours`
- `academic_engagement = (assignments_completed × attendance) / 100`
- `previous_score_quartile` = Cuartil basado en previous_score

**Respuesta:**
```json
{
  "prediction": 78.45,
  "confidence_interval": {
    "lower": 66.29,
    "upper": 90.61
  },
  "performance_category": {
    "category": "Bueno",
    "description": "Rendimiento satisfactorio",
    "alert_level": "info"
  },
  "model_used": "Ensemble_Voting",
  "model_r2": 0.8350
}
```

#### 4. **POST** `/api/predict/batch`
Realiza predicciones por lotes.

**Request:**
```json
{
  "students": [
    {...},  // Estudiante 1
    {...}   // Estudiante 2
  ]
}
```

---

## 💻 Ejemplos de Uso

### cURL

```bash
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours": 6,
    "attendance": 90,
    "sleep_hours": 8,
    "internet_usage": 3,
    "assignments_completed": 10,
    "previous_score": 75,
    "placement_status": "Placed"
  }'
```

### Python

```python
import requests

url = "https://your-app.onrender.com/api/predict"

data = {
    "study_hours": 6,
    "attendance": 90,
    "sleep_hours": 8,
    "internet_usage": 3,
    "assignments_completed": 10,
    "previous_score": 75,
    "placement_status": "Placed"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicción: {result['prediction']:.2f}")
print(f"Categoría: {result['performance_category']['category']}")
print(f"Intervalo: [{result['confidence_interval']['lower']:.2f}, {result['confidence_interval']['upper']:.2f}]")
```

### JavaScript

```javascript
const data = {
  study_hours: 6,
  attendance: 90,
  sleep_hours: 8,
  internet_usage: 3,
  assignments_completed: 10,
  previous_score: 75,
  placement_status: "Placed"
};

fetch('https://your-app.onrender.com/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(data)
})
.then(res => res.json())
.then(result => {
  console.log('Predicción:', result.prediction);
  console.log('Categoría:', result.performance_category.category);
});
```

---

## 🎨 Interfaz Web

La aplicación incluye una interfaz web moderna construida con **Tailwind CSS**:

### Página Principal (`/`)
- ✅ Formulario interactivo con validación
- ✅ **Selector de modelo ML** - Elige entre SVR, Gradient Boosting o Random Forest
- ✅ Resultados en tiempo real
- ✅ Visualización de intervalos de confianza
- ✅ Clasificación de rendimiento por colores
- ✅ Indicadores de feature engineering automático

### Documentación (`/docs`)
- ✅ Especificación completa de endpoints
- ✅ Ejemplos de código en múltiples lenguajes
- ✅ Tabla de parámetros y respuestas
- ✅ Códigos de error y soluciones

---

## 🧪 Pruebas de Funcionamiento

### 1. Verificar Estado del Servidor

```bash
curl https://your-app.onrender.com/api/health
```

### 2. Prueba de Predicción Individual

```bash
python -c "
import requests
response = requests.post('https://your-app.onrender.com/api/predict', json={
    'study_hours': 6,
    'attendance': 90,
    'sleep_hours': 8,
    'internet_usage': 3,
    'assignments_completed': 10,
    'previous_score': 80,
    'placement_status': 'Placed'
})
print(response.json())
"
```

### 3. Prueba de Predicción por Lotes

```bash
curl -X POST https://your-app.onrender.com/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"students": [{...}, {...}]}'
```

## 🛠️ Tecnologías Utilizadas

### Backend
- **Flask** 2.3+ - Framework web
- **scikit-learn** 1.3+ - Machine Learning
- **pandas** 2.0+ - Manipulación de datos
- **numpy** 1.24+ - Cálculos numéricos
- **gunicorn** 21.2+ - Servidor WSGI
- **scipy** 1.10+ - Optimización científica

### Frontend
- **Tailwind CSS** 3+ - Framework CSS
- **Font Awesome** 6+ - Iconos
- **Highlight.js** - Syntax highlighting

### Deployment
- **Render.com** - Hosting cloud
- **Git/GitHub** - Control de versiones

---

## 🐛 Troubleshooting

### Error: Modelo no cargado

**Síntoma:**
```json
{"error": "Modelo no cargado"}
```

**Solución:**
```bash
# Entrenar el modelo primero
python model_training.py

# Verificar que existe models/student_performance_model.pkl
ls models/
```

### Error: Puerto ocupado

**Síntoma:**
```
Address already in use
```

**Solución:**
```bash
# Cambiar puerto en app.py o usar variable de entorno
export PORT=5001
python app.py
```

### Error: Módulo no encontrado

**Síntoma:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solución:**
```bash
pip install -r requirements.txt
```

### Deploy fallido en Render

**Solución:**
1. Verificar que `requirements.txt`, `Procfile`, `runtime.txt` están en la raíz
2. Verificar que el modelo `.pkl` existe en `models/`
3. Revisar logs en Render Dashboard
4. Asegurar que Python version en `runtime.txt` es compatible

---

## 📈 Mejoras Futuras

- [ ] Autenticación con JWT/API Keys
- [ ] Rate limiting para prevenir abuso
- [ ] Caché de predicciones con Redis
- [ ] Monitoreo con Prometheus/Grafana
- [ ] Reentrenamiento automático periódico
- [ ] Soporte para más formatos de entrada (CSV, Excel)
- [ ] Dashboard de analytics de predicciones
- [ ] Tests unitarios y de integración
- [ ] CI/CD con GitHub Actions
- [ ] Dockerización

---

## 📝 Licencia

Proyecto académico - Universidad del Valle
Maestría en Computación para el Desarrollo de Aplicaciones Inteligentes [CODING]

---

## 👥 Autores

**Nestor Cardona** - 202502968
**Carlos Tovar** - 202503011
**Yolima Guadir** - 202502581

**Asignatura**: Aprendizaje de Máquina Aplicado
**Periodo**: Semestre 2 - 2025

---

## 🔗 Enlaces Útiles

- 📚 [Documentación Flask](https://flask.palletsprojects.com/)
- 🧠 [Scikit-learn Docs](https://scikit-learn.org/stable/)
- ☁️ [Render Docs](https://render.com/docs)
- 🎨 [Tailwind CSS](https://tailwindcss.com/)
- 📊 [Kaggle Dataset](https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset)

---

## 📧 Soporte

Para preguntas o issues:
- Abrir un issue en GitHub
- Contactar a los autores via correo institucional

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**

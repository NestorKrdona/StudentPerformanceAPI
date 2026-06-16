"""
API REST para predicción de rendimiento estudiantil.
Servidor Flask con endpoints documentados y manejo de errores.
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Variables globales para los modelos
model_data = None  # Modelo por defecto (SVR)
available_models = {}  # Diccionario con todos los modelos disponibles


def load_model():
    """Carga los modelos serializados al iniciar la aplicación."""
    global model_data, available_models

    # Modelos disponibles
    models_config = {
        'SVR_Optimized': 'models/student_performance_model.pkl',
        'GradientBoosting_Optimized': 'models/gradient_boosting_model.pkl',
        'RandomForest_Optimized': 'models/random_forest_model.pkl'
    }

    print("="*60)
    print("📦 CARGANDO MODELOS DISPONIBLES")
    print("="*60)

    for model_name, model_path in models_config.items():
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                available_models[model_name] = loaded_model
                print(f"✓ {model_name}")
                print(f"  - R²: {loaded_model['metrics']['r2']:.4f}")
                print(f"  - RMSE: {loaded_model['metrics']['rmse']:.4f}")
            except Exception as e:
                print(f"⚠️  Error cargando {model_name}: {e}")
        else:
            print(f"⚠️  {model_name} no encontrado en {model_path}")

    # Establecer modelo por defecto (SVR)
    if 'SVR_Optimized' in available_models:
        model_data = available_models['SVR_Optimized']
        print(f"\n✓ Modelo por defecto: SVR_Optimized")
    elif available_models:
        # Si SVR no está disponible, usar el primero que encuentre
        default_name = list(available_models.keys())[0]
        model_data = available_models[default_name]
        print(f"\n✓ Modelo por defecto: {default_name}")
    else:
        print("\n⚠️  No se encontraron modelos. Ejecuta los scripts de entrenamiento primero.")
        return None

    print("="*60)

    return model_data


# ============================================================================
# RUTAS HTML
# ============================================================================

@app.route('/')
def index():
    """Página principal con interfaz de predicción."""
    return render_template('index.html')


@app.route('/docs')
def docs():
    """Documentación de la API."""
    return render_template('api_docs.html')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Verifica el estado del servidor y modelo.

    Returns:
        JSON con estado del servicio y modelo
    """
    if model_data is None:
        return jsonify({
            'status': 'error',
            'message': 'Modelo no cargado',
            'timestamp': datetime.now().isoformat()
        }), 503

    return jsonify({
        'status': 'ok',
        'model_name': model_data['model_name'],
        'model_metrics': {
            'r2': round(model_data['metrics']['r2'], 4),
            'rmse': round(model_data['metrics']['rmse'], 4),
            'mae': round(model_data['metrics']['mae'], 4)
        },
        'trained_date': model_data['trained_date'],
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    Obtiene información detallada del modelo.

    Returns:
        JSON con métricas de todos los modelos entrenados
    """
    if model_data is None:
        return jsonify({'error': 'Modelo no cargado'}), 503

    return jsonify({
        'best_model': model_data['model_name'],
        'best_model_metrics': model_data['metrics'],
        'all_models_metrics': model_data.get('all_models_metrics', {}),
        'feature_names': model_data['feature_names'],
        'num_features': len(model_data['feature_names']),
        'trained_date': model_data['trained_date']
    }), 200


@app.route('/api/models/available', methods=['GET'])
def available_models_endpoint():
    """
    Lista todos los modelos disponibles con sus métricas.

    Returns:
        JSON con información de todos los modelos cargados
    """
    if not available_models:
        return jsonify({'error': 'No hay modelos cargados'}), 503

    models_info = {}
    for name, data in available_models.items():
        models_info[name] = {
            'metrics': {
                'r2': round(data['metrics']['r2'], 4),
                'rmse': round(data['metrics']['rmse'], 4),
                'mae': round(data['metrics']['mae'], 4)
            },
            'trained_date': data['trained_date']
        }

    return jsonify({
        'available_models': models_info,
        'default_model': model_data['model_name'] if model_data else None
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Realiza una predicción de rendimiento estudiantil.

    Request Body (JSON):
        {
            "study_hours": float (horas de estudio diarias),
            "attendance": float (0-100, porcentaje de asistencia),
            "sleep_hours": float (horas de sueño),
            "internet_usage": float (horas de uso de internet),
            "assignments_completed": int (tareas completadas),
            "previous_score": float (0-100, calificación anterior),
            "placement_status": str ("Placed", "Not Placed", opcional),
            "model_type": str ("SVR_Optimized", "GradientBoosting_Optimized", "RandomForest_Optimized", opcional)
        }

        Nota: Las siguientes features se calculan automáticamente:
        - cognitive_efficiency = study_hours / sleep_hours
        - academic_engagement = (assignments_completed × attendance) / 100
        - previous_score_quartile = Cuartil basado en previous_score

    Returns:
        JSON con predicción y métricas de confianza
    """
    if model_data is None and not available_models:
        return jsonify({'error': 'Modelo no cargado'}), 503

    try:
        # Validar entrada
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400

        # Seleccionar modelo a usar
        model_type = data.get('model_type', 'SVR_Optimized')

        if model_type in available_models:
            selected_model = available_models[model_type]
        elif model_data:
            selected_model = model_data
            model_type = model_data['model_name']
        else:
            return jsonify({'error': 'Modelo solicitado no disponible'}), 400

        # Crear DataFrame con las features esperadas
        # NOTA: Esto debe coincidir con las columnas del dataset después del encoding
        input_data = prepare_input_features(data)

        # Normalizar
        input_normalized = selected_model['scaler_X'].transform(input_data)

        # Predecir
        prediction_normalized = selected_model['model'].predict(input_normalized)

        # Desnormalizar
        prediction = selected_model['scaler_y'].inverse_transform(
            prediction_normalized.reshape(-1, 1)
        )[0][0]

        # Calcular intervalo de confianza basado en RMSE
        rmse = selected_model['metrics']['rmse']
        confidence_interval = {
            'lower': max(0, prediction - 1.96 * rmse),
            'upper': min(100, prediction + 1.96 * rmse)
        }

        # Clasificar rendimiento
        performance_category = classify_performance(prediction)

        return jsonify({
            'prediction': round(prediction, 2),
            'confidence_interval': {
                'lower': round(confidence_interval['lower'], 2),
                'upper': round(confidence_interval['upper'], 2)
            },
            'performance_category': performance_category,
            'model_used': selected_model['model_name'],
            'model_r2': round(selected_model['metrics']['r2'], 4),
            'timestamp': datetime.now().isoformat()
        }), 200

    except KeyError as e:
        return jsonify({
            'error': f'Campo requerido faltante: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Error en predicción: {str(e)}'
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Realiza predicciones múltiples en batch.

    Request Body (JSON):
        {
            "students": [
                {...},  // Objeto con datos de estudiante 1
                {...}   // Objeto con datos de estudiante 2
            ]
        }

    Returns:
        JSON con array de predicciones
    """
    if model_data is None:
        return jsonify({'error': 'Modelo no cargado'}), 503

    try:
        data = request.get_json()

        if 'students' not in data:
            return jsonify({'error': 'Campo "students" requerido'}), 400

        students = data['students']

        if not isinstance(students, list):
            return jsonify({'error': '"students" debe ser un array'}), 400

        predictions = []

        for idx, student in enumerate(students):
            try:
                input_data = prepare_input_features(student)
                input_normalized = model_data['scaler_X'].transform(input_data)
                prediction_normalized = model_data['model'].predict(input_normalized)
                prediction = model_data['scaler_y'].inverse_transform(
                    prediction_normalized.reshape(-1, 1)
                )[0][0]

                predictions.append({
                    'student_id': idx,
                    'prediction': round(prediction, 2),
                    'performance_category': classify_performance(prediction)
                })

            except Exception as e:
                predictions.append({
                    'student_id': idx,
                    'error': str(e)
                })

        return jsonify({
            'predictions': predictions,
            'total': len(students),
            'successful': len([p for p in predictions if 'error' not in p]),
            'model_used': model_data['model_name'],
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def prepare_input_features(data):
    """
    Prepara las features de entrada para el modelo.
    Debe coincidir con el preprocesamiento del entrenamiento.
    Incluye feature engineering.
    """
    # Features base
    study_hours = data.get('study_hours', 0)
    sleep_hours = data.get('sleep_hours', 1)  # Evitar división por 0
    attendance = data.get('attendance', 0)
    assignments_completed = data.get('assignments_completed', 0)
    previous_score = data.get('previous_score', 0)
    internet_usage = data.get('internet_usage', 0)

    # ===== FEATURE ENGINEERING =====
    # 1. Eficiencia cognitiva
    cognitive_efficiency = study_hours / (sleep_hours + 1e-5)

    # 2. Compromiso académico
    academic_engagement = (assignments_completed * attendance) / 100

    # 3. Cuartil de previous_score
    # Determinar cuartil basado en rangos típicos
    if previous_score < 50:
        prev_q1, prev_q2, prev_q3, prev_q4 = 1, 0, 0, 0
    elif previous_score < 65:
        prev_q1, prev_q2, prev_q3, prev_q4 = 0, 1, 0, 0
    elif previous_score < 80:
        prev_q1, prev_q2, prev_q3, prev_q4 = 0, 0, 1, 0
    else:
        prev_q1, prev_q2, prev_q3, prev_q4 = 0, 0, 0, 1

    features = {
        # Features originales
        'study_hours': study_hours,
        'attendance': attendance,
        'sleep_hours': sleep_hours,
        'internet_usage': internet_usage,
        'assignments_completed': assignments_completed,
        'previous_score': previous_score,

        # Features engineered
        'cognitive_efficiency': cognitive_efficiency,
        'academic_engagement': academic_engagement,

        # Previous score quartiles
        'prev_score_Q1_Low': prev_q1,
        'prev_score_Q2_Medium': prev_q2,
        'prev_score_Q3_Good': prev_q3,
        'prev_score_Q4_Excellent': prev_q4,

        # Placement status encoding (si existe en el dataset)
        'placement_status_Placed': 1 if data.get('placement_status') == 'Placed' else 0,
        'placement_status_Not Placed': 1 if data.get('placement_status') == 'Not Placed' else 0,
    }

    df = pd.DataFrame([features])

    # Asegurar que las columnas coincidan con las del modelo
    expected_features = model_data['feature_names']

    # Agregar columnas faltantes con 0
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Ordenar columnas según el modelo
    df = df[expected_features]

    return df


def classify_performance(score):
    """Clasifica el rendimiento en categorías."""
    if score >= 85:
        return {
            'category': 'Excelente',
            'description': 'Rendimiento sobresaliente',
            'alert_level': 'success'
        }
    elif score >= 70:
        return {
            'category': 'Bueno',
            'description': 'Rendimiento satisfactorio',
            'alert_level': 'info'
        }
    elif score >= 55:
        return {
            'category': 'Regular',
            'description': 'Requiere atención',
            'alert_level': 'warning'
        }
    else:
        return {
            'category': 'En Riesgo',
            'description': 'Intervención urgente necesaria',
            'alert_level': 'danger'
        }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500


# ============================================================================
# INICIALIZACIÓN
# ============================================================================

# Cargar modelos al iniciar (se ejecuta siempre, incluso con Gunicorn)
print("="*80)
print("🚀 INICIANDO STUDENT PERFORMANCE API")
print("="*80)
load_model()
print("="*80 + "\n")

if __name__ == '__main__':
    # Configuración del servidor (solo para desarrollo)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False') == 'True'

    print(f"\n📡 Servidor corriendo en puerto {port}")
    print(f"🌐 Interfaz web: http://127.0.0.1:{port}/")
    print(f"📚 Documentación API: http://127.0.0.1:{port}/docs")
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)

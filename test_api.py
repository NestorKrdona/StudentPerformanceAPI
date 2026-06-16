"""
Script de prueba para verificar el funcionamiento de la API
con feature engineering implementado.
"""

import requests
import json

# Configuración
BASE_URL = "http://127.0.0.1:5000"

def test_health():
    """Prueba el endpoint de health check."""
    print("="*80)
    print("TEST 1: Health Check")
    print("="*80)

    response = requests.get(f"{BASE_URL}/api/health")
    data = response.json()

    print(f"Status Code: {response.status_code}")
    print(json.dumps(data, indent=2))
    print()


def test_model_info():
    """Prueba el endpoint de información del modelo."""
    print("="*80)
    print("TEST 2: Model Info")
    print("="*80)

    response = requests.get(f"{BASE_URL}/api/model/info")
    data = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Mejor Modelo: {data['best_model']}")
    print(f"Número de Features: {data['num_features']}")
    print(f"R² Score: {data['best_model_metrics']['r2']:.4f}")
    print()


def test_predict_individual():
    """Prueba predicción individual con feature engineering."""
    print("="*80)
    print("TEST 3: Predicción Individual")
    print("="*80)

    # Datos de prueba
    student_data = {
        "study_hours": 6,
        "attendance": 90,
        "sleep_hours": 8,
        "internet_usage": 3,
        "assignments_completed": 10,
        "previous_score": 75,
        "placement_status": "Placed"
    }

    print("Datos del estudiante:")
    print(json.dumps(student_data, indent=2))
    print()

    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=student_data
    )
    data = response.json()

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        print(f"\n📊 RESULTADOS:")
        print(f"  Predicción: {data['prediction']:.2f} puntos")
        print(f"  Categoría: {data['performance_category']['category']}")
        print(f"  Descripción: {data['performance_category']['description']}")
        print(f"  Intervalo de Confianza: [{data['confidence_interval']['lower']:.2f}, {data['confidence_interval']['upper']:.2f}]")
        print(f"  Modelo Usado: {data['model_used']}")
        print(f"  R² del Modelo: {data['model_r2']:.4f}")
    else:
        print(f"Error: {data.get('error', 'Unknown error')}")

    print()


def test_predict_batch():
    """Prueba predicción por lotes."""
    print("="*80)
    print("TEST 4: Predicción por Lotes")
    print("="*80)

    students = {
        "students": [
            {
                "study_hours": 8,
                "attendance": 95,
                "sleep_hours": 7,
                "internet_usage": 2,
                "assignments_completed": 10,
                "previous_score": 85,
                "placement_status": "Placed"
            },
            {
                "study_hours": 3,
                "attendance": 60,
                "sleep_hours": 5,
                "internet_usage": 8,
                "assignments_completed": 4,
                "previous_score": 45,
                "placement_status": "Not Placed"
            },
            {
                "study_hours": 5,
                "attendance": 80,
                "sleep_hours": 7,
                "internet_usage": 4,
                "assignments_completed": 8,
                "previous_score": 70
            }
        ]
    }

    response = requests.post(
        f"{BASE_URL}/api/predict/batch",
        json=students
    )
    data = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Total de estudiantes: {data['total']}")
    print(f"Predicciones exitosas: {data['successful']}")
    print()

    for prediction in data['predictions']:
        if 'error' not in prediction:
            print(f"Estudiante {prediction['student_id']}:")
            print(f"  Predicción: {prediction['prediction']:.2f}")
            print(f"  Categoría: {prediction['performance_category']['category']}")
        else:
            print(f"Estudiante {prediction['student_id']}: ERROR - {prediction['error']}")

    print()


def test_available_models():
    """Prueba el endpoint de modelos disponibles."""
    print("="*80)
    print("TEST 5: Modelos Disponibles")
    print("="*80)

    response = requests.get(f"{BASE_URL}/api/models/available")
    data = response.json()

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        print(f"\n📦 Modelos Disponibles:")
        print(f"  Modelo por defecto: {data['default_model']}")
        print(f"\n  Modelos cargados:")
        for model_name, model_info in data['available_models'].items():
            metrics = model_info['metrics']
            print(f"    • {model_name}")
            print(f"      - R²: {metrics['r2']:.4f}")
            print(f"      - RMSE: {metrics['rmse']:.4f}")
            print(f"      - MAE: {metrics['mae']:.4f}")
    else:
        print(f"Error: {data.get('error', 'Unknown error')}")

    print()


def test_model_selection():
    """Prueba la selección de diferentes modelos."""
    print("="*80)
    print("TEST 6: Selección de Modelos")
    print("="*80)

    # Datos de prueba estándar
    student_data = {
        "study_hours": 6,
        "attendance": 85,
        "sleep_hours": 7,
        "internet_usage": 4,
        "assignments_completed": 8,
        "previous_score": 72
    }

    models_to_test = [
        "SVR_Optimized",
        "GradientBoosting_Optimized",
        "RandomForest_Optimized"
    ]

    print("Probando el mismo estudiante con diferentes modelos:\n")
    print(f"Datos: study_hours={student_data['study_hours']}, attendance={student_data['attendance']}, previous_score={student_data['previous_score']}\n")

    results = []

    for model_type in models_to_test:
        test_data = student_data.copy()
        test_data['model_type'] = model_type

        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=test_data
        )

        if response.status_code == 200:
            data = response.json()
            results.append({
                'model': model_type,
                'prediction': data['prediction'],
                'r2': data['model_r2'],
                'category': data['performance_category']['category']
            })
            print(f"✅ {model_type}:")
            print(f"   Predicción: {data['prediction']:.2f}")
            print(f"   Categoría: {data['performance_category']['category']}")
            print(f"   R² del modelo: {data['model_r2']:.4f}")
        else:
            print(f"⚠️  {model_type}: No disponible")

    print("\n📊 Comparación de Predicciones:")
    if results:
        predictions = [r['prediction'] for r in results]
        print(f"   Rango: {min(predictions):.2f} - {max(predictions):.2f}")
        print(f"   Promedio: {sum(predictions)/len(predictions):.2f}")

    print()


def test_feature_engineering():
    """Verifica que el feature engineering esté funcionando."""
    print("="*80)
    print("TEST 7: Verificación de Feature Engineering")
    print("="*80)

    # Caso de prueba con valores específicos
    test_case = {
        "study_hours": 8,        # Estudio alto
        "attendance": 100,       # Asistencia perfecta
        "sleep_hours": 8,        # Sueño adecuado
        "internet_usage": 2,     # Uso internet moderado
        "assignments_completed": 10,  # Todas las tareas
        "previous_score": 90,    # Cuartil Q4 (Excellent)
    }

    # Calcular features esperados manualmente
    expected_cognitive_efficiency = test_case['study_hours'] / test_case['sleep_hours']
    expected_academic_engagement = (test_case['assignments_completed'] * test_case['attendance']) / 100

    print("Datos de entrada:")
    print(json.dumps(test_case, indent=2))
    print()

    print("Features Calculados Esperados:")
    print(f"  Cognitive Efficiency: {expected_cognitive_efficiency:.4f}")
    print(f"  Academic Engagement: {expected_academic_engagement:.2f}")
    print(f"  Previous Score Quartile: Q4_Excellent (score >= 80)")
    print()

    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=test_case
    )
    data = response.json()

    if response.status_code == 200:
        print("✅ Predicción exitosa")
        print(f"  Predicción: {data['prediction']:.2f}")
        print(f"  Categoría: {data['performance_category']['category']}")
        print()
        print("📈 Análisis:")
        print(f"  - Eficiencia cognitiva de {expected_cognitive_efficiency:.2f} (estudio/sueño equilibrado)")
        print(f"  - Compromiso académico de {expected_academic_engagement:.0f}% (excelente)")
        print(f"  - Rendimiento previo en cuartil superior (Q4)")
        print(f"  → Predicción esperada: ALTA (>80)")
        print(f"  → Predicción obtenida: {data['prediction']:.2f}")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown')}")

    print()


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*80)
    print("SUITE DE PRUEBAS - STUDENT PERFORMANCE API")
    print("="*80 + "\n")

    try:
        test_health()
        test_model_info()
        test_predict_individual()
        test_predict_batch()
        test_available_models()
        test_model_selection()
        test_feature_engineering()

        print("="*80)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*80)

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar al servidor")
        print("Asegúrate de que el servidor esté corriendo en http://127.0.0.1:5000")
        print("Ejecuta: python app.py")
    except Exception as e:
        print(f"❌ ERROR INESPERADO: {e}")


if __name__ == "__main__":
    run_all_tests()

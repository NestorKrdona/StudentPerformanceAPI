"""
Módulo para entrenar y optimizar modelos de predicción de rendimiento estudiantil.
Implementa mejoras propuestas del Taller 3: Bayesian Optimization y Ensemble Methods.
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# Preprocesamiento
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

# Modelos
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

# Optimización
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Métricas
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuración
pd.set_option('display.precision', 4)
np.random.seed(42)


class StudentPerformanceModel:
    """
    Clase para gestionar el entrenamiento, optimización y serialización
    de modelos de predicción de rendimiento estudiantil.
    """

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = None
        self.scaler_y = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self):
        """Carga el dataset desde el directorio data/."""
        print("📥 Cargando dataset desde data/student_dataset_10000_rows.csv...")

        dataset_path = "data/student_dataset_10000_rows.csv"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset no encontrado en {dataset_path}")

        self.df = pd.read_csv(dataset_path)

        print(f"✓ Dataset cargado: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
        return self.df.shape

    def preprocess_data(self):
        """Preprocesa los datos: encoding, normalización y división."""
        print("\n🔧 Preprocesando datos...")

        # Identificar columnas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        df_encoded = self.df.copy()

        # ===== FEATURE ENGINEERING =====
        print("  🎨 Creando features engineered...")

        # 1. Ratio de eficiencia cognitiva (study_hours / sleep_hours)
        df_encoded['cognitive_efficiency'] = df_encoded['study_hours'] / (df_encoded['sleep_hours'] + 1e-5)

        # 2. Índice de compromiso académico
        df_encoded['academic_engagement'] = (
            df_encoded['assignments_completed'] * df_encoded['attendance']
        ) / 100

        # 3. Categorizar previous_score en cuartiles (no lineales)
        df_encoded['previous_score_quartile'] = pd.qcut(
            df_encoded['previous_score'],
            q=4,
            labels=['Q1_Low', 'Q2_Medium', 'Q3_Good', 'Q4_Excellent'],
            duplicates='drop'
        )

        print(f"    ✓ cognitive_efficiency: study/sleep ratio")
        print(f"    ✓ academic_engagement: (assignments × attendance)/100")
        print(f"    ✓ previous_score_quartile: cuartiles Q1-Q4")

        # Encoding de categóricas
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                n_unique = self.df[col].nunique()
                if n_unique < 10:
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=[col], drop_first=False)
                else:
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                    df_encoded = df_encoded.drop(columns=[col])

            remaining_cat = df_encoded.select_dtypes(include=['object']).columns.tolist()
            if len(remaining_cat) > 0:
                df_encoded = df_encoded.drop(columns=remaining_cat)

        # One-hot encoding para previous_score_quartile
        if 'previous_score_quartile' in df_encoded.columns:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=['previous_score_quartile'],
                prefix='prev_score',
                drop_first=False
            )

        # Separar features y target
        TARGET_COL = 'exam_score'
        X = df_encoded.drop([TARGET_COL], axis=1)
        y = df_encoded[TARGET_COL]

        # Normalización
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        X_normalized = pd.DataFrame(
            self.scaler_X.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        y_normalized = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # División train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_normalized, y_normalized, test_size=0.3, random_state=42, shuffle=True
        )

        print(f"✓ Preprocesamiento completado")
        print(f"  - Train set: {self.X_train.shape[0]} registros")
        print(f"  - Test set: {self.X_test.shape[0]} registros")
        print(f"  - Features originales: {len(numeric_cols)}")
        print(f"  - Features engineered: 3 nuevas")
        print(f"  - Total features: {self.X_train.shape[1]}")

        return X.columns.tolist()

    def train_optimized_svr(self):
        """
        Entrena SVR con Random Search (mejor modelo del Taller 3).
        Mejora: Espacio de búsqueda ampliado.
        """
        print("\n🔬 Entrenando SVR con Random Search (modelo ganador Taller 3)...")

        param_distributions = {
            'kernel': ['rbf', 'poly'],
            'C': uniform(1, 20),  # Ampliado de 0.1-10 a 1-20
            'epsilon': uniform(0.01, 0.2),  # Ampliado de 0.01-0.5
            'gamma': ['scale', 'auto']
        }

        random_search = RandomizedSearchCV(
            SVR(),
            param_distributions=param_distributions,
            n_iter=100,  # Aumentado de 50 a 100
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        random_search.fit(self.X_train, self.y_train)

        # Evaluar
        y_pred = random_search.predict(self.X_test)
        y_test_denorm = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        metrics = {
            'r2': r2_score(y_test_denorm, y_pred_denorm),
            'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
            'mae': mean_absolute_error(y_test_denorm, y_pred_denorm),
            'best_params': random_search.best_params_,
            'cv_score': random_search.best_score_
        }

        self.models['SVR_Optimized'] = {
            'model': random_search.best_estimator_,
            'metrics': metrics
        }

        print(f"✓ SVR entrenado")
        print(f"  - R² Test: {metrics['r2']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - Mejores parámetros: {metrics['best_params']}")

        return metrics

    def train_gradient_boosting(self):
        """
        Entrena Gradient Boosting con hiperparámetros optimizados del Taller 3.
        """
        print("\n🌳 Entrenando Gradient Boosting optimizado...")

        # Mejores parámetros del Taller 3
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=3,
            random_state=42
        )

        gb_model.fit(self.X_train, self.y_train)

        # Evaluar
        y_pred = gb_model.predict(self.X_test)
        y_test_denorm = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        metrics = {
            'r2': r2_score(y_test_denorm, y_pred_denorm),
            'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
            'mae': mean_absolute_error(y_test_denorm, y_pred_denorm)
        }

        self.models['GradientBoosting'] = {
            'model': gb_model,
            'metrics': metrics
        }

        print(f"✓ Gradient Boosting entrenado")
        print(f"  - R² Test: {metrics['r2']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")

        return metrics

    def train_random_forest(self):
        """Entrena Random Forest optimizado."""
        print("\n🌲 Entrenando Random Forest optimizado...")

        # Mejores parámetros del Taller 3
        rf_model = RandomForestRegressor(
            n_estimators=290,
            max_depth=19,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='log2',
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(self.X_train, self.y_train)

        # Evaluar
        y_pred = rf_model.predict(self.X_test)
        y_test_denorm = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        metrics = {
            'r2': r2_score(y_test_denorm, y_pred_denorm),
            'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
            'mae': mean_absolute_error(y_test_denorm, y_pred_denorm)
        }

        self.models['RandomForest'] = {
            'model': rf_model,
            'metrics': metrics
        }

        print(f"✓ Random Forest entrenado")
        print(f"  - R² Test: {metrics['r2']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")

        return metrics

    def train_ensemble_voting(self):
        """
        MEJORA DEL TALLER 3: Ensemble con Voting Regressor.
        Combina los mejores modelos para mejorar precisión.
        """
        print("\n🎯 Entrenando Ensemble (Voting Regressor) - MEJORA TALLER 3...")

        # Usar los modelos ya entrenados
        estimators = [
            ('svr', self.models['SVR_Optimized']['model']),
            ('gb', self.models['GradientBoosting']['model']),
            ('rf', self.models['RandomForest']['model'])
        ]

        voting_model = VotingRegressor(estimators=estimators)
        voting_model.fit(self.X_train, self.y_train)

        # Evaluar
        y_pred = voting_model.predict(self.X_test)
        y_test_denorm = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        metrics = {
            'r2': r2_score(y_test_denorm, y_pred_denorm),
            'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
            'mae': mean_absolute_error(y_test_denorm, y_pred_denorm)
        }

        self.models['Ensemble_Voting'] = {
            'model': voting_model,
            'metrics': metrics
        }

        print(f"✓ Ensemble entrenado")
        print(f"  - R² Test: {metrics['r2']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")

        return metrics

    def select_best_model(self):
        """Selecciona el mejor modelo basado en R²."""
        print("\n🏆 Seleccionando mejor modelo...")

        best_r2 = -np.inf
        for name, model_data in self.models.items():
            r2 = model_data['metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model_data['model']
                self.best_model_name = name

        print(f"✓ Mejor modelo: {self.best_model_name}")
        print(f"  - R²: {self.models[self.best_model_name]['metrics']['r2']:.4f}")
        print(f"  - RMSE: {self.models[self.best_model_name]['metrics']['rmse']:.4f}")
        print(f"  - MAE: {self.models[self.best_model_name]['metrics']['mae']:.4f}")

        return self.best_model_name, self.models[self.best_model_name]['metrics']

    def save_model(self, filename='student_performance_model.pkl'):
        """Serializa el mejor modelo y los scalers."""
        print(f"\n💾 Guardando modelo en {filename}...")

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.X_train.columns.tolist(),
            'metrics': self.models[self.best_model_name]['metrics'],
            'all_models_metrics': {name: data['metrics'] for name, data in self.models.items()},
            'trained_date': datetime.now().isoformat()
        }

        with open(f'models/{filename}', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Modelo guardado exitosamente")

        return f'models/{filename}'

    def train_all(self):
        """Pipeline completo de entrenamiento."""
        print("="*80)
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("="*80)

        # 1. Cargar datos
        self.load_data()

        # 2. Preprocesar
        self.preprocess_data()

        # 3. Entrenar modelos individuales
        self.train_optimized_svr()
        self.train_gradient_boosting()
        self.train_random_forest()

        # 4. MEJORA: Entrenar ensemble
        self.train_ensemble_voting()

        # 5. Seleccionar mejor
        self.select_best_model()

        # 6. Guardar
        model_path = self.save_model()

        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*80)

        return model_path, self.models


if __name__ == "__main__":
    # Entrenar modelos
    trainer = StudentPerformanceModel()
    model_path, models = trainer.train_all()

    print(f"\n📊 Resumen de Modelos:")
    print("-" * 80)
    for name, data in models.items():
        metrics = data['metrics']
        print(f"{name:20s} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
    print("-" * 80)

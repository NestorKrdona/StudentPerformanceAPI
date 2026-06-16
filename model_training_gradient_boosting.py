"""
Módulo para entrenar modelo Gradient Boosting de predicción de rendimiento estudiantil.
Modelo alternativo 1: Gradient Boosting Optimizado.
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
from sklearn.ensemble import GradientBoostingRegressor

# Optimización
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Métricas
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuración
pd.set_option('display.precision', 4)
np.random.seed(42)


class GradientBoostingModel:
    """
    Clase para entrenar modelo Gradient Boosting optimizado.
    """

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.metrics = {}

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
        print(f"  - Total features: {self.X_train.shape[1]}")

        return X.columns.tolist()

    def train_model(self):
        """
        Entrena Gradient Boosting con Random Search para optimización.
        """
        print("\n🌳 Entrenando Gradient Boosting con Random Search...")

        param_distributions = {
            'n_estimators': randint(100, 300),
            'learning_rate': uniform(0.05, 0.25),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'subsample': uniform(0.7, 0.3),
            'max_features': ['sqrt', 'log2', None]
        }

        random_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(self.X_train, self.y_train)

        # Guardar mejor modelo
        self.model = random_search.best_estimator_

        # Evaluar
        y_pred = self.model.predict(self.X_test)
        y_test_denorm = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        self.metrics = {
            'r2': r2_score(y_test_denorm, y_pred_denorm),
            'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
            'mae': mean_absolute_error(y_test_denorm, y_pred_denorm),
            'best_params': random_search.best_params_,
            'cv_score': random_search.best_score_
        }

        print(f"✓ Gradient Boosting entrenado")
        print(f"  - R² Test: {self.metrics['r2']:.4f}")
        print(f"  - RMSE: {self.metrics['rmse']:.4f}")
        print(f"  - MAE: {self.metrics['mae']:.4f}")
        print(f"  - Mejores parámetros: {self.metrics['best_params']}")

        return self.metrics

    def save_model(self, filename='gradient_boosting_model.pkl'):
        """Serializa el modelo y los scalers."""
        print(f"\n💾 Guardando modelo en {filename}...")

        model_data = {
            'model': self.model,
            'model_name': 'GradientBoosting_Optimized',
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.X_train.columns.tolist(),
            'metrics': self.metrics,
            'trained_date': datetime.now().isoformat()
        }

        # Crear directorio models si no existe
        os.makedirs('models', exist_ok=True)

        with open(f'models/{filename}', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Modelo guardado exitosamente")

        return f'models/{filename}'

    def train_all(self):
        """Pipeline completo de entrenamiento."""
        print("="*80)
        print("🚀 ENTRENAMIENTO GRADIENT BOOSTING MODEL")
        print("="*80)

        # 1. Cargar datos
        self.load_data()

        # 2. Preprocesar
        self.preprocess_data()

        # 3. Entrenar modelo
        self.train_model()

        # 4. Guardar
        model_path = self.save_model()

        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*80)

        return model_path, self.metrics


if __name__ == "__main__":
    # Entrenar modelo
    trainer = GradientBoostingModel()
    model_path, metrics = trainer.train_all()

    print(f"\n📊 Métricas del Modelo:")
    print("-" * 80)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print("-" * 80)

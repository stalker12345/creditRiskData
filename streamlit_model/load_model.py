# Скрипт для загрузки модели в Streamlit

import skops.io as sio
import json
import os
import pandas as pd
import numpy as np
import joblib
from scipy import stats

def load_model():
    """Загружает модель"""
    model_path = os.path.join(os.path.dirname(__file__), "model.skops")
    unknown_types = sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=unknown_types)

def load_metadata():
    """Загружает метаданные"""
    with open(os.path.join(os.path.dirname(__file__), "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_metrics():
    """Загружает метрики"""
    with open(os.path.join(os.path.dirname(__file__), "model_metrics.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_boxcox_params():
    """Загружает параметры Box-Cox"""
    lambdas_path = os.path.join(os.path.dirname(__file__), "lambdas.pkl")
    return joblib.load(lambdas_path) if os.path.exists(lambdas_path) else None

def apply_boxcox_transform(df, lambdas_dict):
    """Применяет Box-Cox преобразования"""
    if lambdas_dict is None:
        return df.copy()

    df_transformed = df.copy()
    features_for_boxcox = [
        "person_age", "person_emp_length", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
    ]

    for col in features_for_boxcox:
        if col not in df_transformed.columns or col not in lambdas_dict:
            continue
        lambda_param, shift = lambdas_dict[col]
        col_data = df_transformed[col] + shift if shift > 0 else df_transformed[col]
        try:
            df_transformed[col] = stats.boxcox(col_data, lmbda=lambda_param)
        except Exception:
            pass
    return df_transformed

def validate_input_data(df_input, metadata):
    """Проверяет обязательные поля и базовую корректность входных данных."""
    required = metadata["all_features"]
    missing = [feature for feature in required if feature not in df_input.columns]
    if missing:
        raise ValueError(f"Отсутствуют признаки: {missing}")

    if 'person_income' in df_input.columns and (df_input['person_income'] <= 0).any():
        raise ValueError("Признак person_income должен быть больше 0.")

    categorical_constraints = {
        'person_home_ownership': metadata.get('person_home_ownership_categories', []),
        'loan_intent': metadata.get('loan_intent_categories', []),
        'loan_grade': metadata.get('grade_order', metadata.get('loan_grade_categories', [])),
    }

    for column, allowed_values in categorical_constraints.items():
        if column not in df_input.columns or not allowed_values:
            continue

        invalid_values = sorted(set(df_input[column].dropna()) - set(allowed_values))
        if invalid_values:
            raise ValueError(
                f"Недопустимые значения для {column}: {invalid_values}. "
                f"Допустимые значения: {allowed_values}"
            )

def predict_risk(user_data, model, metadata, lambdas_dict=None):
    """
    Делает предсказание риска дефолта.
    
    Параметры:
    -----------
    user_data : dict or pd.DataFrame
        Входные данные в исходном формате (до преобразований):
        - person_income: исходное значение в долларах (например, 50000), НЕ логарифмированное
        - Остальные числовые признаки: исходные значения
        - Категориальные признаки: строки ('RENT', 'EDUCATION', 'B' и т.д.)
    model : sklearn Pipeline
        Обученная модель
    metadata : dict
        Метаданные о признаках
    lambdas_dict : dict, optional
        Параметры Box-Cox преобразований
        
    Возвращает:
    -----------
    dict : Результаты предсказания
    """
    df_input = pd.DataFrame([user_data]) if isinstance(user_data, dict) else user_data.copy()

    validate_input_data(df_input, metadata)
    
    # ШАГ 1: Логарифмирование person_income (применяется ДО Box-Cox)
    # Модель была обучена на логарифмированных значениях
    if 'person_income' in df_input.columns:
        df_input['person_income'] = np.log(df_input['person_income'])
    
    # ШАГ 2: Box-Cox преобразования для 6 признаков
    if lambdas_dict:
        df_input = apply_boxcox_transform(df_input, lambdas_dict)
    
    # ШАГ 3: Pipeline автоматически применяет:
    # - RobustScaler для всех числовых признаков
    # - OneHotEncoder для person_home_ownership и loan_intent
    # - OrdinalEncoder для loan_grade
    
    # ШАГ 4: Предсказание модели
    probabilities = model.predict_proba(df_input)[0]
    class_to_probability = dict(zip(model.classes_, probabilities))
    probability_default = float(class_to_probability[1])
    probability_non_default = float(class_to_probability[0])
    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    prediction = int(probability_default >= decision_threshold)
    
    return {
        "prediction": int(prediction),
        "prediction_class": metadata["class_names"][str(prediction)],
        "probability_default": probability_default,
        "probability_non_default": probability_non_default,
        "risk_level": "Высокий риск" if probability_default >= decision_threshold else "Низкий риск"
    }

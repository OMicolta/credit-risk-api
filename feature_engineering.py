"""
Feature Engineering para Predicción
====================================
Replica las transformaciones aplicadas durante el entrenamiento.
"""

import numpy as np
import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Feature Engineering a los datos de entrada.
    Debe replicar EXACTAMENTE lo que se hizo en entrenamiento.
    """
    df = df.copy()
    
    # ============================================
    # 1. RATIOS FINANCIEROS
    # ============================================
    
    # Ratio crédito / ingreso
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    # Ratio anualidad / ingreso
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    # Ratio crédito / anualidad
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    
    # Ratio precio bienes / crédito
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    
    # Diferencia crédito - precio bienes
    df['CREDIT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    
    # Ingreso per cápita familiar
    df['INCOME_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # Crédito per cápita
    df['CREDIT_PER_FAMILY'] = df['AMT_CREDIT'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # ============================================
    # 2. FEATURES DE EDAD Y TIEMPO
    # ============================================
    
    # Edad en años
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    
    # Años empleado
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365
    df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].apply(lambda x: x if x < 50 else 0)
    
    # Años desde registro
    df['YEARS_REGISTRATION'] = -df['DAYS_REGISTRATION'] / 365
    
    # Años desde cambio de ID
    df['YEARS_ID_PUBLISH'] = -df['DAYS_ID_PUBLISH'] / 365
    
    # Ratio empleo / edad
    df['EMPLOYED_AGE_RATIO'] = df['YEARS_EMPLOYED'] / (df['AGE_YEARS'] + 1)
    
    # Edad al comenzar empleo
    df['AGE_AT_EMPLOYMENT'] = df['AGE_YEARS'] - df['YEARS_EMPLOYED']
    
    # ============================================
    # 3. FEATURES EXT_SOURCE (MUY IMPORTANTES)
    # ============================================
    
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    # Rellenar nulos con 0.5 (valor medio)
    for col in ext_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.5)
        else:
            df[col] = 0.5
    
    # Estadísticas de EXT_SOURCE
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_SUM'] = df[ext_cols].sum(axis=1)
    
    # Combinaciones de EXT_SOURCE
    df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # EXT_SOURCE ponderado por crédito
    df['EXT_CREDIT_WEIGHTED'] = df['EXT_SOURCE_MEAN'] * df['AMT_CREDIT']
    
    # ============================================
    # 4. FEATURES DE REGIÓN
    # ============================================
    
    df['REGION_SCORE'] = df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']
    
    # ============================================
    # 5. FEATURES DE CÍRCULO SOCIAL
    # ============================================
    
    df['SOCIAL_CIRCLE_DEFAULT'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] + df['DEF_60_CNT_SOCIAL_CIRCLE']
    df['SOCIAL_CIRCLE_OBS'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['OBS_60_CNT_SOCIAL_CIRCLE']
    df['SOCIAL_DEFAULT_RATIO'] = df['SOCIAL_CIRCLE_DEFAULT'] / (df['SOCIAL_CIRCLE_OBS'] + 1)
    
    # ============================================
    # 6. FEATURES BINARIAS DERIVADAS
    # ============================================
    
    # Tiene carro y propiedad
    has_car = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0}) if 'FLAG_OWN_CAR' in df.columns else 0
    has_realty = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0}) if 'FLAG_OWN_REALTY' in df.columns else 0
    df['HAS_CAR_REALTY'] = (has_car & has_realty).astype(int)
    
    # Edad mayor a 40
    df['IS_OVER_40'] = (df['AGE_YEARS'] > 40).astype(int)
    
    # Empleo estable (más de 3 años)
    df['STABLE_EMPLOYMENT'] = (df['YEARS_EMPLOYED'] > 3).astype(int)
    
    # Alto crédito (usamos un umbral fijo basado en la mediana del entrenamiento)
    CREDIT_MEDIAN = 500000  # Ajustar según tu dataset
    df['HIGH_CREDIT'] = (df['AMT_CREDIT'] > CREDIT_MEDIAN).astype(int)
    
    # ============================================
    # 7. INTERACCIONES IMPORTANTES
    # ============================================
    
    df['INCOME_EXT_INTERACTION'] = df['AMT_INCOME_TOTAL'] * df['EXT_SOURCE_MEAN']
    df['AGE_STABILITY_INTERACTION'] = df['AGE_YEARS'] * df['EMPLOYED_AGE_RATIO']
    df['CREDIT_SCORE_INTERACTION'] = df['AMT_CREDIT'] * df['EXT_SOURCE_MEAN']
    
    # ============================================
    # 8. TRANSFORMAR CATEGÓRICAS
    # ============================================
    
    # FLAG_OWN_CAR
    if 'FLAG_OWN_CAR' in df.columns:
        df['FLAG_OWN_CAR_Y'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
        df = df.drop(columns=['FLAG_OWN_CAR'])
    
    # FLAG_OWN_REALTY
    if 'FLAG_OWN_REALTY' in df.columns:
        df['FLAG_OWN_REALTY_Y'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
        df = df.drop(columns=['FLAG_OWN_REALTY'])
    
    # ============================================
    # 9. LIMPIAR INFINITOS Y NULOS
    # ============================================
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def validate_input(data: dict) -> bool:
    """Valida que los campos requeridos estén presentes"""
    required_fields = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    for field in required_fields:
        if field not in data:
            return False
    return True

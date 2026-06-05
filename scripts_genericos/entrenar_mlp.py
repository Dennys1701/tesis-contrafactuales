#!/usr/bin/env python3
# entrenar_mlp.py - Entrena MLP y guarda modelo en subcarpeta versionada.

import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[100, 32])
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--exp_id', type=str, default='')
    args = parser.parse_args()

    # --- Rutas ---
    # Datos originales (siempre desde carpeta base)
    original_base = os.path.join('experimentos', args.dataset)
    datos_originales = os.path.join(original_base, 'datos')
    csv_files = [f for f in os.listdir(datos_originales) if f.endswith('.csv')]
    data_path = os.path.join(datos_originales, csv_files[0])
    print(f"📂 Datos desde: {data_path}")

    # Subcarpeta versionada para guardar modelo
    if args.exp_id:
        exp_dir = os.path.join(original_base, args.exp_id)
        modelos_dir = os.path.join(exp_dir, 'modelos')
    else:
        modelos_dir = os.path.join(original_base, 'modelos')
    os.makedirs(modelos_dir, exist_ok=True)

    # --- Leer datos ---
    df = pd.read_csv(data_path)
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    # Codificar categóricas
    feature_encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        feature_encoders[col] = le

    # Codificar objetivo
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_encoded[target_col])
    X_raw = df_encoded[feature_cols].values

    # Escalar
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # --- Entrenar MLP ---
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu',
        solver='lbfgs',
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=args.verbose,
        alpha=args.alpha
    )
    mlp.fit(X, y)
    acc = accuracy_score(y, mlp.predict(X))
    print(f"✅ Precisión entrenamiento: {acc:.4f}")

    # --- Guardar ---
    modelo = {
        'model': mlp,
        'label_encoder': label_encoder,
        'feature_encoders': feature_encoders,
        'scaler': scaler,
        'categorical_features': categorical_cols,
        'metadata': {
            'features': feature_cols.tolist(),
            'target': target_col,
            'classes': label_encoder.classes_.tolist(),
            'train_accuracy': acc
        }
    }
    out_path = os.path.join(modelos_dir, f'mlp_{args.dataset}_limpio.pkl')
    joblib.dump(modelo, out_path)
    print(f"💾 Modelo guardado en {out_path}")

if __name__ == '__main__':
    main()
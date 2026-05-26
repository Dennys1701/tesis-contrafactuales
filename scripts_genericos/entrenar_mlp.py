#!/usr/bin/env python3
# entrenar_mlp.py - Entrena un MLP para un dataset y guarda modelo limpio.
# Uso: python entrenar_mlp.py --dataset iris

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Nombre del dataset (carpeta en experimentos/)')
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[100, 32])
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    base_exp = os.path.join('experimentos', args.dataset)
    datos_dir = os.path.join(base_exp, 'datos')
    modelos_dir = os.path.join(base_exp, 'modelos')
    os.makedirs(modelos_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(datos_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV en {datos_dir}")
    data_path = os.path.join(datos_dir, csv_files[0])

    df = pd.read_csv(data_path)
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu',
        solver='adam',
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=True
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión en test: {acc:.4f}")

    modelo_limpio = {
        'model': mlp,
        'label_encoder': label_encoder,
        'feature_encoders': {},
        'metadata': {
            'file_name': csv_files[0],
            'features': feature_names,
            'target': target_name,
            'classes': label_encoder.classes_.tolist(),
            'sample_count': len(df),
            'scaler': scaler
        }
    }
    output_path = os.path.join(modelos_dir, f'mlp_{args.dataset}_limpio.pkl')
    joblib.dump(modelo_limpio, output_path)
    print(f"✅ Modelo guardado en {output_path}")

if __name__ == '__main__':
    main()
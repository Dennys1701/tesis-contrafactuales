#!/usr/bin/env python3
# extraer_arboles.py - Extrae árboles Trepan y Trepan Reloaded para un dataset.
# Uso: python extraer_arboles.py --dataset iris --exp_id mi_exp --max_depth 15

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from surrogate.trepan_extractor import TREPANExtractor
from surrogate.trepan_reloaded_extractor import TrepanReloadedExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_samples_split', type=int, default=10)
    parser.add_argument('--min_samples_leaf', type=int, default=5)
    args = parser.parse_args()

    # Construir rutas versionadas
    if args.exp_id:
        base_exp = os.path.join('experimentos', args.dataset, args.exp_id)
    else:
        base_exp = os.path.join('experimentos', args.dataset)
    modelos_dir = os.path.join(base_exp, 'modelos')
    os.makedirs(modelos_dir, exist_ok=True)

    # Cargar MLP y metadatos desde la misma subcarpeta (o desde la base si no existe)
    mlp_path = os.path.join(modelos_dir, f'mlp_{args.dataset}_limpio.pkl')
    if not os.path.exists(mlp_path):
        # Fallback: buscar en la carpeta base sin versionar
        base_fallback = os.path.join('experimentos', args.dataset)
        mlp_path = os.path.join(base_fallback, 'modelos', f'mlp_{args.dataset}_limpio.pkl')
    mlp_data = joblib.load(mlp_path)
    mlp = mlp_data['model']
    scaler = mlp_data.get('scaler')
    label_encoder = mlp_data['label_encoder']
    feature_encoders = mlp_data.get('feature_encoders', {})
    feature_names = mlp_data['metadata']['features']
    class_names = mlp_data['metadata']['classes']

    # Leer datos originales (siempre desde la carpeta base)
    original_data_dir = os.path.join('experimentos', args.dataset, 'datos')
    csv_files = [f for f in os.listdir(original_data_dir) if f.endswith('.csv')]
    data_path = os.path.join(original_data_dir, csv_files[0])
    df = pd.read_csv(data_path)

    X_raw = df.iloc[:, :-1].values.copy()
    y_raw = df.iloc[:, -1].values

    # Aplicar codificadores a categóricas
    for col_name, le in feature_encoders.items():
        if col_name in df.columns:
            col_idx = df.columns.get_loc(col_name)
            X_raw[:, col_idx] = le.transform(X_raw[:, col_idx].astype(str))
            print(f"   ✔️ Aplicado codificador a '{col_name}'")
        else:
            print(f"⚠️ Advertencia: columna '{col_name}' no encontrada.")

    # Escalar
    if scaler is not None:
        X = scaler.transform(X_raw)
        print("   ✔️ Datos escalados")
    else:
        X = X_raw
        print("   ⚠️ Sin escalado")

    y = label_encoder.transform(y_raw)
    print(f"   ✔️ Clases codificadas: {label_encoder.classes_}")

    # Extraer árboles con los parámetros de profundidad
    print("\nExtrayendo Trepan original...")
    trepan = TREPANExtractor(max_depth=args.max_depth,
                             min_samples_split=args.min_samples_split,
                             min_samples_leaf=args.min_samples_leaf)
    trepan.extract_tree(mlp, X, y, sample_size=args.sample_size,
                        feature_names=feature_names, class_names=class_names)
    trepan_path = os.path.join(modelos_dir, f'trepan_{args.dataset}.pkl')
    joblib.dump(trepan, trepan_path)

    print("\nExtrayendo Trepan Reloaded...")
    trepan_rel = TrepanReloadedExtractor(max_depth=args.max_depth,
                                         min_samples_split=args.min_samples_split,
                                         min_samples_leaf=args.min_samples_leaf)
    trepan_rel.extract_tree(mlp, X, y, sample_size=args.sample_size,
                            feature_names=feature_names, class_names=class_names)
    trepan_rel_path = os.path.join(modelos_dir, f'trepan_reloaded_{args.dataset}.pkl')
    joblib.dump(trepan_rel, trepan_rel_path)

    print("\n✅ Árboles guardados en:", modelos_dir)

if __name__ == '__main__':
    main()
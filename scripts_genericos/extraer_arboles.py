#!/usr/bin/env python3
# extraer_arboles.py - Extrae árboles Trepan y Trepan Reloaded para un dataset.
# Uso: python extraer_arboles.py --dataset iris

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trepan_extractor import TREPANExtractor
from trepan_reloaded_extractor import TrepanReloadedExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sample_size', type=int, default=2000)
    args = parser.parse_args()

    base_exp = os.path.join('experimentos', args.dataset)
    modelos_dir = os.path.join(base_exp, 'modelos')
    datos_dir = os.path.join(base_exp, 'datos')

    mlp_path = os.path.join(modelos_dir, f'mlp_{args.dataset}_limpio.pkl')
    mlp_data = joblib.load(mlp_path)
    mlp = mlp_data['model']
    scaler = mlp_data['metadata'].get('scaler')
    label_encoder = mlp_data['label_encoder']
    feature_names = mlp_data['metadata']['features']
    class_names = mlp_data['metadata']['classes']

    csv_files = [f for f in os.listdir(datos_dir) if f.endswith('.csv')]
    data_path = os.path.join(datos_dir, csv_files[0])
    df = pd.read_csv(data_path)
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    if scaler is not None:
        X = scaler.transform(X_raw)
    else:
        X = X_raw
    y = label_encoder.transform(y_raw)

    print("Extrayendo Trepan original...")
    trepan = TREPANExtractor()
    trepan.extract_tree(mlp, X, y, sample_size=args.sample_size,
                        feature_names=feature_names, class_names=class_names)
    joblib.dump(trepan, os.path.join(modelos_dir, f'trepan_{args.dataset}.pkl'))

    print("Extrayendo Trepan Reloaded...")
    trepan_rel = TrepanReloadedExtractor()
    trepan_rel.extract_tree(mlp, X, y, sample_size=args.sample_size,
                            feature_names=feature_names, class_names=class_names)
    joblib.dump(trepan_rel, os.path.join(modelos_dir, f'trepan_reloaded_{args.dataset}.pkl'))

    print("✅ Árboles guardados.")

if __name__ == '__main__':
    main()
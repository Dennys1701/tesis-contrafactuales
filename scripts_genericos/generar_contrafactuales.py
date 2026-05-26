#!/usr/bin/env python3
# scripts_genericos/generar_contrafactuales.py
# Soporte para categóricas, guarda parámetros de ejecución en JSON

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cogs.evolution import Evolution
from cogs.fitness import gower_fitness_function

def load_dataset_and_model(dataset_name):
    base = os.path.join('experimentos', dataset_name)
    data_dir = os.path.join(base, 'datos')
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    processed = [f for f in csv_files if 'procesado' in f.lower()]
    data_path = os.path.join(data_dir, (processed[0] if processed else csv_files[0]))

    mlp_path = os.path.join(base, 'modelos', f'mlp_{dataset_name}_limpio.pkl')
    trepan_path = os.path.join(base, 'modelos', f'trepan_{dataset_name}.pkl')
    trepan_rel_path = os.path.join(base, 'modelos', f'trepan_reloaded_{dataset_name}.pkl')

    df = pd.read_csv(data_path)
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    mlp_data = joblib.load(mlp_path)
    mlp_model = mlp_data['model']
    label_encoder = mlp_data.get('label_encoder')
    scaler = mlp_data.get('metadata', {}).get('scaler', None)

    meta_files = [f for f in os.listdir(data_dir) if f.lower().endswith('metadata.pkl')]
    if meta_files:
        meta = joblib.load(os.path.join(data_dir, meta_files[0]))
        categorical_indices = meta.get('categorical_feature_indices', [])
        categorical_categories = meta.get('categorical_feature_categories', [])
    else:
        categorical_indices = []
        categorical_categories = []

    if scaler is not None:
        X = scaler.transform(X_raw)
    else:
        X = X_raw

    if label_encoder is not None:
        y = label_encoder.transform(y_raw)
        class_names = label_encoder.classes_.tolist()
    else:
        y = y_raw.astype(int)
        class_names = sorted(np.unique(y))

    trepan_data = joblib.load(trepan_path)
    trepan_tree = trepan_data.explainer_tree
    trepan_rel_data = joblib.load(trepan_rel_path)
    trepan_rel_tree = trepan_rel_data.explainer_tree

    return (X, y, mlp_model, trepan_tree, trepan_rel_tree,
            class_names, scaler, categorical_indices, categorical_categories)

def generar_contrafactual(x_orig, target_class, mlp_model, feature_intervals,
                          categorical_indices, pop_size, n_gen):
    evol = Evolution(
        x=x_orig,
        fitness_function=gower_fitness_function,
        fitness_function_kwargs={
            'blackbox': mlp_model,
            'desired_class': target_class,
            'apply_fixes': True
        },
        feature_intervals=feature_intervals,
        indices_categorical_features=categorical_indices,
        plausibility_constraints=[None] * len(x_orig),
        evolution_type='classic',
        population_size=pop_size,
        n_generations=n_gen,
        mutation_probability='inv_mutable_genotype_length',
        num_features_mutation_strength=0.25,
        verbose=False
    )
    evol.run()
    return evol.elite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sample_ratio', type=float, default=0.33)
    parser.add_argument('--pop_size', type=int, default=500)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    (X, y, mlp, trepan, trepan_rel, class_names,
     scaler, cat_indices, cat_categories) = load_dataset_and_model(args.dataset)

    n_total = len(X)
    n_muestras = int(n_total * args.sample_ratio)
    indices = np.random.choice(n_total, size=n_muestras, replace=False)

    n_features = X.shape[1]
    feature_intervals = []
    for i in range(n_features):
        if i in cat_indices:
            idx_in_cat = cat_indices.index(i)
            cat_list = cat_categories[idx_in_cat]
            feature_intervals.append(np.array(cat_list))
        else:
            col_min = float(X[:, i].min())
            col_max = float(X[:, i].max())
            feature_intervals.append((col_min, col_max))

    unique_classes = np.unique(y)
    resultados = []

    for idx in indices:
        x_orig = X[idx]
        y_true = y[idx]
        y_mlp = mlp.predict([x_orig])[0]
        y_trepan = trepan.predict([x_orig])[0]
        y_trepan_rel = trepan_rel.predict([x_orig])[0]

        if not (y_mlp == y_trepan == y_trepan_rel):
            resultados.append({
                'sample_index': int(idx),
                'coinciden': False,
                'x_original': x_orig.tolist(),
                'y_true': int(y_true),
                'y_mlp': int(y_mlp),
                'y_trepan': int(y_trepan),
                'y_trepan_rel': int(y_trepan_rel)
            })
            continue

        if len(unique_classes) == 2:
            target_class = 1 - y_mlp
        else:
            target_class = unique_classes[unique_classes != y_mlp][0]

        cf = generar_contrafactual(x_orig, target_class, mlp, feature_intervals,
                                   cat_indices, args.pop_size, args.generations)

        if cf is None:
            resultados.append({
                'sample_index': int(idx),
                'coinciden': True,
                'x_original': x_orig.tolist(),
                'y_true': int(y_true),
                'y_mlp': int(y_mlp),
                'y_trepan': int(y_trepan),
                'y_trepan_rel': int(y_trepan_rel),
                'cogs': {'success': False}
            })
            continue

        y_cf_mlp = mlp.predict([cf])[0]
        if y_cf_mlp != target_class:
            resultados.append({
                'sample_index': int(idx),
                'coinciden': True,
                'x_original': x_orig.tolist(),
                'y_true': int(y_true),
                'y_mlp': int(y_mlp),
                'y_trepan': int(y_trepan),
                'y_trepan_rel': int(y_trepan_rel),
                'cogs': {'success': False, 'x_cf': cf.tolist(), 'class_cf': int(y_cf_mlp)}
            })
            continue

        y_cf_trepan = trepan.predict([cf])[0]
        y_cf_trepan_rel = trepan_rel.predict([cf])[0]

        valido_trepan = (y_cf_trepan != y_mlp)
        coincide_trepan = valido_trepan and (y_cf_trepan == y_cf_mlp)
        valido_trepan_rel = (y_cf_trepan_rel != y_mlp)
        coincide_trepan_rel = valido_trepan_rel and (y_cf_trepan_rel == y_cf_mlp)

        resultados.append({
            'sample_index': int(idx),
            'coinciden': True,
            'x_original': x_orig.tolist(),
            'y_true': int(y_true),
            'y_mlp': int(y_mlp),
            'y_trepan': int(y_trepan),
            'y_trepan_rel': int(y_trepan_rel),
            'cogs': {
                'success': True,
                'x_cf': cf.tolist(),
                'class_cf': int(y_cf_mlp),
                'valido_trepan': bool(valido_trepan),
                'coincide_trepan': bool(coincide_trepan),
                'valido_trepan_rel': bool(valido_trepan_rel),
                'coincide_trepan_rel': bool(coincide_trepan_rel)
            }
        })

    output_dir = os.path.join('experimentos', args.dataset, 'resultados')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'cf_results_{args.dataset}_cogs.json')

    final_output = {
        'execution_parameters': {
            'dataset': args.dataset,
            'sample_ratio': args.sample_ratio,
            'population_size': args.pop_size,
            'generations': args.generations,
            'seed': args.seed,
            'n_total_samples': n_total,
            'n_selected_samples': n_muestras,
            'n_features': n_features,
            'categorical_features_indices': cat_indices
        },
        'results': resultados
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"✅ Resultados guardados en {output_path}")

if __name__ == '__main__':
    main()
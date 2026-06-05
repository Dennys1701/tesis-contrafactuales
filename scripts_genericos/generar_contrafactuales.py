#!/usr/bin/env python3
# scripts_genericos/generar_contrafactuales.py
# Soporte para categóricas, guarda parámetros de ejecución en JSON
# MODIFICADO: soporte para versionado con --exp_id

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

# -------------------------------------------------------------------
# FUNCIÓN PRINCIPAL PARA GENERAR UN CONTRAFACTUAL USANDO CoGS
# -------------------------------------------------------------------
def generar_contrafactual(x_orig, target_class, blackbox, feature_intervals,
                          categorical_indices, pop_size, n_gen):
    """
    Genera un contrafactual que cambia la clase de blackbox (MLP) a target_class.
    
    Parámetros:
    - x_orig: array 1D, instancia original (escalada y codificada)
    - target_class: int, clase deseada para el contrafactual
    - blackbox: modelo MLP (con método predict)
    - feature_intervals: lista de intervalos (min, max) para numéricas, o lista de valores para categóricas
    - categorical_indices: lista de índices de características categóricas
    - pop_size: tamaño de la población
    - n_gen: número de generaciones
    
    Retorna:
    - array 1D con el contrafactual, o None si no se encuentra.
    """
    fitness_kwargs = {
        'blackbox': blackbox,
        'desired_class': target_class,
        'apply_fixes': False
    }
    
    evol = Evolution(
        x=x_orig,
        fitness_function=gower_fitness_function,
        fitness_function_kwargs=fitness_kwargs,
        feature_intervals=feature_intervals,
        indices_categorical_features=categorical_indices,
        plausibility_constraints=None,
        evolution_type='classic',
        population_size=pop_size,
        n_generations=n_gen,
        mutation_probability='inv_mutable_genotype_length',
        num_features_mutation_strength=0.25,
        num_features_mutation_strength_decay=None,
        num_features_mutation_strength_decay_generations=None,
        init_temperature=0.8,
        selection_name='tournament_2',
        noisy_evaluations=False,
        verbose=False
    )
    
    evol.run()
    best = evol.elite
    if best is None:
        return None
    
    pred = blackbox.predict([best])[0]
    if pred != target_class:
        return None
    return best

# -------------------------------------------------------------------
# CARGA DE DATOS Y MODELOS (con preprocesamiento coherente y soporte versionado)
# -------------------------------------------------------------------
def load_dataset_and_model(dataset_name, exp_id=""):
    # Base de datos original (sin versionar) para los CSV
    original_base = os.path.join('experimentos', dataset_name)
    data_dir = os.path.join(original_base, 'datos')
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_path = os.path.join(data_dir, csv_files[0])

    # Determinar directorio de modelos (versionado o no)
    if exp_id:
        models_dir = os.path.join('experimentos', dataset_name, exp_id, 'modelos')
    else:
        models_dir = os.path.join(original_base, 'modelos')
    
    # Rutas de los modelos (primero en versionado, luego fallback)
    mlp_path = os.path.join(models_dir, f'mlp_{dataset_name}_limpio.pkl')
    trepan_path = os.path.join(models_dir, f'trepan_{dataset_name}.pkl')
    trepan_rel_path = os.path.join(models_dir, f'trepan_reloaded_{dataset_name}.pkl')
    
    # Fallback a la carpeta base si no existen en la versionada
    if not os.path.exists(mlp_path):
        mlp_path = os.path.join(original_base, 'modelos', f'mlp_{dataset_name}_limpio.pkl')
    if not os.path.exists(trepan_path):
        trepan_path = os.path.join(original_base, 'modelos', f'trepan_{dataset_name}.pkl')
    if not os.path.exists(trepan_rel_path):
        trepan_rel_path = os.path.join(original_base, 'modelos', f'trepan_reloaded_{dataset_name}.pkl')

    df = pd.read_csv(data_path)
    X_raw = df.iloc[:, :-1].values.copy()
    y_raw = df.iloc[:, -1].values

    mlp_data = joblib.load(mlp_path)
    mlp_model = mlp_data['model']
    label_encoder = mlp_data.get('label_encoder')
    scaler = mlp_data.get('scaler')
    feature_encoders = mlp_data.get('feature_encoders', {})

    # Aplicar codificadores a las columnas categóricas
    for col_name, le in feature_encoders.items():
        if col_name in df.columns:
            col_idx = df.columns.get_loc(col_name)
            X_raw[:, col_idx] = le.transform(X_raw[:, col_idx].astype(str))
        else:
            print(f"⚠️ Advertencia: columna '{col_name}' no encontrada en el dataset.")

    # Escalar (si hay scaler)
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

    # Cargar árboles
    trepan_data = joblib.load(trepan_path)
    trepan_tree = trepan_data.explainer_tree
    trepan_rel_data = joblib.load(trepan_rel_path)
    trepan_rel_tree = trepan_rel_data.explainer_tree

    # Obtener índices y valores reales de las columnas categóricas
    categorical_indices = []
    categorical_values = []
    for col_name in feature_encoders.keys():
        if col_name in df.columns:
            col_idx = df.columns.get_loc(col_name)
            categorical_indices.append(col_idx)
            unique_vals = np.unique(X[:, col_idx])
            categorical_values.append(unique_vals)

    return (X, y, mlp_model, trepan_tree, trepan_rel_tree,
            class_names, scaler, categorical_indices, categorical_values)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sample_ratio', type=float, default=0.33)
    parser.add_argument('--pop_size', type=int, default=500)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_id', type=str, default='', help='Identificador de experimento (subcarpeta)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    (X, y, mlp, trepan, trepan_rel, class_names,
     scaler, cat_indices, cat_values) = load_dataset_and_model(args.dataset, args.exp_id)
    
    # ---- VERIFICACIÓN DEL MLP ----
    preds = mlp.predict(X)
    unique_pred = np.unique(preds)
    unique_true = np.unique(y)
    print("Distribución de clases reales:", np.unique(y, return_counts=True))
    print("Distribución de clases predichas:", np.unique(preds, return_counts=True))
    if len(np.unique(preds)) < 2:
        print("⚠️ ERROR: El MLP solo predice una clase. Reentrena con más capacidad.")
        return
    missing_classes = set(unique_true) - set(unique_pred)
    if missing_classes:
        print(f"⚠️ ADVERTENCIA: El MLP no predice las clases {missing_classes}. Los CF pueden no ser válidos.")
    # ------------------------------

    n_total = len(X)
    n_muestras = int(n_total * args.sample_ratio)
    indices = np.random.choice(n_total, size=n_muestras, replace=False)

    n_features = X.shape[1]
    feature_intervals = []
    for i in range(n_features):
        if i in cat_indices:
            idx_in_cat = cat_indices.index(i)
            feature_intervals.append(cat_values[idx_in_cat])
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
            candidates = [c for c in unique_classes if c != y_mlp]
            candidates.sort(key=lambda c: abs(c - y_mlp))
            target_class = candidates[0]

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

    # --- Guardado versionado ---
    if args.exp_id:
        output_dir = os.path.join('experimentos', args.dataset, args.exp_id, 'resultados')
    else:
        output_dir = os.path.join('experimentos', args.dataset, 'resultados')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'cf_results_{args.dataset}_cogs.json')

    final_output = {
        'execution_parameters': {
            'dataset': args.dataset,
            'exp_id': args.exp_id,
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
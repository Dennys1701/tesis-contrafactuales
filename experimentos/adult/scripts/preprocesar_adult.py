#!/usr/bin/env python3
# Preprocesa Adult: imputa, codifica ordinal, guarda metadatos
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'experimentos', 'adult', 'datos')
    os.makedirs(data_dir, exist_ok=True)
    input_csv = os.path.join(data_dir, 'adult.csv')
    output_csv = os.path.join(data_dir, 'adult_procesado.csv')
    metadata_pkl = os.path.join(data_dir, 'adult_metadata.pkl')

    df = pd.read_csv(input_csv)
    df.replace('?', np.nan, inplace=True)

    target = 'income'
    X = df.drop(columns=[target])
    y = df[target]
    y_encoded = y.map({'<=50K': 0, '>50K': 1})

    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                        'race', 'gender', 'native-country']

    num_imputer = SimpleImputer(strategy='median')
    X_num = num_imputer.fit_transform(X[numeric_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat_str = cat_imputer.fit_transform(X[categorical_cols])

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_encoded = ordinal_encoder.fit_transform(X_cat_str)

    X_processed = np.hstack([X_num, X_cat_encoded])

    all_cols = numeric_cols + categorical_cols
    df_out = pd.DataFrame(X_processed, columns=all_cols)
    df_out[target] = y_encoded
    df_out.to_csv(output_csv, index=False)

    metadata = {
        'categorical_feature_indices': list(range(len(numeric_cols), len(all_cols))),
        'categorical_feature_categories': [list(range(len(cat))) for cat in ordinal_encoder.categories_],
        'feature_names': all_cols,
        'target_name': target,
        'original_feature_names': all_cols,
        'categorical_original_names': categorical_cols,
        'numeric_original_names': numeric_cols
    }
    joblib.dump(metadata, metadata_pkl)
    print(f"✅ Preprocesado guardado en {output_csv}")
    print(f"✅ Metadatos guardados en {metadata_pkl}")

if __name__ == '__main__':
    main()
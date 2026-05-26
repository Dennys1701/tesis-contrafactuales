import os
import pandas as pd
import joblib

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_dir = os.path.join(base_dir, 'experimentos', 'south_german_credit', 'datos')
    csv_path = os.path.join(data_dir, 'south_german_credit.csv')

    if not os.path.exists(csv_path):
        print(f"❌ No se encuentra {csv_path}")
        return

    df = pd.read_csv(csv_path)
    categorical_cols = [
        'account_check_status', 'credit_history', 'purpose', 'savings',
        'present_emp_since', 'personal_status_sex', 'other_debtors', 'property',
        'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker'
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != 'credit_risk']
    all_cols = [c for c in df.columns if c != 'credit_risk']
    categorical_indices = [all_cols.index(c) for c in categorical_cols]
    categorical_categories = [sorted(df[col].unique()) for col in categorical_cols]

    metadata = {
        'categorical_feature_indices': categorical_indices,
        'categorical_feature_categories': categorical_categories,
        'feature_names': all_cols,
        'target_name': 'credit_risk',
        'original_feature_names': all_cols,
        'categorical_original_names': categorical_cols,
        'numeric_original_names': numeric_cols
    }
    metadata_path = os.path.join(data_dir, 'south_german_credit_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadatos guardados en {metadata_path}")

if __name__ == '__main__':
    main()
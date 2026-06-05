#!/bin/bash
# run_all_experiments.sh - Ejecución con guardado versionado y parámetros de árboles

SAMPLE_RATIO=0.33
SEED=42

# Parámetros de los árboles (ajústalos según tus pruebas)
TREE_MAX_DEPTH=20
TREE_MIN_SPLIT=2
TREE_MIN_LEAF=1

# Generar ID único para esta ejecución
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="depth${TREE_MAX_DEPTH}_split${TREE_MIN_SPLIT}_leaf${TREE_MIN_LEAF}_${TIMESTAMP}"

echo "========================================="
echo "ID del experimento: $EXP_ID"
echo "========================================="

# Lista de datasets
datasets="iris wdbc sonar german_credit hepatitis wine"

for dataset in $datasets; do
    case $dataset in
        iris)
            mlp_hidden="30 15"
            mlp_iter=2000
            trepan_sample=2000
            cogs_pop=1500
            cogs_gen=300
            ;;
        wdbc)
            mlp_hidden="40 20"
            mlp_iter=2000
            trepan_sample=3000
            cogs_pop=1000
            cogs_gen=200
            ;;
        sonar)
            mlp_hidden="80 40"
            mlp_iter=3000
            trepan_sample=15000
            cogs_pop=800
            cogs_gen=400
            ;;
        german_credit)
            mlp_hidden="30 15"
            mlp_iter=2000
            trepan_sample=4000
            cogs_pop=1200
            cogs_gen=250
            ;;
        hepatitis)
            mlp_hidden="20 10"
            mlp_iter=1000
            trepan_sample=2000
            cogs_pop=800
            cogs_gen=200
            ;;
        wine)
            mlp_hidden="20 10"
            mlp_iter=1000
            trepan_sample=2000
            cogs_pop=1500
            cogs_gen=300
            ;;
    esac

    echo "========================================="
    echo "Procesando dataset: $dataset"
    echo "========================================="

    # Crear estructura versionada (subcarpeta con EXP_ID)
    EXP_DIR="experimentos/${dataset}/${EXP_ID}"
    mkdir -p "${EXP_DIR}/modelos"
    mkdir -p "${EXP_DIR}/resultados"

    # Entrenar MLP (usando exp_id)
    python scripts_genericos/entrenar_mlp.py \
        --dataset $dataset \
        --hidden_layers $mlp_hidden \
        --max_iter $mlp_iter \
        --random_state $SEED \
        --exp_id $EXP_ID

    # Extraer árboles con parámetros de profundidad
    python scripts_genericos/extraer_arboles.py \
        --dataset $dataset \
        --sample_size $trepan_sample \
        --exp_id $EXP_ID \
        --max_depth $TREE_MAX_DEPTH \
        --min_samples_split $TREE_MIN_SPLIT \
        --min_samples_leaf $TREE_MIN_LEAF

    # Generar contrafactuales
    python scripts_genericos/generar_contrafactuales.py \
        --dataset $dataset \
        --sample_ratio $SAMPLE_RATIO \
        --pop_size $cogs_pop \
        --generations $cogs_gen \
        --seed $SEED \
        --exp_id $EXP_ID

    # Calcular indicadores
    python scripts_genericos/procedimiento.py \
        --dataset $dataset \
        --cf_method cogs \
        --exp_id $EXP_ID

    echo ""
done

echo "========================================="
echo "Experimento completado. Resultados guardados en:"
echo "  experimentos/*/${EXP_ID}/resultados/"
echo "========================================="
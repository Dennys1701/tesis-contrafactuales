# Mejora de modelos sustitutos (Trepan) usando explicaciones contrafácticas (CoGS)

## Trabajo de Diploma en Inteligencia Artificial Explicable (XAI)

### Descripción
Este proyecto implementa:
- **Trepan** y **Trepan-Reloaded**: extractores de árboles de decisión como modelos sustitutos de una MLP (caja negra).
- **CoGS** (Counterfactual Generation using evolutionary Strategy): generador de explicaciones contrafácticas.
- **Método de mejora**: uso de los contrafactuales para aumentar el conjunto de entrenamiento y mejorar la fidelidad de los árboles sustitutos.

### Estructura del proyecto
diploma-xai-trepan-cogs/
├── src/
│ ├── surrogate/ # Extractores Trepan y Trepan-Reloaded
│ └── cogs/ # Algoritmo evolutivo CoGS
├── scripts_genericos/ # Scripts de entrenamiento, extracción y evaluación
├── run_all_experiments.sh # Orquestador de experimentos
└── README.md

text

### Requisitos
- Python 3.9+
- Bibliotecas: `pip install scikit-learn pandas numpy joblib`

### Uso rápido
```bash
# 1. Configurar parámetros en run_all_experiments.sh
# 2. Ejecutar experimento completo
./run_all_experiments.sh

# 3. Mejorar árboles con contrafactuales
python scripts_genericos/mejorar_arboles.py --dataset iris --exp_id <ID_BASE> --output_exp_id mejorado_v1
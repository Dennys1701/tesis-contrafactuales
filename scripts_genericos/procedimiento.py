#!/usr/bin/env python3
# procedimiento.py - Calcula indicadores a partir de los resultados JSON versionados.
# Uso: python procedimiento.py --dataset iris --exp_id mi_exp

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cf_method', default='cogs', choices=['cogs'])
    parser.add_argument('--exp_id', type=str, default='')
    args = parser.parse_args()

    if args.exp_id:
        json_path = os.path.join('experimentos', args.dataset, args.exp_id, 'resultados',
                                  f'cf_results_{args.dataset}_{args.cf_method}.json')
    else:
        json_path = os.path.join('experimentos', args.dataset, 'resultados',
                                  f'cf_results_{args.dataset}_{args.cf_method}.json')

    if not os.path.exists(json_path):
        print(f"❌ No se encuentra {json_path}. Ejecuta generar_contrafactuales.py primero.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'results' in data:
        resultados = data['results']
        print("\n⚙️ Parámetros de ejecución:")
        for k, v in data.get('execution_parameters', {}).items():
            print(f"   {k}: {v}")
    else:
        resultados = data

    total = len(resultados)
    coinciden_inicial = sum(1 for r in resultados if r['coinciden'])
    cfs_generados = sum(1 for r in resultados if r.get('cogs', {}).get('success', False))

    trepan_valido = 0
    trepan_coincide = 0
    reloaded_valido = 0
    reloaded_coincide = 0

    for r in resultados:
        cogs = r.get('cogs', {})
        if cogs.get('success', False):
            if cogs.get('valido_trepan', False):
                trepan_valido += 1
                if cogs.get('coincide_trepan', False):
                    trepan_coincide += 1
            if cogs.get('valido_trepan_rel', False):
                reloaded_valido += 1
                if cogs.get('coincide_trepan_rel', False):
                    reloaded_coincide += 1

    if cfs_generados == 0:
        print("No se generaron contrafactuales.")
        return

    print("\n" + "="*80)
    print(f"INDICADORES PARA {args.dataset.upper()} - {args.cf_method.upper()} (exp_id: {args.exp_id or 'default'})")
    print("="*80)
    print(f"Total ejemplos evaluados: {total}")
    print(f"Coincidencia inicial MLP = árboles: {coinciden_inicial} ({coinciden_inicial/total*100:.1f}%)")
    print(f"Contrafactuales generados: {cfs_generados} ({cfs_generados/coinciden_inicial*100:.1f}%)")
    print("\n--- TREPAN ---")
    print(f"CF válidos en Trepan: {trepan_valido} ({trepan_valido/cfs_generados*100:.1f}%)")
    print(f"CF con clase árbol == clase MLP (I1): {trepan_coincide} ({trepan_coincide/cfs_generados*100:.1f}%)")
    print(f"CF válidos pero clase diferente (I2): {trepan_valido - trepan_coincide} ({(trepan_valido - trepan_coincide)/cfs_generados*100:.1f}%)")
    print("\n--- TREPAN RELOADED ---")
    print(f"CF válidos en Reloaded: {reloaded_valido} ({reloaded_valido/cfs_generados*100:.1f}%)")
    print(f"CF con clase árbol == clase MLP (I1): {reloaded_coincide} ({reloaded_coincide/cfs_generados*100:.1f}%)")
    print(f"CF válidos pero clase diferente (I2): {reloaded_valido - reloaded_coincide} ({(reloaded_valido - reloaded_coincide)/cfs_generados*100:.1f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
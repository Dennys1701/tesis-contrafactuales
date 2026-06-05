from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import graphviz
import os
import webbrowser
import numpy as np
from sklearn.metrics import accuracy_score
from dtreeviz import dtreeviz

class TREPANExtractor:
    def __init__(self, max_depth=6, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.explainer_tree = None
        self.last_exported_image_path = None

    def extract_tree(self, mlp_model, X_encoded, y_encoded, sample_size=2000, feature_names=None, class_names=None):
        """
        Implementa algoritmo Trepan-Original melhorado para extração de árvore interpretável.
        
        O algoritmo Trepan-Original funciona da seguinte forma:
        1. Gera dados sintéticos inteligentes baseados no MLP
        2. Treina árvore de decisão iterativamente
        3. Valida fidelidade sobre dados reais e sintéticos
        4. Otimiza para máxima interpretabilidade
        
        Args:
            mlp_model: Modelo MLP treinado
            X_encoded: Dados codificados
            y_encoded: Labels codificados
            sample_size: Tamanho da amostra sintética
            feature_names: Lista com nomes dos atributos (ex: ['petalwidth', 'petallength', ...])
            class_names: Lista com nomes das classes (ex: ['Iris-setosa', 'Iris-versicolor', ...])
        """
        n_features = X_encoded.shape[1]
        
        # Se feature_names não fornecido, cria nomes genéricos
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Se class_names não fornecido, tenta inferir ou cria genéricos
        if class_names is None:
            unique_classes = np.unique(y_encoded)
            class_names = [f"class_{int(cls)}" for cls in unique_classes]
        
        # Fase 1: Geração inteligente de dados sintéticos
        X_synthetic = self._generate_intelligent_synthetic_data(
            mlp_model, X_encoded, sample_size
        )
        
        # Fase 2: Predição do MLP sobre dados sintéticos
        y_synthetic = mlp_model.predict(X_synthetic)
        
        # Fase 3: Treinamento iterativo da árvore (algoritmo Trepan-Original)
        self.explainer_tree = self._train_trepan_tree(
            X_synthetic, y_synthetic, X_encoded, y_encoded
        )
        
        # Fase 4: Validação de fidelidade
        fidelity_synthetic = self._calculate_fidelity(
            self.explainer_tree, mlp_model, X_synthetic, y_synthetic
        )
        fidelity_real = self._calculate_fidelity(
            self.explainer_tree, mlp_model, X_encoded, y_encoded
        )
        
        # Fase 5: Geração de regras interpretáveis com nomes de atributos reais
        # Usar las clases reales del árbol para evitar error de discrepancia
        if self.explainer_tree.classes_.shape[0] != len(class_names):
            effective_class_names = [str(c) for c in self.explainer_tree.classes_]
        else:
            effective_class_names = class_names

        rules = export_text(
            self.explainer_tree,
            feature_names=feature_names,
            class_names=effective_class_names
        )
        
        # Relatório detalhado de interpretabilidade
        explanation = self._generate_interpretability_report(
            fidelity_synthetic, fidelity_real, X_encoded.shape[0]
        )

        return rules + explanation
    
    def _generate_intelligent_synthetic_data(self, mlp_model, X_encoded, sample_size):
        """
        Gera dados sintéticos inteligentes baseados no algoritmo Trepan-Original.
        
        Em vez de geração aleatória, usa:
        - Amostragem baseada em densidade dos dados originais
        - Geração em regiões de alta incerteza do MLP
        - Balanceamento entre diferentes classes
        """
        n_features = X_encoded.shape[1]
        
        # 1. Amostragem baseada em densidade (50% dos dados)
        density_samples = int(sample_size * 0.5)
        X_density = self._sample_by_density(X_encoded, density_samples)
        
        # 2. Geração em regiões de alta incerteza (30% dos dados)
        uncertainty_samples = int(sample_size * 0.3)
        X_uncertainty = self._sample_uncertainty_regions(mlp_model, X_encoded, uncertainty_samples)
        
        # 3. Geração aleatória balanceada (20% dos dados)
        random_samples = sample_size - density_samples - uncertainty_samples
        X_random = np.random.uniform(
            low=np.min(X_encoded, axis=0),
            high=np.max(X_encoded, axis=0),
            size=(random_samples, n_features)
        )
        
        # Combinar todos os tipos de amostras
        X_synthetic = np.vstack([X_density, X_uncertainty, X_random])
        
        return X_synthetic
    
    def _sample_by_density(self, X_encoded, n_samples):
        """Amostra pontos baseados na densidade dos dados originais"""
        from sklearn.neighbors import KernelDensity
        
        # Estima densidade dos dados originais
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(X_encoded)
        
        # Gera amostras baseadas na densidade estimada
        X_density = kde.sample(n_samples)
        return X_density
    
    def _sample_uncertainty_regions(self, mlp_model, X_encoded, n_samples):
        """Gera amostras em regiões onde o MLP tem alta incerteza"""
        # Calcula probabilidades de predição para encontrar regiões de incerteza
        if hasattr(mlp_model, 'predict_proba'):
            probas = mlp_model.predict_proba(X_encoded)
            # Calcula entropia como medida de incerteza
            entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
            
            # Seleciona pontos com alta entropia (alta incerteza)
            high_uncertainty_indices = np.argsort(entropy)[-n_samples:]
            X_uncertainty_base = X_encoded[high_uncertainty_indices]
            
            # Adiciona ruído gaussiano para gerar variações
            noise = np.random.normal(0, 0.1, X_uncertainty_base.shape)
            X_uncertainty = X_uncertainty_base + noise
            
            return X_uncertainty
        else:
            # Fallback para geração aleatória se não há predict_proba
            return np.random.uniform(
                low=np.min(X_encoded, axis=0),
                high=np.max(X_encoded, axis=0),
                size=(n_samples, X_encoded.shape[1])
            )
    
    def _train_trepan_tree(self, X_synthetic, y_synthetic, X_real, y_real):
        # Usar los parámetros pasados al constructor, no los adaptativos
        tree = DecisionTreeClassifier(
           max_depth=self.max_depth,
           min_samples_split=self.min_samples_split,
           min_samples_leaf=self.min_samples_leaf,
           random_state=42,
           criterion='gini'
        )
        tree.fit(X_synthetic, y_synthetic)
        return tree
    
    def _validate_tree_quality(self, tree, X_real, y_real):
        """Valida qualidade da árvore gerada"""
        # Verifica se a árvore não está muito simples (underfitting)
        if tree.get_depth() < 2:
            print("⚠️ Aviso: Árvore muito simples, pode não capturar padrões complexos")
        
        # Verifica se a árvore não está muito complexa (overfitting)
        if tree.get_depth() > 8:
            print("⚠️ Aviso: Árvore muito complexa, pode ser difícil de interpretar")
    
    def _calculate_fidelity(self, tree, mlp_model, X_data, y_data):
        """Calcula fidelidade da árvore em relação ao MLP"""
        # Predições da árvore
        y_tree_pred = tree.predict(X_data)
        
        # Predições do MLP
        y_mlp_pred = mlp_model.predict(X_data)
        
        # Fidelidade = concordância entre árvore e MLP
        fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
        
        return fidelity
    
    def _generate_interpretability_report(self, fidelity_synthetic, fidelity_real, n_real_samples):
        """Gera relatório detalhado de interpretabilidade"""
        report = f"\n{'='*60}\n"
        report += f"📊 RELATÓRIO DE INTERPRETABILIDADE TREPAN-ORIGINAL\n"
        report += f"{'='*60}\n\n"
        
        report += f"🎯 Fidelidade da Árvore ao MLP:\n"
        report += f"   • Sobre dados sintéticos: {fidelity_synthetic:.3f} ({fidelity_synthetic*100:.1f}%)\n"
        report += f"   • Sobre dados reais: {fidelity_real:.3f} ({fidelity_real*100:.1f}%)\n"
        report += f"   • Amostras reais validadas: {n_real_samples}\n\n"
        
        # Interpretação da fidelidade
        avg_fidelity = (fidelity_synthetic + fidelity_real) / 2
        if avg_fidelity > 0.9:
            report += f"✅ Excelente interpretabilidade! A árvore representa fielmente o MLP.\n"
        elif avg_fidelity > 0.8:
            report += f"👍 Boa interpretabilidade. A árvore captura bem os padrões do MLP.\n"
        elif avg_fidelity > 0.7:
            report += f"⚠️ Interpretabilidade moderada. Alguns padrões podem não estar representados.\n"
        else:
            report += f"❌ Baixa interpretabilidade. A árvore não representa bem o MLP.\n"
        
        report += f"\n💡 Interpretabilidade significa que:\n"
        report += f"   • A árvore toma decisões similares ao MLP\n"
        report += f"   • As regras são compreensíveis por humanos\n"
        report += f"   • Você pode entender como o MLP funciona\n\n"
        
        report += f"🌳 Próximos passos:\n"
        report += f"   • Use '🌳 Visualizar Árvore' para explorar as regras\n"
        report += f"   • Use '💡 Explicações Naturais' para linguagem simples\n"
        report += f"   • Use '📊 Comparar Modelos' para análise detalhada\n"
        
        return report
    
    def generate_human_readable_rules(self, feature_names, class_names):
        """
        Gera regras em linguagem natural compreensível para humanos.
        
        Converte a árvore de decisão em regras simples e explicativas.
        """
        if self.explainer_tree is None:
            return "❌ Nenhuma árvore foi gerada ainda. Execute extract_tree primeiro."
        
        rules = []
        self._extract_rules_recursive(
            self.explainer_tree.tree_, 0, [], rules, feature_names, class_names
        )
        
        # Formata regras em linguagem natural
        human_rules = self._format_rules_for_humans(rules, class_names)
        
        return human_rules
    
    def _extract_rules_recursive(self, tree, node_id, conditions, rules, feature_names, class_names):
        """Extrai regras recursivamente da árvore"""
        from sklearn.tree import _tree
        
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Nó folha - adiciona regra completa
            class_id = np.argmax(tree.value[node_id][0])
            class_name = class_names[class_id]
            confidence = tree.value[node_id][0][class_id] / np.sum(tree.value[node_id][0])
            samples = tree.n_node_samples[node_id]
            
            rules.append({
                'conditions': conditions.copy(),
                'class': class_name,
                'confidence': confidence,
                'samples': samples
            })
            return
        
        # Nó interno - continua recursão
        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        
        # Condição para filho esquerdo (<=)
        left_conditions = conditions + [f"{feature_name} ≤ {threshold:.3f}"]
        self._extract_rules_recursive(
            tree, tree.children_left[node_id], left_conditions, rules, feature_names, class_names
        )
        
        # Condição para filho direito (>)
        right_conditions = conditions + [f"{feature_name} > {threshold:.3f}"]
        self._extract_rules_recursive(
            tree, tree.children_right[node_id], right_conditions, rules, feature_names, class_names
        )
    
    def _format_rules_for_humans(self, rules, class_names):
        """Formata regras em linguagem natural compreensível"""
        if not rules:
            return "❌ Nenhuma regra foi extraída da árvore."
        
        formatted_rules = f"\n{'='*60}\n"
        formatted_rules += f"🧠 REGRAS EM LINGUAGEM NATURAL\n"
        formatted_rules += f"{'='*60}\n\n"
        
        formatted_rules += f"📋 O modelo toma decisões baseado nas seguintes regras:\n\n"
        
        for i, rule in enumerate(rules, 1):
            formatted_rules += f"🔹 Regra {i}:\n"
            
            if rule['conditions']:
                formatted_rules += f"   Se {' E '.join(rule['conditions'])}\n"
            else:
                formatted_rules += f"   Se nenhuma condição específica\n"
            
            formatted_rules += f"   Então: Classe '{rule['class']}'\n"
            formatted_rules += f"   Confiança: {rule['confidence']:.1%}\n"
            formatted_rules += f"   Amostras: {rule['samples']}\n\n"
        
        # Análise de complexidade
        avg_conditions = np.mean([len(rule['conditions']) for rule in rules])
        formatted_rules += f"📊 Análise de Complexidade:\n"
        formatted_rules += f"   • Total de regras: {len(rules)}\n"
        formatted_rules += f"   • Condições médias por regra: {avg_conditions:.1f}\n"
        
        if avg_conditions <= 2:
            formatted_rules += f"   ✅ Regras simples e fáceis de entender\n"
        elif avg_conditions <= 4:
            formatted_rules += f"   ⚠️ Regras moderadamente complexas\n"
        else:
            formatted_rules += f"   ❌ Regras complexas, considere simplificar\n"
        
        formatted_rules += f"\n💡 Dicas para interpretação:\n"
        formatted_rules += f"   • Regras com alta confiança são mais confiáveis\n"
        formatted_rules += f"   • Regras com muitas amostras são mais representativas\n"
        formatted_rules += f"   • Use essas regras para entender o comportamento do MLP\n"
        
        return formatted_rules
    
    def get_tree_complexity_metrics(self):
        """Calcula métricas de complexidade da árvore para interpretabilidade"""
        if self.explainer_tree is None:
            return None
        
        tree = self.explainer_tree.tree_
        
        metrics = {
            'depth': self.explainer_tree.get_depth(),
            'n_leaves': self.explainer_tree.get_n_leaves(),
            'n_nodes': tree.node_count,
            'avg_path_length': self._calculate_avg_path_length(tree),
            'complexity_score': self._calculate_complexity_score(tree)
        }
        
        return metrics
    
    def _calculate_avg_path_length(self, tree):
        """Calcula comprimento médio dos caminhos na árvore"""
        from sklearn.tree import _tree
        
        def get_path_length(node_id, current_length=0):
            if tree.children_left[node_id] == _tree.TREE_LEAF:
                return current_length
            
            left_length = get_path_length(tree.children_left[node_id], current_length + 1)
            right_length = get_path_length(tree.children_right[node_id], current_length + 1)
            
            return (left_length + right_length) / 2
        
        return get_path_length(0)
    
    def _calculate_complexity_score(self, tree):
        """Calcula score de complexidade (0-1, onde 0 é mais simples)"""
        depth = self.explainer_tree.get_depth()
        n_leaves = self.explainer_tree.get_n_leaves()
        
        # Normaliza métricas para score entre 0 e 1
        depth_score = min(depth / 10, 1.0)  # Assumindo profundidade máxima de 10
        leaves_score = min(n_leaves / 50, 1.0)  # Assumindo máximo de 50 folhas
        
        # Score combinado (média ponderada)
        complexity_score = (depth_score * 0.6 + leaves_score * 0.4)
        
        return complexity_score
    
    def validate_interpretability(self, mlp_model, X_encoded, y_encoded, feature_names, class_names):
        """
        Valida a interpretabilidade da árvore gerada usando múltiplas métricas.
        
        Retorna um relatório completo de validação da interpretabilidade.
        """
        if self.explainer_tree is None:
            return "❌ Nenhuma árvore foi gerada ainda. Execute extract_tree primeiro."
        
        validation_report = f"\n{'='*70}\n"
        validation_report += f"🔍 VALIDAÇÃO DE INTERPRETABILIDADE TREPAN-ORIGINAL\n"
        validation_report += f"{'='*70}\n\n"
        
        # 1. Métricas de Fidelidade
        fidelity_metrics = self._calculate_fidelity_metrics(mlp_model, X_encoded, y_encoded)
        validation_report += self._format_fidelity_report(fidelity_metrics)
        
        # 2. Métricas de Complexidade
        complexity_metrics = self.get_tree_complexity_metrics()
        validation_report += self._format_complexity_report(complexity_metrics)
        
        # 3. Métricas de Consistência
        consistency_metrics = self._calculate_consistency_metrics(mlp_model, X_encoded, y_encoded)
        validation_report += self._format_consistency_report(consistency_metrics)
        
        # 4. Métricas de Estabilidade
        stability_metrics = self._calculate_stability_metrics(mlp_model, X_encoded, y_encoded)
        validation_report += self._format_stability_report(stability_metrics)
        
        # 5. Score Geral de Interpretabilidade
        overall_score = self._calculate_overall_interpretability_score(
            fidelity_metrics, complexity_metrics, consistency_metrics, stability_metrics
        )
        validation_report += self._format_overall_score_report(overall_score)
        
        return validation_report
    
    def _calculate_fidelity_metrics(self, mlp_model, X_encoded, y_encoded):
        """Calcula métricas detalhadas de fidelidade"""
        # Predições da árvore e MLP
        y_tree_pred = self.explainer_tree.predict(X_encoded)
        y_mlp_pred = mlp_model.predict(X_encoded)
        
        # Fidelidade geral
        overall_fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
        
        # Fidelidade por classe
        class_fidelities = {}
        unique_classes = np.unique(y_encoded)
        
        for cls in unique_classes:
            mask = y_encoded == cls
            if np.sum(mask) > 0:
                class_fidelity = accuracy_score(
                    y_mlp_pred[mask], y_tree_pred[mask]
                )
                class_fidelities[cls] = class_fidelity
        
        # Fidelidade em regiões de alta confiança do MLP
        if hasattr(mlp_model, 'predict_proba'):
            probas = mlp_model.predict_proba(X_encoded)
            max_probas = np.max(probas, axis=1)
            high_confidence_mask = max_probas > 0.8
            
            if np.sum(high_confidence_mask) > 0:
                high_conf_fidelity = accuracy_score(
                    y_mlp_pred[high_confidence_mask], 
                    y_tree_pred[high_confidence_mask]
                )
            else:
                high_conf_fidelity = 0.0
        else:
            high_conf_fidelity = overall_fidelity
        
        return {
            'overall': overall_fidelity,
            'by_class': class_fidelities,
            'high_confidence': high_conf_fidelity,
            'n_samples': len(X_encoded)
        }
    
    def _calculate_consistency_metrics(self, mlp_model, X_encoded, y_encoded):
        """Calcula métricas de consistência da árvore"""
        # Testa consistência com pequenas perturbações nos dados
        n_samples = min(100, len(X_encoded))
        test_indices = np.random.choice(len(X_encoded), n_samples, replace=False)
        X_test = X_encoded[test_indices]
        
        # Predições originais
        y_tree_orig = self.explainer_tree.predict(X_test)
        y_mlp_orig = mlp_model.predict(X_test)
        
        # Adiciona pequeno ruído e testa consistência
        noise_std = 0.01
        X_noisy = X_test + np.random.normal(0, noise_std, X_test.shape)
        
        y_tree_noisy = self.explainer_tree.predict(X_noisy)
        y_mlp_noisy = mlp_model.predict(X_noisy)
        
        # Consistência da árvore
        tree_consistency = accuracy_score(y_tree_orig, y_tree_noisy)
        
        # Consistência do MLP
        mlp_consistency = accuracy_score(y_mlp_orig, y_mlp_noisy)
        
        # Consistência relativa (árvore vs MLP)
        relative_consistency = tree_consistency / (mlp_consistency + 1e-10)
        
        return {
            'tree_consistency': tree_consistency,
            'mlp_consistency': mlp_consistency,
            'relative_consistency': relative_consistency,
            'n_test_samples': n_samples
        }
    
    def _calculate_stability_metrics(self, mlp_model, X_encoded, y_encoded):
        """Calcula métricas de estabilidade da árvore"""
        # Testa estabilidade com diferentes subconjuntos dos dados
        n_iterations = 5
        fidelity_scores = []
        
        for i in range(n_iterations):
            # Amostra aleatória dos dados
            n_samples = min(500, len(X_encoded))
            indices = np.random.choice(len(X_encoded), n_samples, replace=False)
            X_subset = X_encoded[indices]
            y_subset = y_encoded[indices]
            
            # Treina árvore temporária
            temp_tree = DecisionTreeClassifier(
                max_depth=self.explainer_tree.get_depth(),
                random_state=42
            )
            temp_tree.fit(X_subset, y_subset)
            
            # Calcula fidelidade
            y_tree_pred = temp_tree.predict(X_subset)
            y_mlp_pred = mlp_model.predict(X_subset)
            fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
            
            fidelity_scores.append(fidelity)
        
        # Estatísticas de estabilidade
        mean_fidelity = np.mean(fidelity_scores)
        std_fidelity = np.std(fidelity_scores)
        stability_score = 1 - std_fidelity  # Menor variância = maior estabilidade
        
        return {
            'mean_fidelity': mean_fidelity,
            'std_fidelity': std_fidelity,
            'stability_score': stability_score,
            'n_iterations': n_iterations
        }
    
    def _format_fidelity_report(self, fidelity_metrics):
        """Formata relatório de métricas de fidelidade"""
        report = f"🎯 MÉTRICAS DE FIDELIDADE:\n"
        report += f"   • Fidelidade geral: {fidelity_metrics['overall']:.3f} ({fidelity_metrics['overall']*100:.1f}%)\n"
        report += f"   • Fidelidade alta confiança: {fidelity_metrics['high_confidence']:.3f} ({fidelity_metrics['high_confidence']*100:.1f}%)\n"
        report += f"   • Amostras validadas: {fidelity_metrics['n_samples']}\n\n"
        
        report += f"   📊 Fidelidade por classe:\n"
        for cls, fidelity in fidelity_metrics['by_class'].items():
            report += f"      - Classe {cls}: {fidelity:.3f} ({fidelity*100:.1f}%)\n"
        
        report += f"\n"
        return report
    
    def _format_complexity_report(self, complexity_metrics):
        """Formata relatório de métricas de complexidade"""
        report = f"🌳 MÉTRICAS DE COMPLEXIDADE:\n"
        report += f"   • Profundidade: {complexity_metrics['depth']}\n"
        report += f"   • Número de folhas: {complexity_metrics['n_leaves']}\n"
        report += f"   • Número de nós: {complexity_metrics['n_nodes']}\n"
        report += f"   • Comprimento médio do caminho: {complexity_metrics['avg_path_length']:.2f}\n"
        report += f"   • Score de complexidade: {complexity_metrics['complexity_score']:.3f}\n\n"
        
        # Interpretação da complexidade
        if complexity_metrics['complexity_score'] < 0.3:
            report += f"   ✅ Árvore simples e fácil de interpretar\n"
        elif complexity_metrics['complexity_score'] < 0.6:
            report += f"   ⚠️ Árvore moderadamente complexa\n"
        else:
            report += f"   ❌ Árvore complexa, pode ser difícil de interpretar\n"
        
        report += f"\n"
        return report
    
    def _format_consistency_report(self, consistency_metrics):
        """Formata relatório de métricas de consistência"""
        report = f"🔄 MÉTRICAS DE CONSISTÊNCIA:\n"
        report += f"   • Consistência da árvore: {consistency_metrics['tree_consistency']:.3f}\n"
        report += f"   • Consistência do MLP: {consistency_metrics['mlp_consistency']:.3f}\n"
        report += f"   • Consistência relativa: {consistency_metrics['relative_consistency']:.3f}\n"
        report += f"   • Amostras testadas: {consistency_metrics['n_test_samples']}\n\n"
        
        # Interpretação da consistência
        if consistency_metrics['relative_consistency'] > 0.9:
            report += f"   ✅ Árvore muito consistente com o MLP\n"
        elif consistency_metrics['relative_consistency'] > 0.7:
            report += f"   👍 Árvore razoavelmente consistente\n"
        else:
            report += f"   ⚠️ Árvore pode ser inconsistente em algumas regiões\n"
        
        report += f"\n"
        return report
    
    def _format_stability_report(self, stability_metrics):
        """Formata relatório de métricas de estabilidade"""
        report = f"⚖️ MÉTRICAS DE ESTABILIDADE:\n"
        report += f"   • Fidelidade média: {stability_metrics['mean_fidelity']:.3f}\n"
        report += f"   • Desvio padrão: {stability_metrics['std_fidelity']:.3f}\n"
        report += f"   • Score de estabilidade: {stability_metrics['stability_score']:.3f}\n"
        report += f"   • Iterações testadas: {stability_metrics['n_iterations']}\n\n"
        
        # Interpretação da estabilidade
        if stability_metrics['stability_score'] > 0.8:
            report += f"   ✅ Árvore muito estável\n"
        elif stability_metrics['stability_score'] > 0.6:
            report += f"   👍 Árvore razoavelmente estável\n"
        else:
            report += f"   ⚠️ Árvore pode ser instável\n"
        
        report += f"\n"
        return report
    
    def _calculate_overall_interpretability_score(self, fidelity_metrics, complexity_metrics, 
                                                consistency_metrics, stability_metrics):
        """Calcula score geral de interpretabilidade"""
        # Pesos para diferentes métricas
        weights = {
            'fidelity': 0.4,      # Fidelidade é mais importante
            'complexity': 0.25,   # Complexidade afeta interpretabilidade
            'consistency': 0.2,   # Consistência é importante
            'stability': 0.15     # Estabilidade é menos crítica
        }
        
        # Normaliza métricas para 0-1
        fidelity_score = fidelity_metrics['overall']
        complexity_score = 1 - complexity_metrics['complexity_score']  # Inverte (menor complexidade = melhor)
        consistency_score = consistency_metrics['relative_consistency']
        stability_score = stability_metrics['stability_score']
        
        # Score ponderado
        overall_score = (
            weights['fidelity'] * fidelity_score +
            weights['complexity'] * complexity_score +
            weights['consistency'] * consistency_score +
            weights['stability'] * stability_score
        )
        
        return {
            'overall_score': overall_score,
            'fidelity_score': fidelity_score,
            'complexity_score': complexity_score,
            'consistency_score': consistency_score,
            'stability_score': stability_score,
            'weights': weights
        }
    
    def _format_overall_score_report(self, overall_score):
        """Formata relatório do score geral"""
        report = f"{'='*70}\n"
        report += f"🏆 SCORE GERAL DE INTERPRETABILIDADE\n"
        report += f"{'='*70}\n\n"
        
        report += f"📊 Score Final: {overall_score['overall_score']:.3f} ({overall_score['overall_score']*100:.1f}%)\n\n"
        
        report += f"📈 Contribuição de cada métrica:\n"
        report += f"   • Fidelidade: {overall_score['fidelity_score']:.3f} (peso: {overall_score['weights']['fidelity']})\n"
        report += f"   • Simplicidade: {overall_score['complexity_score']:.3f} (peso: {overall_score['weights']['complexity']})\n"
        report += f"   • Consistência: {overall_score['consistency_score']:.3f} (peso: {overall_score['weights']['consistency']})\n"
        report += f"   • Estabilidade: {overall_score['stability_score']:.3f} (peso: {overall_score['weights']['stability']})\n\n"
        
        # Interpretação do score geral
        score = overall_score['overall_score']
        if score > 0.9:
            report += f"🌟 EXCELENTE interpretabilidade! A árvore representa perfeitamente o MLP.\n"
        elif score > 0.8:
            report += f"✅ MUITO BOA interpretabilidade. A árvore é altamente confiável.\n"
        elif score > 0.7:
            report += f"👍 BOA interpretabilidade. A árvore captura bem os padrões do MLP.\n"
        elif score > 0.6:
            report += f"⚠️ Interpretabilidade MODERADA. Algumas limitações podem existir.\n"
        else:
            report += f"❌ BAIXA interpretabilidade. A árvore não representa bem o MLP.\n"
        
        report += f"\n💡 Recomendações:\n"
        if score < 0.7:
            report += f"   • Considere aumentar o tamanho da amostra sintética\n"
            report += f"   • Ajuste os parâmetros da árvore (profundidade, min_samples)\n"
            report += f"   • Verifique a qualidade dos dados de entrada\n"
        else:
            report += f"   • A árvore está pronta para uso em produção\n"
            report += f"   • Use as regras geradas para explicar decisões\n"
            report += f"   • Monitore a performance ao longo do tempo\n"
        
        report += f"\n{'='*70}\n"
        return report

    def export_tree_image(self, feature_names, class_names, output_file="tree_visualization.png", open_image=True):
        """
        Exporta a árvore gerada como imagem (PNG) usando Graphviz
        """
        if self.explainer_tree is None:
            raise ValueError("Árvore ainda não foi gerada. Execute extract_tree primeiro.")

        dot_data = export_graphviz(
            self.explainer_tree,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        graph.format = "png"
        graph.render(output_file, cleanup=True)

        file_path = os.path.abspath(f"{output_file}.png")
        self.last_exported_image_path = file_path

        if open_image:
            webbrowser.open(f"file://{file_path}")

        return f"Árvore exportada como imagem para {file_path}"

    def export_tree_dtreeviz(self, X_encoded, y_encoded, feature_names, class_names, output_file="tree_viz.svg", open_image=True):
        """
        Exporta visualização interpretável usando dtreeviz
        """
        if self.explainer_tree is None:
            raise ValueError("Árvore ainda não foi gerada. Execute extract_tree primeiro.")

        try:
            # Tentar usar dtreeviz se disponível
            viz = dtreeviz(
                self.explainer_tree,
                X_encoded,
                y_encoded,
                target_name="class",
                feature_names=feature_names,
                class_names=class_names,
                fancy=True
            )
            viz.save(output_file)

            abs_path = os.path.abspath(output_file)
            if open_image:
                webbrowser.open(f"file://{abs_path}")

            return f"Visualização interativa exportada para {abs_path}"
            
        except ImportError:
            # Se dtreeviz não estiver disponível, usar exportação simples
            return self.export_tree_image(feature_names, class_names, output_file.replace('.svg', '.png'), open_image)
        except Exception as e:
            # Se houver outro erro, usar exportação simples como fallback
            return self.export_tree_image(feature_names, class_names, output_file.replace('.svg', '.png'), open_image)


    def open_last_exported_image(self):
        """
        Abre a última imagem exportada, se existir
        """
        if self.last_exported_image_path and os.path.exists(self.last_exported_image_path):
            webbrowser.open(f"file://{self.last_exported_image_path}")
            return f"Imagem aberta: {self.last_exported_image_path}"
        else:
            return "Nenhuma imagem exportada encontrada."

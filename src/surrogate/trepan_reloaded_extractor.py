from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import graphviz
import os
import webbrowser
import numpy as np
from sklearn.metrics import accuracy_score
from dtreeviz import dtreeviz
import owlready2
from owlready2 import get_ontology, OwlReadyOntologyParsingError, IRIS
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import difflib
import pickle
import json
from pathlib import Path

class TrepanReloadedExtractor:
    """
    Extrator Trepan-Reloaded com integração de ontologia.
    
    Esta é uma versão melhorada do Trepan-Original que incorpora:
    1. Conhecimento de domínio através de ontologias OWL
    2. Explicações semânticas mais ricas
    3. Validação de coerência semântica
    4. Relacionamentos entre conceitos do domínio
    """
    
    def __init__(self, ontology=None, max_depth=6, min_samples_split=10, min_samples_leaf=5):
        self.explainer_tree = None
        self.ontology = ontology
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.domain_knowledge = {}
        self.semantic_rules = []
        self.last_exported_image_path = None
        self.feature_semantics = {}
        self.class_semantics = {}
        self.mapping_cache = {}
        self.unmapped_features = []
        self.unmapped_classes = []
        self.mapping_scores = {}
        self.ontology_hierarchy = {}
        self.property_constraints = {}
        self.datatype_ranges = {}
    
    def extract_tree(self, mlp_model, X_encoded, y_encoded, sample_size=2000, feature_names=None, class_names=None):
        """
        Método compatível que delega para extract_tree_with_ontology.
        Mantém compatibilidade com código que espera Trepan-Original.
        """
        return self.extract_tree_with_ontology(
            mlp_model, X_encoded, y_encoded, 
            feature_names, class_names, sample_size
        )
        
    def extract_tree_with_ontology(self, mlp_model, X_encoded, y_encoded, 
                                 feature_names, class_names, sample_size=2000):
        """
        Implementa algoritmo Trepan-Reloaded com integração de ontologia.
        
        O algoritmo Trepan-Reloaded funciona da seguinte forma:
        1. Analisa a ontologia para extrair conhecimento de domínio
        2. Gera dados sintéticos inteligentes baseados no MLP E na ontologia
        3. Treina árvore de decisão com restrições semânticas
        4. Valida fidelidade e coerência semântica
        5. Gera explicações ricas em conhecimento de domínio
        """
        n_features = X_encoded.shape[1]
        
        # Fase 1: Análise da ontologia e extração de conhecimento de domínio
        self._extract_domain_knowledge(feature_names, class_names)
        
        # Fase 2: Geração inteligente de dados sintéticos com conhecimento de domínio
        X_synthetic = self._generate_ontology_aware_synthetic_data(
            mlp_model, X_encoded, sample_size, feature_names
        )
        
        # Fase 3: Predição do MLP sobre dados sintéticos
        y_synthetic = mlp_model.predict(X_synthetic)
        
        # Fase 4: Treinamento da árvore com restrições semânticas
        self.explainer_tree = self._train_ontology_constrained_tree(
            X_synthetic, y_synthetic, X_encoded, y_encoded, feature_names, class_names
        )
        
        # Fase 5: Validação de fidelidade e coerência semântica
        fidelity_metrics = self._calculate_ontology_aware_fidelity(
            self.explainer_tree, mlp_model, X_encoded, y_encoded, X_synthetic, y_synthetic
        )
        
        # Fase 6: Geração de regras interpretáveis com conhecimento de domínio
        # Ajuste para evitar error de discrepancia de clases
        if self.explainer_tree.classes_.shape[0] != len(class_names):
            effective_class_names = [str(c) for c in self.explainer_tree.classes_]
        else:
            effective_class_names = class_names

        rules = export_text(
            self.explainer_tree,
            feature_names=feature_names,
            class_names=effective_class_names
        )
        
        # Fase 7: Relatório detalhado de interpretabilidade com ontologia
        explanation = self._generate_ontology_aware_report(
            fidelity_metrics, X_encoded.shape[0], feature_names, class_names
        )

        return rules + explanation
    
    def _extract_domain_knowledge(self, feature_names, class_names):
        """Extrai conhecimento de domínio da ontologia"""
        if not self.ontology:
            # Se não há ontologia, cria conhecimento básico baseado nos nomes
            self._create_basic_domain_knowledge(feature_names, class_names)
            return
        
        try:
            # Extrai classes da ontologia
            ontology_classes = list(self.ontology.classes())
            ontology_properties = list(self.ontology.properties())
            
            # Mapeia features para conceitos semânticos
            self._map_features_to_concepts(feature_names, ontology_classes, ontology_properties)
            
            # Mapeia classes para conceitos semânticos
            self._map_classes_to_concepts(class_names, ontology_classes)
            
            # Extrai relacionamentos semânticos
            self._extract_semantic_relationships(ontology_classes, ontology_properties)
            
            # Inicializa domain_knowledge com estrutura completa
            self.domain_knowledge = {
                'features': self.feature_semantics,
                'classes': self.class_semantics,
                'relationships': self.domain_knowledge.get('relationships', []),
                'hierarchy': self.ontology_hierarchy,
                'property_constraints': self.property_constraints,
                'datatype_ranges': self.datatype_ranges,
                'mapping_stats': {
                    'total_features': len(feature_names),
                    'mapped_features': sum(1 for f in self.feature_semantics.values() 
                                         if f.get('ontology_concept') is not None),
                    'unmapped_features': len(self.unmapped_features),
                    'avg_mapping_score': np.mean(list(self.mapping_scores.values())) if self.mapping_scores else 0.0
                }
            }
            
            print(f"✅ Conhecimento de domínio extraído:")
            print(f"   • {len(self.feature_semantics)} features mapeadas")
            print(f"   • {len(self.class_semantics)} classes mapeadas")
            print(f"   • {len(self.domain_knowledge.get('relationships', []))} relacionamentos")
            print(f"   • {len(self.unmapped_features)} features não mapeadas")
            
        except Exception as e:
            print(f"⚠️ Erro ao extrair conhecimento de domínio: {e}")
            self._create_basic_domain_knowledge(feature_names, class_names)
    
    def _create_basic_domain_knowledge(self, feature_names, class_names):
        """Cria conhecimento básico quando não há ontologia"""
        # Analisa nomes de features para inferir conceitos
        for i, feature in enumerate(feature_names):
            concept_type = self._infer_concept_type(feature)
            self.feature_semantics[i] = {
                'name': feature,
                'concept_type': concept_type,
                'semantic_group': self._group_by_semantics(feature)
            }
        
        # Analisa nomes de classes para inferir conceitos
        for i, class_name in enumerate(class_names):
            concept_type = self._infer_concept_type(class_name)
            self.class_semantics[i] = {
                'name': class_name,
                'concept_type': concept_type,
                'semantic_group': self._group_by_semantics(class_name)
            }
        
        self.domain_knowledge = {
            'features': self.feature_semantics,
            'classes': self.class_semantics,
            'relationships': []
        }
    
    def _infer_concept_type(self, name):
        """Infere tipo de conceito baseado no nome"""
        name_lower = str(name).lower()
        
        # Padrões para diferentes tipos de conceitos
        if any(word in name_lower for word in ['age', 'idade', 'tempo', 'time', 'duration']):
            return 'temporal'
        elif any(word in name_lower for word in ['size', 'tamanho', 'length', 'width', 'height']):
            return 'dimensional'
        elif any(word in name_lower for word in ['count', 'number', 'quantidade', 'num']):
            return 'quantitative'
        elif any(word in name_lower for word in ['color', 'cor', 'type', 'tipo', 'category']):
            return 'categorical'
        elif any(word in name_lower for word in ['score', 'rating', 'pontuação', 'nota']):
            return 'evaluative'
        else:
            return 'general'
    
    def _group_by_semantics(self, name):
        """Agrupa features por semântica"""
        name_lower = str(name).lower()

        
        if any(word in name_lower for word in ['medical', 'health', 'saúde', 'medical']):
            return 'medical'
        elif any(word in name_lower for word in ['financial', 'finance', 'money', 'dinheiro']):
            return 'financial'
        elif any(word in name_lower for word in ['social', 'society', 'social']):
            return 'social'
        elif any(word in name_lower for word in ['technical', 'tech', 'system', 'sistema']):
            return 'technical'
        else:
            return 'general'
    
    def _map_features_to_concepts(self, feature_names, ontology_classes, ontology_properties):
        """Mapeia features para conceitos da ontologia com fallback inteligente"""
        for i, feature in enumerate(feature_names):
            # Busca conceito correspondente na ontologia
            matching_concept = self._find_matching_concept(feature, ontology_classes)
            matching_score = self.mapping_scores.get(feature, 0.0)
            
            # Extrai propriedades semânticas relacionadas
            semantic_props = self._extract_semantic_properties(feature, ontology_properties)
            
            # Se não encontrou match direto, tenta fallback com melhor score
            if not matching_concept:
                best_suggestion = self._find_best_fallback_match(feature, ontology_classes)
                if best_suggestion:
                    matching_concept = best_suggestion['name']
                    matching_score = best_suggestion['score']
                    self.unmapped_features.append({
                        'feature': feature,
                        'suggested_match': matching_concept,
                        'confidence': matching_score
                    })
                else:
                    self.unmapped_features.append({
                        'feature': feature,
                        'suggested_match': None,
                        'confidence': 0.0
                    })
            
            # Extrai informações adicionais do conceito se encontrado
            concept_info = {}
            if matching_concept:
                concept_obj = self._get_concept_object(matching_concept, ontology_classes)
                if concept_obj:
                    concept_info = self._extract_concept_details(concept_obj)
            
            self.feature_semantics[i] = {
                'name': feature,
                'ontology_concept': matching_concept,
                'mapping_score': matching_score,
                'semantic_properties': semantic_props,
                'concept_type': self._infer_concept_type(feature),
                'concept_info': concept_info,
                'rdfs_labels': concept_info.get('labels', []),
                'rdfs_comment': concept_info.get('comment', '')
            }
    
    def _map_classes_to_concepts(self, class_names, ontology_classes):
        """Mapeia classes para conceitos da ontologia"""
        for i, class_name in enumerate(class_names):
            matching_concept = self._find_matching_concept(class_name, ontology_classes)
            
            self.class_semantics[i] = {
                'name': class_name,
                'ontology_concept': matching_concept,
                'concept_type': self._infer_concept_type(class_name)
            }
    
    def _find_matching_concept(self, name, ontology_classes):
        """Encontra conceito correspondente na ontologia com matching robusto"""
        # Verifica cache primeiro
        cache_key = f"match_{name}"
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        name_lower = name.lower()
        name_clean = self._normalize_name(name)
        candidates = []
        
        for concept in ontology_classes:
            concept_name = concept.name.lower()
            concept_clean = self._normalize_name(concept.name)
            
            # Extrai labels RDFS (rdfs:label, skos:prefLabel)
            labels = self._extract_rdfs_labels(concept)
            label_matches = [self._normalize_name(label) for label in labels]
            
            # Score de matching (0.0 a 1.0)
            score = 0.0
            match_type = None
            
            # 1. Match exato
            if name_clean == concept_clean:
                score = 1.0
                match_type = 'exact'
            # 2. Match em labels RDFS
            elif any(name_clean == label for label in label_matches):
                score = 0.95
                match_type = 'rdfs_label'
            # 3. Match de substring (nome contém conceito ou vice-versa)
            elif name_clean in concept_clean or concept_clean in name_clean:
                overlap_ratio = min(len(name_clean), len(concept_clean)) / max(len(name_clean), len(concept_clean))
                score = 0.6 + (overlap_ratio * 0.2)
                match_type = 'substring'
            # 4. Similaridade usando difflib
            else:
                similarity = difflib.SequenceMatcher(None, name_clean, concept_clean).ratio()
                if similarity > 0.7:
                    score = similarity * 0.8
                    match_type = 'similarity'
            
            # 5. Verifica labels também com similaridade
            for label in label_matches:
                label_similarity = difflib.SequenceMatcher(None, name_clean, label).ratio()
                if label_similarity > score:
                    score = label_similarity * 0.85
                    match_type = 'label_similarity'
            
            # 6. Busca por sinônimos (remove prefixos comuns, espaços, etc)
            name_words = set(name_clean.split())
            concept_words = set(concept_clean.split())
            if name_words and concept_words:
                word_overlap = len(name_words & concept_words) / max(len(name_words), len(concept_words))
                if word_overlap > 0.5:
                    score = max(score, word_overlap * 0.7)
                    match_type = 'word_overlap'
            
            if score > 0.5:  # Threshold mínimo
                candidates.append({
                    'concept': concept,
                    'name': concept.name,
                    'score': score,
                    'match_type': match_type,
                    'labels': labels
                })
        
        # Retorna o melhor match
        if candidates:
            best_match = max(candidates, key=lambda x: x['score'])
            self.mapping_cache[cache_key] = best_match['name']
            self.mapping_scores[name] = best_match['score']
            return best_match['name']
        
        # Nenhum match encontrado
        self.mapping_cache[cache_key] = None
        return None
    
    def _normalize_name(self, name):
        """Normaliza nome para comparação"""
        # Remove caracteres especiais, espaços, convert para minúsculas
        normalized = re.sub(r'[^a-z0-9]', '', name.lower())
        # Remove prefixos comuns de ontologias (ex: has_, is_, etc)
        normalized = re.sub(r'^(has|is|contains|hasvalue|hasproperty)', '', normalized)
        return normalized
    
    def _extract_rdfs_labels(self, concept):
        """Extrai labels RDFS de um conceito (rdfs:label, skos:prefLabel, etc)"""
        labels = []
        try:
            # rdfs:label
            if hasattr(concept, 'label'):
                if isinstance(concept.label, list):
                    labels.extend([str(l) for l in concept.label])
                else:
                    labels.append(str(concept.label))
            
            # skos:prefLabel
            if hasattr(concept, 'prefLabel'):
                if isinstance(concept.prefLabel, list):
                    labels.extend([str(l) for l in concept.prefLabel])
                else:
                    labels.append(str(concept.prefLabel))
            
            # rdfs:comment (pode conter descrições úteis)
            if hasattr(concept, 'comment'):
                comment = str(concept.comment) if not isinstance(concept.comment, list) else str(concept.comment[0])
                # Extrai palavras-chave do comentário
                words = re.findall(r'\b[a-z]{3,}\b', comment.lower())
                labels.extend(words[:5])  # Limita a 5 palavras-chave
                
        except Exception:
            pass
        
        return [l.lower().strip() for l in labels if l]
    
    def _extract_semantic_properties(self, feature_name, ontology_properties):
        """Extrai propriedades semânticas relacionadas à feature"""
        properties = []
        feature_lower = feature_name.lower()
        
        for prop in ontology_properties:
            prop_name = prop.name.lower()
            if feature_lower in prop_name or prop_name in feature_lower:
                properties.append(prop.name)
        
        return properties
    
    def _extract_semantic_relationships(self, ontology_classes, ontology_properties):
        """Extrai relacionamentos semânticos da ontologia com mais detalhes"""
        relationships = []
        hierarchy_map = defaultdict(list)
        
        # Extrai relacionamentos hierárquicos (subclasse de)
        for concept in ontology_classes:
            concept_name = concept.name
            if hasattr(concept, 'is_a'):
                for parent in concept.is_a:
                    parent_name = parent.name if hasattr(parent, 'name') else str(parent)
                    hierarchy_map[parent_name].append(concept_name)
                    relationships.append({
                        'type': 'hierarchy',
                        'child': concept_name,
                        'parent': parent_name,
                        'relation': 'subClassOf'
                    })
            
            # Extrai equivalent classes
            if hasattr(concept, 'equivalent_to'):
                for equiv in concept.equivalent_to:
                    equiv_name = equiv.name if hasattr(equiv, 'name') else str(equiv)
                    relationships.append({
                        'type': 'equivalence',
                        'class1': concept_name,
                        'class2': equiv_name,
                        'relation': 'equivalentClass'
                    })
            
            # Extrai disjoint classes
            if hasattr(concept, 'disjoint_with'):
                for disjoint in concept.disjoint_with:
                    disjoint_name = disjoint.name if hasattr(disjoint, 'name') else str(disjoint)
                    relationships.append({
                        'type': 'disjoint',
                        'class1': concept_name,
                        'class2': disjoint_name,
                        'relation': 'disjointWith'
                    })
        
        # Extrai relacionamentos de propriedade com mais detalhes
        for prop in ontology_properties:
            prop_name = prop.name
            prop_info = {
                'type': 'property',
                'property': prop_name,
                'relation': 'property'
            }
            
            # Domain
            if hasattr(prop, 'domain'):
                domain_list = prop.domain if isinstance(prop.domain, list) else [prop.domain]
                domains = []
                for domain in domain_list:
                    domain_name = domain.name if hasattr(domain, 'name') else str(domain)
                    domains.append(domain_name)
                prop_info['domain'] = domains[0] if len(domains) == 1 else domains
                prop_info['domains'] = domains
            
            # Range
            if hasattr(prop, 'range'):
                range_val = prop.range
                if hasattr(range_val, 'name'):
                    prop_info['range'] = range_val.name
                    prop_info['range_type'] = 'class'
                else:
                    # Pode ser um datatype (xsd:integer, xsd:float, etc)
                    range_str = str(range_val)
                    prop_info['range'] = range_str
                    prop_info['range_type'] = 'datatype'
                    # Extrai informações do datatype
                    self._extract_datatype_info(range_str, prop_name)
                prop_info['range_obj'] = range_val
            
            relationships.append(prop_info)
            
            # Guarda restrições de propriedade
            if 'domain' in prop_info:
                if prop_name not in self.property_constraints:
                    self.property_constraints[prop_name] = {}
                self.property_constraints[prop_name]['domain'] = prop_info.get('domain')
                self.property_constraints[prop_name]['range'] = prop_info.get('range')
                self.property_constraints[prop_name]['range_type'] = prop_info.get('range_type', 'class')
            
            # Verifica propriedades inversas
            if hasattr(prop, 'inverse_property'):
                inverse = prop.inverse_property
                relationships.append({
                    'type': 'inverse_property',
                    'property1': prop_name,
                    'property2': inverse.name if hasattr(inverse, 'name') else str(inverse),
                    'relation': 'inverseOf'
                })
        
        self.domain_knowledge['relationships'] = relationships
        self.ontology_hierarchy = dict(hierarchy_map)
    
    def _extract_datatype_info(self, datatype_str, prop_name):
        """Extrai informações de datatypes OWL (xsd:integer, xsd:float, etc)"""
        # Padrões comuns de datatypes
        if 'integer' in datatype_str.lower() or 'int' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': None,
                'max': None
            }
        elif 'float' in datatype_str.lower() or 'double' in datatype_str.lower() or 'decimal' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'float',
                'min': None,
                'max': None
            }
        elif 'boolean' in datatype_str.lower() or 'bool' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'boolean',
                'min': 0,
                'max': 1
            }
        elif 'nonNegativeInteger' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': 0,
                'max': None
            }
        elif 'positiveInteger' in datatype_str.lower():
            self.datatype_ranges[prop_name] = {
                'type': 'integer',
                'min': 1,
                'max': None
            }
    
    def _get_concept_object(self, concept_name, ontology_classes):
        """Retorna o objeto do conceito dado o nome"""
        for concept in ontology_classes:
            if concept.name == concept_name:
                return concept
        return None
    
    def _extract_concept_details(self, concept):
        """Extrai detalhes completos de um conceito"""
        details = {
            'labels': self._extract_rdfs_labels(concept),
            'comment': '',
            'properties': []
        }
        
        # Comentário RDFS
        try:
            if hasattr(concept, 'comment'):
                if isinstance(concept.comment, list):
                    details['comment'] = ' '.join([str(c) for c in concept.comment])
                else:
                    details['comment'] = str(concept.comment)
        except:
            pass
        
        return details
    
    def _find_best_fallback_match(self, name, ontology_classes, min_score=0.4):
        """Encontra melhor match mesmo com score baixo (fallback)"""
        name_clean = self._normalize_name(name)
        candidates = []
        
        for concept in ontology_classes:
            concept_clean = self._normalize_name(concept.name)
            labels = self._extract_rdfs_labels(concept)
            
            # Calcula similaridade básica
            similarity = difflib.SequenceMatcher(None, name_clean, concept_clean).ratio()
            if similarity >= min_score:
                candidates.append({
                    'name': concept.name,
                    'score': similarity,
                    'type': 'name_similarity'
                })
            
            # Verifica labels
            for label in labels:
                label_norm = self._normalize_name(label)
                label_sim = difflib.SequenceMatcher(None, name_clean, label_norm).ratio()
                if label_sim >= min_score:
                    candidates.append({
                        'name': concept.name,
                        'score': label_sim,
                        'type': 'label_similarity'
                    })
        
        if candidates:
            return max(candidates, key=lambda x: x['score'])
        return None
    
    def _generate_ontology_aware_synthetic_data(self, mlp_model, X_encoded, sample_size, feature_names):
        """
        Gera dados sintéticos inteligentes considerando conhecimento de domínio.
        
        Em vez de geração puramente aleatória, usa:
        - Restrições semânticas baseadas na ontologia
        - Relacionamentos entre features
        - Padrões de domínio conhecidos
        """
        n_features = X_encoded.shape[1]
        
        # 1. Amostragem baseada em densidade (40% dos dados)
        density_samples = int(sample_size * 0.4)
        X_density = self._sample_by_density(X_encoded, density_samples)
        
        # 2. Geração em regiões de alta incerteza (25% dos dados)
        uncertainty_samples = int(sample_size * 0.25)
        X_uncertainty = self._sample_uncertainty_regions(mlp_model, X_encoded, uncertainty_samples)
        
        # 3. Geração baseada em conhecimento de domínio (25% dos dados)
        domain_samples = int(sample_size * 0.25)
        X_domain = self._sample_by_domain_knowledge(X_encoded, domain_samples, feature_names)
        
        # 4. Geração aleatória balanceada (10% dos dados)
        random_samples = sample_size - density_samples - uncertainty_samples - domain_samples
        X_random = np.random.uniform(
            low=np.min(X_encoded, axis=0),
            high=np.max(X_encoded, axis=0),
            size=(random_samples, n_features)
        )
        
        # Combinar todos os tipos de amostras
        X_synthetic = np.vstack([X_density, X_uncertainty, X_domain, X_random])
        
        return X_synthetic
    
    def _sample_by_domain_knowledge(self, X_encoded, n_samples, feature_names):
        """Gera amostras baseadas em conhecimento de domínio usando relacionamentos ontológicos"""
        n_features = X_encoded.shape[1]
        X_domain = np.zeros((n_samples, n_features))
        
        # Identifica features relacionadas através de propriedades ontológicas
        feature_relationships = self._identify_related_features(feature_names)
        
        for i in range(n_samples):
            # Determina valores baseados em grupos relacionados primeiro
            assigned_features = set()
            
            # Processa grupos de features relacionadas
            for group in feature_relationships:
                if not group:  # Skip empty groups
                    continue
                
                # Gera valores para features relacionadas considerando restrições
                group_values = self._generate_related_feature_values(
                    group, X_encoded, feature_names
                )
                
                for feat_idx, value in group_values.items():
                    if feat_idx not in assigned_features:
                        X_domain[i, feat_idx] = value
                        assigned_features.add(feat_idx)
            
            # Para features não relacionadas ou não em grupos, gera individualmente
            for j in range(n_features):
                if j in assigned_features:
                    continue
                    
                feature_info = self.feature_semantics.get(j, {})
                concept_type = feature_info.get('concept_type', 'general')
                
                # Usa informações de datatype se disponível
                datatype_info = None
                ontology_concept = feature_info.get('ontology_concept')
                if ontology_concept:
                    # Verifica se há restrições de datatype nas propriedades
                    for prop_name, prop_constraints in self.property_constraints.items():
                        if prop_name.lower() in feature_names[j].lower():
                            range_type = prop_constraints.get('range_type')
                            if range_type == 'datatype':
                                datatype_info = self.datatype_ranges.get(prop_name)
                                break
                
                # Gera valor baseado no tipo de conceito e restrições
                value = self._generate_value_by_concept_type(
                    j, concept_type, X_encoded, datatype_info
                )
                X_domain[i, j] = value
        
        return X_domain
    
    def _identify_related_features(self, feature_names):
        """Identifica features relacionadas através de propriedades ontológicas"""
        groups = []
        processed = set()
        
        # Agrupa features que compartilham propriedades ontológicas
        for prop_name, prop_constraints in self.property_constraints.items():
            domain = prop_constraints.get('domain')
            if not domain:
                continue
                
            # Encontra features relacionadas a este domínio
            related_indices = []
            for i, feature in enumerate(feature_names):
                if i in processed:
                    continue
                    
                feature_info = self.feature_semantics.get(i, {})
                ontology_concept = feature_info.get('ontology_concept', '')
                
                # Verifica se a feature pertence ao domínio da propriedade
                if isinstance(domain, str):
                    if domain.lower() in feature.lower() or feature.lower() in domain.lower():
                        related_indices.append(i)
                        processed.add(i)
                elif isinstance(domain, list):
                    for d in domain:
                        if d.lower() in feature.lower() or feature.lower() in d.lower():
                            related_indices.append(i)
                            processed.add(i)
                            break
            
            if len(related_indices) > 1:
                groups.append(related_indices)
        
        # Se não encontrou grupos, agrupa por hierarquia ontológica
        if not groups:
            for i, feature_info in self.feature_semantics.items():
                ontology_concept = feature_info.get('ontology_concept')
                if ontology_concept and ontology_concept in self.ontology_hierarchy:
                    # Features relacionadas hierarquicamente
                    related = []
                    for j, other_info in self.feature_semantics.items():
                        if i != j:
                            other_concept = other_info.get('ontology_concept')
                            if other_concept in self.ontology_hierarchy.get(ontology_concept, []):
                                related.append(j)
                    if related:
                        groups.append([i] + related)
        
        return groups if groups else [[i] for i in range(len(feature_names))]
    
    def _generate_related_feature_values(self, feature_indices, X_encoded, feature_names):
        """Gera valores para features relacionadas respeitando restrições ontológicas"""
        values = {}
        
        # Primeiro, identifica as restrições ontológicas para este grupo
        constraints = []
        for idx in feature_indices:
            feature_info = self.feature_semantics.get(idx, {})
            ontology_concept = feature_info.get('ontology_concept')
            if ontology_concept:
                constraints.append({
                    'index': idx,
                    'concept': ontology_concept,
                    'concept_type': feature_info.get('concept_type', 'general')
                })
        
        # Gera valores considerando relacionamentos
        for idx in feature_indices:
            feature_info = self.feature_semantics.get(idx, {})
            concept_type = feature_info.get('concept_type', 'general')
            
            # Gera valor básico
            value = self._generate_value_by_concept_type(idx, concept_type, X_encoded, None)
            
            # Ajusta baseado em relacionamentos com outras features do grupo
            # Por exemplo, se duas features são inversas ou relacionadas
            for other_idx in feature_indices:
                if other_idx == idx:
                    continue
                    
                # Aplica restrições de relacionamento se existirem
                # (implementação simplificada - pode ser expandida)
                pass
            
            values[idx] = value
        
        return values
    
    def _generate_value_by_concept_type(self, feature_idx, concept_type, X_encoded, datatype_info):
        """Gera valor baseado no tipo de conceito e restrições de datatype"""
        col_data = X_encoded[:, feature_idx]
        col_min = np.min(col_data)
        col_max = np.max(col_data)
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        
        # Aplica restrições de datatype se disponível
        if datatype_info:
            dtype_type = datatype_info.get('type')
            dtype_min = datatype_info.get('min')
            dtype_max = datatype_info.get('max')
            
            if dtype_min is not None:
                col_min = max(col_min, dtype_min)
            if dtype_max is not None:
                col_max = min(col_max, dtype_max)
        
        # Gera valor baseado no tipo de conceito
        if concept_type == 'temporal':
            value = np.random.normal(col_mean, col_std * 0.5)
            value = max(col_min, min(col_max, value))
        elif concept_type == 'dimensional':
            # Valores dimensionais tendem a ser positivos
            if col_min >= 0:
                value = np.random.exponential(max(col_mean, 1))
                value = min(col_max, value)
            else:
                value = np.random.normal(col_mean, col_std * 0.7)
        elif concept_type == 'quantitative':
            value = np.random.normal(col_mean, col_std)
            value = max(col_min, min(col_max, value))
        elif concept_type == 'categorical':
            unique_values = np.unique(col_data)
            value = np.random.choice(unique_values)
        else:
            value = np.random.uniform(col_min, col_max)
        
        # Aplica restrições de datatype novamente
        if datatype_info:
            dtype_type = datatype_info.get('type')
            if dtype_type == 'integer':
                value = int(round(value))
            elif dtype_type == 'boolean':
                value = 1.0 if value > 0.5 else 0.0
        
        return value
    
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
    
    def _train_ontology_constrained_tree(self, X_synthetic, y_synthetic, X_real, y_real, feature_names, class_names):
    # Usar los parámetros pasados al constructor (o sus valores por defecto)
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            criterion='gini'
        )
        tree.fit(X_synthetic, y_synthetic)
        self._validate_semantic_coherence(tree, feature_names, class_names)
        return tree
    
    def _validate_semantic_coherence(self, tree, feature_names, class_names):
        """Valida coerência semântica da árvore gerada com validações robustas"""
        tree_structure = tree.tree_
        validation_issues = []
        validation_warnings = []
        
        # Analisa cada nó de decisão
        for i in range(tree_structure.node_count):
            if tree_structure.children_left[i] != -1:  # Não é folha
                feature_idx = tree_structure.feature[i]
                threshold = tree_structure.threshold[i]
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                
                # Verifica se a decisão faz sentido semanticamente
                feature_info = self.feature_semantics.get(feature_idx, {})
                concept_type = feature_info.get('concept_type', 'general')
                ontology_concept = feature_info.get('ontology_concept')
                
                # 1. Validações por tipo de conceito
                if concept_type == 'temporal':
                    if threshold < 0:
                        validation_warnings.append(
                            f"⚠️ Threshold temporal negativo em {feature_name} ({threshold:.3f})"
                        )
                elif concept_type == 'dimensional':
                    if threshold < 0:
                        validation_warnings.append(
                            f"⚠️ Threshold dimensional negativo em {feature_name} ({threshold:.3f})"
                        )
                
                # 2. Validações baseadas em restrições de datatype
                if ontology_concept:
                    # Verifica restrições de propriedades relacionadas
                    for prop_name, prop_constraints in self.property_constraints.items():
                        if prop_name.lower() in feature_name.lower():
                            range_type = prop_constraints.get('range_type')
                            if range_type == 'datatype':
                                datatype_info = self.datatype_ranges.get(prop_name)
                                if datatype_info:
                                    dtype_min = datatype_info.get('min')
                                    dtype_max = datatype_info.get('max')
                                    if dtype_min is not None and threshold < dtype_min:
                                        validation_issues.append(
                                            f"❌ Threshold {threshold:.3f} em {feature_name} viola min={dtype_min}"
                                        )
                                    if dtype_max is not None and threshold > dtype_max:
                                        validation_issues.append(
                                            f"❌ Threshold {threshold:.3f} em {feature_name} viola max={dtype_max}"
                                        )
                
                # 3. Validações de relacionamentos ontológicos
                if ontology_concept:
                    # Verifica se há classes disjoint que podem causar conflitos
                    relationships = self.domain_knowledge.get('relationships', [])
                    for rel in relationships:
                        if rel.get('type') == 'disjoint':
                            class1 = rel.get('class1')
                            class2 = rel.get('class2')
                            # Se a feature mapeia para uma classe disjoint, verifica se não há contradição
                            # (implementação simplificada - pode ser expandida)
                            pass
                
                # 4. Validação de ranges de valores baseados em dados reais
                if feature_idx < tree_structure.n_features:
                    # Valida se o threshold está dentro de um range razoável
                    # (pode ser expandido para usar estatísticas dos dados)
                    pass
        
        # Reporta validações
        if validation_issues:
            print(f"\n🚨 Problemas de coerência semântica encontrados ({len(validation_issues)}):")
            for issue in validation_issues[:10]:  # Limita a 10 para não sobrecarregar
                print(f"  {issue}")
            if len(validation_issues) > 10:
                print(f"  ... e mais {len(validation_issues) - 10} problemas")
        
        if validation_warnings:
            print(f"\n⚠️ Avisos de coerência semântica ({len(validation_warnings)}):")
            for warning in validation_warnings[:10]:
                print(f"  {warning}")
            if len(validation_warnings) > 10:
                print(f"  ... e mais {len(validation_warnings) - 10} avisos")
        
        return {
            'issues': validation_issues,
            'warnings': validation_warnings,
            'total_nodes': tree_structure.node_count,
            'validated_nodes': sum(1 for i in range(tree_structure.node_count) 
                                 if tree_structure.children_left[i] != -1)
        }
    
    def _calculate_ontology_aware_fidelity(self, tree, mlp_model, X_real, y_real, 
                                          X_synthetic, y_synthetic):
        """Calcula fidelidade considerando conhecimento de domínio"""
        # Fidelidade básica
        fidelity_real = self._calculate_fidelity(tree, mlp_model, X_real, y_real)
        fidelity_synthetic = self._calculate_fidelity(tree, mlp_model, X_synthetic, y_synthetic)
        
        # Fidelidade por grupo semântico
        semantic_fidelity = self._calculate_semantic_group_fidelity(tree, mlp_model, X_real, y_real)
        
        return {
            'overall_real': fidelity_real,
            'overall_synthetic': fidelity_synthetic,
            'semantic_groups': semantic_fidelity,
            'n_real_samples': len(X_real),
            'n_synthetic_samples': len(X_synthetic)
        }
    
    def _calculate_semantic_group_fidelity(self, tree, mlp_model, X_real, y_real):
        """Calcula fidelidade por grupo semântico"""
        semantic_fidelity = {}
        
        # Agrupa features por semântica
        semantic_groups = {}
        for feature_idx, feature_info in self.feature_semantics.items():
            semantic_group = feature_info.get('semantic_group', 'general')
            if semantic_group not in semantic_groups:
                semantic_groups[semantic_group] = []
            semantic_groups[semantic_group].append(feature_idx)
        
        # Calcula fidelidade para cada grupo
        for group_name, feature_indices in semantic_groups.items():
            if feature_indices:
                # Cria subconjunto dos dados com apenas features deste grupo
                X_group = X_real[:, feature_indices]
                
                # Treina árvore temporária apenas com este grupo
                temp_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
                temp_tree.fit(X_group, y_real)
                
                # Calcula fidelidade
                y_tree_pred = temp_tree.predict(X_group)
                y_mlp_pred = mlp_model.predict(X_real)  # MLP usa todas as features
                
                # Fidelidade aproximada (pode não ser exata devido à diferença de features)
                fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
                semantic_fidelity[group_name] = fidelity
        
        return semantic_fidelity
    
    def _calculate_fidelity(self, tree, mlp_model, X_data, y_data):
        """Calcula fidelidade da árvore em relação ao MLP"""
        # Predições da árvore
        y_tree_pred = tree.predict(X_data)
        
        # Predições do MLP
        y_mlp_pred = mlp_model.predict(X_data)
        
        # Fidelidade = concordância entre árvore e MLP
        fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
        
        return fidelity
    
    def _generate_ontology_aware_report(self, fidelity_metrics, n_real_samples, 
                                       feature_names, class_names):
        """Gera relatório detalhado de interpretabilidade com ontologia"""
        report = f"\n{'='*70}\n"
        report += f"🧠 RELATÓRIO DE INTERPRETABILIDADE TREPAN-RELOADED\n"
        report += f"{'='*70}\n\n"
        
        report += f"🎯 Fidelidade da Árvore ao MLP:\n"
        report += f"   • Sobre dados sintéticos: {fidelity_metrics['overall_synthetic']:.3f} ({fidelity_metrics['overall_synthetic']*100:.1f}%)\n"
        report += f"   • Sobre dados reais: {fidelity_metrics['overall_real']:.3f} ({fidelity_metrics['overall_real']*100:.1f}%)\n"
        report += f"   • Amostras reais validadas: {n_real_samples}\n\n"
        
        # Fidelidade por grupo semântico
        if fidelity_metrics['semantic_groups']:
            report += f"📊 Fidelidade por Grupo Semântico:\n"
            for group, fidelity in fidelity_metrics['semantic_groups'].items():
                report += f"   • {group}: {fidelity:.3f} ({fidelity*100:.1f}%)\n"
            report += f"\n"
        
        # Conhecimento de domínio extraído
        report += f"🧠 Conhecimento de Domínio Incorporado:\n"
        report += f"   • Features analisadas: {len(self.feature_semantics)}\n"
        report += f"   • Classes analisadas: {len(self.class_semantics)}\n"
        report += f"   • Relacionamentos semânticos: {len(self.domain_knowledge.get('relationships', []))}\n"
        
        # Features mapeadas vs não mapeadas
        mapped_features = sum(1 for f in self.feature_semantics.values() 
                             if f.get('ontology_concept') is not None)
        unmapped_count = len(self.unmapped_features)
        if unmapped_count > 0:
            report += f"   • Features mapeadas para ontologia: {mapped_features}/{len(self.feature_semantics)}\n"
            report += f"   • Features não mapeadas: {unmapped_count}\n"
            if unmapped_count <= 5:
                for unmapped in self.unmapped_features:
                    if unmapped.get('suggested_match'):
                        report += f"     - '{unmapped['feature']}' (sugestão: {unmapped['suggested_match']}, confiança: {unmapped['confidence']:.2f})\n"
                    else:
                        report += f"     - '{unmapped['feature']}' (sem match encontrado)\n"
        
        # Qualidade dos mapeamentos
        if self.mapping_scores:
            avg_score = np.mean(list(self.mapping_scores.values()))
            report += f"   • Score médio de mapeamento: {avg_score:.2f}\n"
        
        # Hierarquias e restrições
        if self.ontology_hierarchy:
            report += f"   • Hierarquias ontológicas extraídas: {len(self.ontology_hierarchy)}\n"
        if self.property_constraints:
            report += f"   • Restrições de propriedades: {len(self.property_constraints)}\n"
        if self.datatype_ranges:
            report += f"   • Datatypes identificados: {len(self.datatype_ranges)}\n"
        
        report += f"\n"
        
        # Interpretação da fidelidade
        avg_fidelity = (fidelity_metrics['overall_synthetic'] + fidelity_metrics['overall_real']) / 2
        if avg_fidelity > 0.9:
            report += f"🌟 EXCELENTE interpretabilidade! A árvore representa fielmente o MLP com conhecimento de domínio.\n"
        elif avg_fidelity > 0.8:
            report += f"✅ MUITO BOA interpretabilidade. A árvore captura bem os padrões do MLP.\n"
        elif avg_fidelity > 0.7:
            report += f"👍 BOA interpretabilidade. A árvore representa adequadamente o MLP.\n"
        else:
            report += f"⚠️ Interpretabilidade MODERADA. Alguns padrões podem não estar representados.\n"
        
        report += f"\n💡 Vantagens do Trepan-Reloaded:\n"
        report += f"   • Incorpora conhecimento de domínio da ontologia\n"
        report += f"   • Gera explicações semanticamente coerentes\n"
        report += f"   • Considera relacionamentos entre conceitos\n"
        report += f"   • Valida coerência semântica das decisões\n\n"
        
        report += f"🌳 Próximos passos:\n"
        report += f"   • Use '🌳 Visualizar Árvore' para explorar as regras\n"
        report += f"   • Use '💡 Explicações Semânticas' para análises detalhadas\n"
        report += f"   • Use '📊 Comparar Modelos' para análise comparativa\n"
        
        return report
    
    def generate_semantic_explanations(self, feature_names, class_names):
        """
        Gera explicações semânticas ricas baseadas no conhecimento de domínio.
        
        Converte a árvore de decisão em explicações que incorporam:
        - Conceitos de domínio
        - Relacionamentos semânticos
        - Contexto do problema
        """
        if self.explainer_tree is None:
            return "❌ Nenhuma árvore foi gerada ainda. Execute extract_tree_with_ontology primeiro."
        
        rules = []
        self._extract_semantic_rules_recursive(
            self.explainer_tree.tree_, 0, [], rules, feature_names, class_names
        )
        
        # Formata regras em linguagem semântica rica
        semantic_explanations = self._format_semantic_rules(rules, class_names)
        
        return semantic_explanations
    
    def _extract_semantic_rules_recursive(self, tree, node_id, conditions, rules, 
                                        feature_names, class_names):
        """Extrai regras recursivamente da árvore com contexto semântico"""
        from sklearn.tree import _tree
        
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Nó folha - adiciona regra completa com contexto semântico
            class_id = np.argmax(tree.value[node_id][0])
            class_name = class_names[class_id]
            confidence = tree.value[node_id][0][class_id] / np.sum(tree.value[node_id][0])
            samples = tree.n_node_samples[node_id]
            
            # Adiciona contexto semântico da classe
            class_info = self.class_semantics.get(class_id, {})
            
            rules.append({
                'conditions': conditions.copy(),
                'class': class_name,
                'confidence': confidence,
                'samples': samples,
                'class_context': class_info,
                'semantic_interpretation': self._generate_semantic_interpretation(conditions, class_name)
            })
            return
        
        # Nó interno - continua recursão com contexto semântico
        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        
        # Adiciona contexto semântico da feature
        feature_info = self.feature_semantics.get(tree.feature[node_id], {})
        
        # Condição para filho esquerdo (<=)
        left_conditions = conditions + [{
            'feature': feature_name,
            'operator': '≤',
            'threshold': threshold,
            'feature_context': feature_info,
            'semantic_meaning': self._get_semantic_meaning(feature_name, '≤', threshold)
        }]
        self._extract_semantic_rules_recursive(
            tree, tree.children_left[node_id], left_conditions, rules, feature_names, class_names
        )
        
        # Condição para filho direito (>)
        right_conditions = conditions + [{
            'feature': feature_name,
            'operator': '>',
            'threshold': threshold,
            'feature_context': feature_info,
            'semantic_meaning': self._get_semantic_meaning(feature_name, '>', threshold)
        }]
        self._extract_semantic_rules_recursive(
            tree, tree.children_right[node_id], right_conditions, rules, feature_names, class_names
        )
    
    def _get_semantic_meaning(self, feature_name, operator, threshold):
        """Gera significado semântico de uma condição"""
        feature_info = None
        for info in self.feature_semantics.values():
            if info.get('name') == feature_name:
                feature_info = info
                break
        
        if not feature_info:
            return f"{feature_name} {operator} {threshold:.3f}"
        
        concept_type = feature_info.get('concept_type', 'general')
        
        # Gera significado baseado no tipo de conceito
        if concept_type == 'temporal':
            if operator == '≤':
                return f"até {threshold:.1f} unidades de tempo"
            else:
                return f"após {threshold:.1f} unidades de tempo"
        elif concept_type == 'dimensional':
            if operator == '≤':
                return f"até {threshold:.1f} unidades de medida"
            else:
                return f"maior que {threshold:.1f} unidades de medida"
        elif concept_type == 'quantitative':
            if operator == '≤':
                return f"até {threshold:.1f}"
            else:
                return f"maior que {threshold:.1f}"
        elif concept_type == 'categorical':
            return f"categoria {threshold:.0f}"
        else:
            return f"{feature_name} {operator} {threshold:.3f}"
    
    def _generate_semantic_interpretation(self, conditions, class_name):
        """Gera interpretação semântica de um conjunto de condições"""
        if not conditions:
            return f"Classificação padrão como '{class_name}'"
        
        # Agrupa condições por tipo semântico
        temporal_conditions = []
        dimensional_conditions = []
        quantitative_conditions = []
        categorical_conditions = []
        
        for condition in conditions:
            feature_context = condition.get('feature_context', {})
            concept_type = feature_context.get('concept_type', 'general')
            
            if concept_type == 'temporal':
                temporal_conditions.append(condition)
            elif concept_type == 'dimensional':
                dimensional_conditions.append(condition)
            elif concept_type == 'quantitative':
                quantitative_conditions.append(condition)
            elif concept_type == 'categorical':
                categorical_conditions.append(condition)
        
        # Gera interpretação semântica
        interpretation_parts = []
        
        if temporal_conditions:
            interpretation_parts.append("considerando aspectos temporais")
        if dimensional_conditions:
            interpretation_parts.append("considerando dimensões físicas")
        if quantitative_conditions:
            interpretation_parts.append("considerando valores quantitativos")
        if categorical_conditions:
            interpretation_parts.append("considerando categorias")
        
        if interpretation_parts:
            return f"Classificação como '{class_name}' " + " e ".join(interpretation_parts)
        else:
            return f"Classificação como '{class_name}' baseada em múltiplos critérios"
    
    def _format_semantic_rules(self, rules, class_names):
        """Formata regras em linguagem semântica rica"""
        if not rules:
            return "❌ Nenhuma regra foi extraída da árvore."
        
        formatted_rules = f"\n{'='*70}\n"
        formatted_rules += f"🧠 EXPLICAÇÕES SEMÂNTICAS TREPAN-RELOADED\n"
        formatted_rules += f"{'='*70}\n\n"
        
        formatted_rules += f"📋 O modelo toma decisões baseado em conhecimento de domínio:\n\n"
        
        for i, rule in enumerate(rules, 1):
            formatted_rules += f"🔹 Regra Semântica {i}:\n"
            
            if rule['conditions']:
                # Formata condições com contexto semântico
                condition_texts = []
                for condition in rule['conditions']:
                    semantic_meaning = condition.get('semantic_meaning', 
                                                   f"{condition['feature']} {condition['operator']} {condition['threshold']:.3f}")
                    condition_texts.append(semantic_meaning)
                
                formatted_rules += f"   Se {' E '.join(condition_texts)}\n"
            else:
                formatted_rules += f"   Se nenhuma condição específica\n"
            
            formatted_rules += f"   Então: Classe '{rule['class']}'\n"
            formatted_rules += f"   Confiança: {rule['confidence']:.1%}\n"
            formatted_rules += f"   Amostras: {rule['samples']}\n"
            formatted_rules += f"   Interpretação: {rule['semantic_interpretation']}\n\n"
        
        # Análise de complexidade semântica
        avg_conditions = np.mean([len(rule['conditions']) for rule in rules])
        formatted_rules += f"📊 Análise de Complexidade Semântica:\n"
        formatted_rules += f"   • Total de regras: {len(rules)}\n"
        formatted_rules += f"   • Condições médias por regra: {avg_conditions:.1f}\n"
        
        # Análise de tipos de conceitos utilizados
        concept_types_used = set()
        for rule in rules:
            for condition in rule['conditions']:
                feature_context = condition.get('feature_context', {})
                concept_type = feature_context.get('concept_type', 'general')
                concept_types_used.add(concept_type)
        
        formatted_rules += f"   • Tipos de conceitos utilizados: {', '.join(concept_types_used)}\n"
        
        if avg_conditions <= 2:
            formatted_rules += f"   ✅ Regras simples e semanticamente claras\n"
        elif avg_conditions <= 4:
            formatted_rules += f"   ⚠️ Regras moderadamente complexas\n"
        else:
            formatted_rules += f"   ❌ Regras complexas, considere simplificar\n"
        
        formatted_rules += f"\n💡 Vantagens das Explicações Semânticas:\n"
        formatted_rules += f"   • Incorporam conhecimento de domínio\n"
        formatted_rules += f"   • Fornecem contexto semântico rico\n"
        formatted_rules += f"   • Consideram relacionamentos entre conceitos\n"
        formatted_rules += f"   • São mais compreensíveis para especialistas do domínio\n"
        
        return formatted_rules
    
    def export_tree_image(self, feature_names, class_names, output_file="tree_visualization.png", open_image=True):
        """Exporta a árvore gerada como imagem com contexto semântico"""
        if self.explainer_tree is None:
            raise ValueError("Árvore ainda não foi gerada. Execute extract_tree_with_ontology primeiro.")

        # Adiciona contexto semântico aos nomes das features
        enhanced_feature_names = []
        for i, feature_name in enumerate(feature_names):
            feature_info = self.feature_semantics.get(i, {})
            concept_type = feature_info.get('concept_type', 'general')
            semantic_group = feature_info.get('semantic_group', 'general')
            
            enhanced_name = f"{feature_name}\n[{concept_type}]"
            enhanced_feature_names.append(enhanced_name)

        dot_data = export_graphviz(
            self.explainer_tree,
            out_file=None,
            feature_names=enhanced_feature_names,
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

        return f"Árvore com contexto semântico exportada como imagem para {file_path}"
    
    def save_domain_knowledge(self, file_path):
        """Salva conhecimento de domínio extraído em arquivo para reutilização"""
        try:
            save_data = {
                'feature_semantics': self.feature_semantics,
                'class_semantics': self.class_semantics,
                'domain_knowledge': self.domain_knowledge,
                'mapping_cache': self.mapping_cache,
                'mapping_scores': self.mapping_scores,
                'ontology_hierarchy': self.ontology_hierarchy,
                'property_constraints': self.property_constraints,
                'datatype_ranges': self.datatype_ranges,
                'unmapped_features': self.unmapped_features,
                'unmapped_classes': self.unmapped_classes
            }
            
            # Converte objetos owlready2 para strings onde necessário
            serializable_data = self._make_serializable(save_data)
            
            with open(file_path, 'wb') as f:
                pickle.dump(serializable_data, f)
            
            print(f"✅ Conhecimento de domínio salvo em {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar conhecimento de domínio: {e}")
            return False
    
    def load_domain_knowledge(self, file_path):
        """Carrega conhecimento de domínio de arquivo"""
        try:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.feature_semantics = loaded_data.get('feature_semantics', {})
            self.class_semantics = loaded_data.get('class_semantics', {})
            self.domain_knowledge = loaded_data.get('domain_knowledge', {})
            self.mapping_cache = loaded_data.get('mapping_cache', {})
            self.mapping_scores = loaded_data.get('mapping_scores', {})
            self.ontology_hierarchy = loaded_data.get('ontology_hierarchy', {})
            self.property_constraints = loaded_data.get('property_constraints', {})
            self.datatype_ranges = loaded_data.get('datatype_ranges', {})
            self.unmapped_features = loaded_data.get('unmapped_features', [])
            self.unmapped_classes = loaded_data.get('unmapped_classes', [])
            
            print(f"✅ Conhecimento de domínio carregado de {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar conhecimento de domínio: {e}")
            return False
    
    def _make_serializable(self, data):
        """Converte dados para formato serializável (remove objetos não serializáveis)"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            # Tenta converter para string
            try:
                return str(data)
            except:
                return None
    
    def get_mapping_statistics(self):
        """Retorna estatísticas sobre o mapeamento de features/classes"""
        stats = {
            'features': {
                'total': len(self.feature_semantics),
                'mapped': sum(1 for f in self.feature_semantics.values() 
                             if f.get('ontology_concept') is not None),
                'unmapped': len(self.unmapped_features),
                'avg_score': np.mean(list(self.mapping_scores.values())) if self.mapping_scores else 0.0
            },
            'classes': {
                'total': len(self.class_semantics),
                'mapped': sum(1 for c in self.class_semantics.values() 
                             if c.get('ontology_concept') is not None),
                'unmapped': len(self.unmapped_classes),
                'avg_score': np.mean([v for k, v in self.mapping_scores.items() 
                                    if k in [c.get('name') for c in self.class_semantics.values()]]) 
                           if self.mapping_scores else 0.0
            },
            'relationships': {
                'hierarchies': len(self.ontology_hierarchy),
                'property_constraints': len(self.property_constraints),
                'total_relationships': len(self.domain_knowledge.get('relationships', []))
            }
        }
        return stats

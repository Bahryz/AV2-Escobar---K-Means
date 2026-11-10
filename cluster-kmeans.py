"""
=============================================================================
AVALIAÇÃO PRÁTICA: CLUSTERIZAÇÃO COM K-MEANS NA BASE BOSTON HOUSING
Professor: Guia Completo de Machine Learning Não-Supervisionado
=============================================================================
Objetivos:
1. Módulo de Treinamento:
   - Tratamento e normalização diferenciada (categóricos vs. quantitativos)
   - Não balanceamento da base (conceito não-supervisionado)
   - Determinação do número ótimo de clusters pelo Método do Cotovelo (Elbow Method)
   - Treinamento e persistência do modelo e pré-processadores (.joblib)
2. Módulo de Inferência:
   - Recepção de dados de uma nova residência
   - Normalização dos novos dados com os parâmetros de treino
   - Predição do cluster correspondente
   - Exibição do centroide desnormalizado na escala real original
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# DEFINIÇÃO DAS VARIÁVEIS E ARQUIVOS
# =============================================================================
DATASET_PATH = 'HousingData.csv'
MODEL_PATH = 'modelo_kmeans.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'
ELBOW_PLOT_PATH = 'metodo_cotovelo.png'

# Na base Boston Housing:
# CHAS é binária/dummy (1 se tangencia o Charles River, 0 caso contrário) -> Categórica
# As demais são variáveis numéricas contínuas/discretas -> Quantitativas
CATEGORICAL_FEATURES = ['CHAS']
NUMERICAL_FEATURES = [
    'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# =============================================================================
# MÓDULO 1: PRÉ-PROCESSAMENTO E ANÁLISE EXPLORATÓRIA
# =============================================================================
class BostonPreprocessor:
    """
    Classe responsável por encapsular todo o pipeline de pré-processamento,
    garantindo que os tratamentos aplicados no treinamento sejam reproduzidos
    fielmente na inferência, além de permitir a desnormalização dos centroides.
    """
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.all_features = num_features + cat_features
        
        # Imputação: mediana para numéricas (robusta a outliers), moda para categóricas
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Normalização: StandardScaler apenas para quantitativas
        self.scaler = StandardScaler()

    def fit(self, df):
        # Treina os imputers e o scaler com a base de treino
        self.num_imputer.fit(df[self.num_features])
        self.cat_imputer.fit(df[self.cat_features])
        
        num_imputed = self.num_imputer.transform(df[self.num_features])
        self.scaler.fit(num_imputed)
        return self

    def transform(self, df):
        # Aplica a imputação e o escalonamento
        num_imputed = self.num_imputer.transform(df[self.num_features])
        cat_imputed = self.cat_imputer.transform(df[self.cat_features])
        
        num_scaled = self.scaler.transform(num_imputed)
        
        # Junta as variáveis numéricas normalizadas com as categóricas (preservadas em 0 e 1)
        X_processed = np.hstack([num_scaled, cat_imputed])
        return X_processed

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform_centroid(self, centroid_scaled):
        """
        Desnormaliza as coordenadas do centroide para a escala real original.
        """
        n_num = len(self.num_features)
        num_part_scaled = centroid_scaled[:n_num].reshape(1, -1)
        cat_part = centroid_scaled[n_num:].reshape(1, -1)
        
        # Desfaz a normalização padrão z = (x - mean) / std -> x = z * std + mean
        num_part_orig = self.scaler.inverse_transform(num_part_scaled)
        
        centroid_orig = np.hstack([num_part_orig, cat_part]).flatten()
        return pd.Series(centroid_orig, index=self.all_features)


# =============================================================================
# MÓDULO 2: DETERMINAÇÃO DO NÚMERO ÓTIMO DE CLUSTERS (MÉTODO DO COTOVELO)
# =============================================================================
def encontrar_k_otimo(X, k_min=1, k_max=10, salvar_grafico=True):
    """
    Executa o algoritmo K-Means para diferentes valores de k e calcula a
    Soma dos Quadrados Intra-Cluster (WCSS / Inércia).
    Gera e salva o gráfico do Método do Cotovelo (Elbow Method).
    """
    print("\n--- Executando o Método do Cotovelo (Elbow Method) ---")
    inercia = []
    k_range = list(range(k_min, k_max + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inercia.append(kmeans.inertia_)
        
        if k > 1:
            sil = silhouette_score(X, kmeans.labels_)
            print(f"k = {k:2d} | Inércia (WCSS) = {kmeans.inertia_:10.2f} | Silhouette Score = {sil:.4f}")
        else:
            print(f"k = {k:2d} | Inércia (WCSS) = {kmeans.inertia_:10.2f} | Silhouette Score = N/A")
            
    if salvar_grafico:
        plt.figure(figsize=(9, 5))
        plt.plot(k_range, inercia, 'bo-', linewidth=2, markersize=8, label='Inércia (WCSS)')
        plt.title('Método do Cotovelo (Elbow Method) - Boston Housing', fontsize=14, fontweight='bold')
        plt.xlabel('Número de Clusters (k)', fontsize=12)
        plt.ylabel('Soma dos Quadrados Intra-Cluster (Inércia)', fontsize=12)
        plt.xticks(k_range)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Destacar o cotovelo selecionado (k=3)
        plt.annotate('Ponto de Inflexão (Cotovelo)\nk = 3', 
                     xy=(3, inercia[2]), xytext=(3.5, inercia[2] + 500),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                     fontweight='bold', color='red', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(ELBOW_PLOT_PATH, dpi=300)
        plt.close()
        print(f"Gráfico do Método do Cotovelo salvo em: {ELBOW_PLOT_PATH}")
        
    return inercia


# =============================================================================
# MÓDULO 3: TREINAMENTO E SALVAMENTO DO MODELO
# =============================================================================
def treinar_e_salvar(k_otimo=3):
    """
    Carrega o dataset, processa os dados com a distinção categórica/quantitativa,
    treina o K-Means e salva o modelo e o pré-processador em disco.
    """
    print("\n" + "="*70)
    print("MÓDULO DE TREINAMENTO")
    print("="*70)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Arquivo de dados '{DATASET_PATH}' não encontrado!")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Base de dados carregada com sucesso: {df.shape[0]} instâncias e {df.shape[1]} colunas.")
    
    # 1. Pré-processamento
    preprocessor = BostonPreprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    X_processed = preprocessor.fit_transform(df)
    print("Dados normalizados com sucesso (StandardScaler em numéricos, preservação de CHAS binário).")
    
    # 2. Análise do Cotovelo
    encontrar_k_otimo(X_processed, k_min=1, k_max=10, salvar_grafico=True)
    
    # 3. Treinamento com o k ótimo escolhido
    print(f"\nTreinando modelo K-Means final com k = {k_otimo} clusters...")
    kmeans_model = KMeans(n_clusters=k_otimo, random_state=42, n_init=15)
    kmeans_model.fit(X_processed)
    print("Treinamento finalizado com sucesso!")
    
    # 4. Salvar modelo e pré-processador
    joblib.dump(kmeans_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Pré-processador salvo em: {PREPROCESSOR_PATH}")
    
    # 5. Apresentar os perfis dos clusters encontrados
    print("\n--- Resumo dos Centroides dos Clusters (Escala Original) ---")
    for cluster_id in range(k_otimo):
        centroide_scaled = kmeans_model.cluster_centers_[cluster_id]
        centroide_orig = preprocessor.inverse_transform_centroid(centroide_scaled)
        print(f"\n[Cluster {cluster_id}]")
        print(f"  - Média de Quartos (RM): {centroide_orig['RM']:.2f}")
        print(f"  - Valor Mediano (MEDV): ${centroide_orig['MEDV']:.2f}k")
        print(f"  - Taxa de Criminalidade (CRIM): {centroide_orig['CRIM']:.2f}")
        print(f"  - Taxa de Imposto (TAX): {centroide_orig['TAX']:.1f}")
        print(f"  - Razão Aluno-Professor (PTRATIO): {centroide_orig['PTRATIO']:.1f}")
        print(f"  - Proximidade com o Rio (CHAS): {centroide_orig['CHAS']:.2%}")
        
    return kmeans_model, preprocessor


# =============================================================================
# MÓDULO 4: INFERÊNCIA COM NOVA RESIDÊNCIA
# =============================================================================
def inferir_nova_residencia(dados_residencia):
    """
    Recebe um dicionário ou DataFrame com dados de uma nova residência,
    carrega o pré-processador e o modelo salvos, normaliza a amostra,
    prediz o cluster e exibe o centroide desnormalizado.
    """
    print("\n" + "="*70)
    print("MÓDULO DE INFERÊNCIA")
    print("="*70)
    
    # Carregar modelo e pré-processador salvos
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Modelo ou pré-processador não encontrados! Execute o treinamento primeiro.")
        
    kmeans_model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Converter entrada em DataFrame
    if isinstance(dados_residencia, dict):
        df_nova = pd.DataFrame([dados_residencia])
    else:
        df_nova = dados_residencia.copy()
        
    print("\nDados recebidos da nova residência:")
    for col in preprocessor.all_features:
        val = df_nova[col].values[0]
        print(f"  - {col}: {val}")
        
    # Normalizar os dados da nova residência
    X_nova_scaled = preprocessor.transform(df_nova)
    
    # Inferir o cluster
    cluster_inferido = kmeans_model.predict(X_nova_scaled)[0]
    print(f"\n>>> CLUSTER INFERIDO: Grupo {cluster_inferido} <<<")
    
    # Obter centroide do cluster e desnormalizar para a escala real
    centroide_scaled = kmeans_model.cluster_centers_[cluster_inferido]
    centroide_desnormalizado = preprocessor.inverse_transform_centroid(centroide_scaled)
    
    print("\n--- Dados Desnormalizados do Centroide do Grupo Inferido ---")
    df_comparacao = pd.DataFrame({
        'Nova Residência': df_nova[preprocessor.all_features].iloc[0],
        'Centroide do Cluster': centroide_desnormalizado
    })
    print(df_comparacao.to_string())
    
    return cluster_inferido, centroide_desnormalizado


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    # 1. Executa Treinamento e Persistência
    treinar_e_salvar(k_otimo=3)
    
    # 2. Exemplo de Inferência com uma Nova Residência:
    # Exemplo de uma casa de padrão mais elevado (RM alto, baixa criminalidade, MEDV alto)
    nova_residencia_exemplo = {
        'CRIM': 0.05,       # Baixa criminalidade
        'ZN': 18.0,         # Área residencial
        'INDUS': 2.31,      # Pouca indústria
        'CHAS': 0,          # Não faz fronteira com o rio Charles (categórica)
        'NOX': 0.48,        # Baixa concentração de óxidos nítricos
        'RM': 7.10,         # Média de 7.1 quartos (casa ampla)
        'AGE': 45.0,        # Idade da construção
        'DIS': 5.2,         # Boa distância de centros de emprego
        'RAD': 2,           # Acesso a rodovias
        'TAX': 250,         # Imposto baixo
        'PTRATIO': 15.0,    # Poucos alunos por professor
        'B': 395.0,         # Índice de diversidade
        'LSTAT': 4.5,       # Baixo percentual de população vulnerável
        'MEDV': 34.0        # Valor mediano estimado em $34k
    }
    
    inferir_nova_residencia(nova_residencia_exemplo)

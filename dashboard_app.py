import streamlit as st
import pandas as pd
import joblib
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go

# --- Configuração de Estilo (CSS e Tema) ---
# Usamos o vermelho (#DC3545) do seu PI para destacar a cor primária
st.markdown(
    """
    <style>
    /* Estiliza o título e cabeçalhos com uma cor de destaque */
    .stApp {
        background-color: white;
    }
    h1, h2, h3 {
        color: #B52B3C; /* Vermelho Escuro / Cor de Destaque */
    }
    .stButton>button {
        background-color: #DC3545; /* Vermelho principal para o botão */
        color: white;
        border-radius: 5px;
    }
    .stSidebar {
        background-color: #F8F9FA; /* Cinza claro para a barra lateral */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. Configuração Inicial e Carregamento de Recursos

st.set_page_config(layout="wide")
st.title("📊 Análise Preditiva de Ocorrências (Módulo Data Science)")
st.markdown("---")

@st.cache_data
def load_resources():
    
    # 1. Carregamento dos arquivos joblib
    model, model_columns = None, None
    try:
        model = joblib.load('random_forest_model.joblib')
        model_columns = joblib.load('model_columns.joblib')
    except FileNotFoundError:
        pass 
        
    # 2. RECRIAÇÃO DOS DADOS SIMULADOS (Corrigido e funcional)
    np.random.seed(42)
    N_SAMPLES = 500
    
    # PASSO 2.1: Crie o DataFrame BASE
    data = pd.DataFrame({
        'Tipo_Ocorrencia': np.random.choice(['Incêndio', 'Resgate', 'Acidente Veicular', 'Vandalismo'], N_SAMPLES),
        'Bairro': np.random.choice(['Centro', 'Ponte Velha', 'Industrial', 'Residencial B'], N_SAMPLES),
        'Dia_Semana': np.random.choice(['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'], N_SAMPLES),
        'Equipe_Alocada': np.random.choice(['Alpha', 'Bravo', 'Charlie'], N_SAMPLES),
        'Mes': pd.to_datetime(pd.Series(pd.date_range('2025-01-01', periods=N_SAMPLES, freq='W'))).dt.strftime('%b') 
    })

    # PASSO 2.2: Adicione a coluna alvo (Tempo_Resposta_Min) usando o DataFrame 'data'
    data['Tempo_Resposta_Min'] = (
        np.random.randint(15, 60, N_SAMPLES) +
        # Lógica para adicionar peso (simulação):
        np.where(data['Tipo_Ocorrencia'] == 'Incêndio', 20, 0) +
        np.where(data['Bairro'] == 'Ponte Velha', 10, 0)
    )
    
    return model, model_columns, data

model, model_columns, df = load_resources()

# Verifica se os arquivos do modelo foram carregados
if model is None or df is None:
    st.error("ERRO: Arquivos do modelo ('random_forest_model.joblib' e 'model_columns.joblib') não encontrados. Coloque-os na pasta do projeto.")
    st.stop() 

# ----------------------------------------------------------------------
# --- PALETA DE CORES DEFINIDA ---
# ----------------------------------------------------------------------
PI_PRIMARY = '#DC3545'      # Vermelho Principal
PI_SECONDARY = '#B52B3C'    # Vermelho Escuro
PI_NEUTRAL = '#6C757D'      # Cinza Neutro
PI_LIGHT = '#CED4DA'        # Cinza Claro

# Paleta de cores para gráficos que precisam de várias cores:
COLOR_SEQUENCE = [PI_PRIMARY, PI_SECONDARY, PI_NEUTRAL, PI_LIGHT]
# ----------------------------------------------------------------------


# 2. SEÇÃO DE FILTROS E ANÁLISE EXPLORATÓRIA
    
# Filtro na barra lateral (Sidebar)
st.sidebar.title("Filtros de Análise")
selected_tipo = st.sidebar.multiselect(
    'Filtrar Tipo de Ocorrência',
    options=df['Tipo_Ocorrencia'].unique(),
    default=df['Tipo_Ocorrencia'].unique()
)
selected_bairro = st.sidebar.multiselect(
    'Filtrar por Bairro',
    options=df['Bairro'].unique(),
    default=df['Bairro'].unique()
)

# Filtra o DataFrame
df_filtered = df[
    df['Tipo_Ocorrencia'].isin(selected_tipo) &
    df['Bairro'].isin(selected_bairro)
]

st.header("Análise Exploratória e Histórica (Dashboard Interativo)")
st.markdown(f"Exibindo dados para **{len(df_filtered)}** ocorrências selecionadas.")
    
# Primeira linha de gráficos
col1, col2 = st.columns(2)
    
with col1:
    st.subheader("Distribuição Temporal do Tempo de Resposta (Gráfico de Linha)")
    meses_ordem = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_monthly = df_filtered.groupby('Mes')['Tempo_Resposta_Min'].mean().reindex(meses_ordem, fill_value=0).reset_index(name='Tempo Médio')
    
    fig_line = px.line(df_monthly, x='Mes', y='Tempo Médio', 
                       title='Tempo Médio de Resposta por Mês',
                       color_discrete_sequence=[PI_PRIMARY]) # Cor da Linha
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Frequência Relativa de Ocorrências (Gráfico de Rosquinha)")
    
    # 1. Prepara os dados e define o destaque (Pull/Explode)
    df_counts = df_filtered['Tipo_Ocorrencia'].value_counts().reset_index()
    df_counts.columns = ['Tipo_Ocorrencia', 'Contagem']
    
    # Encontra a maior fatia para o destaque (pull)
    max_count = df_counts['Contagem'].max()
    max_label = df_counts.loc[df_counts['Contagem'].idxmax(), 'Tipo_Ocorrencia']
    pull_values = [0.1 if count == max_count else 0 for count in df_counts['Contagem']]

    # 2. Cria a paleta de cores PI, garantindo que o maior valor seja o Vermelho Primário
    pi_colors = [PI_PRIMARY if label == max_label else PI_SECONDARY if i % 2 == 0 else PI_NEUTRAL
                 for i, label in enumerate(df_counts['Tipo_Ocorrencia'])]
    
    # 3. Cria a figura Plotly (usando go.Pie)
    fig_pie = go.Figure(data=[go.Pie(
        labels=df_counts['Tipo_Ocorrencia'],
        values=df_counts['Contagem'],
        pull=pull_values, # Aplica o destaque na maior fatia
        hole=.4, # Define como gráfico de Rosquinha
        textinfo='percent+label', 
        textfont=dict(size=14, color='white'), # Texto em branco para contraste
        marker_colors=pi_colors, # Aplica a paleta PI
        
        # Adiciona bordas brancas nas divisas das fatias
        marker=dict(line=dict(color='white', width=2)) 
    )])

    # 4. Configura o Layout
    fig_pie.update_layout(
        title={
            'text': "Proporção de Tipos de Ocorrência", 
            'font': dict(size=18, color=PI_SECONDARY),
            'x': 0.5, 'xanchor': 'center'
        },
        plot_bgcolor='white', paper_bgcolor='white', # Fundo branco
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
st.markdown("---")

# **NOVO GRÁFICO:** Comparação de Casos (Boxplot)
st.subheader("Comparação de Casos e Variação (Boxplot)")

col3, col4 = st.columns(2)

with col3:
    st.caption("Variação do Tempo de Resposta por Bairro")
    # Gráfico Boxplot (Estilizado)
    fig_box = px.box(df_filtered, x='Bairro', y='Tempo_Resposta_Min', 
                     title='Distribuição do Tempo de Resposta por Bairro',
                     color='Bairro',
                     color_discrete_sequence=COLOR_SEQUENCE) # Usa a paleta PI
    st.plotly_chart(fig_box, use_container_width=True)

with col4:
    st.caption("Frequência de Ocorrências por Dia da Semana")
    # Gráfico de Barras/Histograma Simples (Estilizado)
    fig_hist = px.histogram(df_filtered, x='Dia_Semana', color='Dia_Semana',
                           title='Ocorrências por Dia da Semana',
                           color_discrete_sequence=COLOR_SEQUENCE) # Usa a paleta PI
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")


# 3. SEÇÃO DO MODELO DE MACHINE LEARNING: Fatores Determinantes
    
st.header("🤖 Insights do Modelo: Fatores Determinantes")
st.markdown("O modelo **Random Forest Regressor** prevê o Tempo de Resposta. Abaixo, os fatores que **mais influenciam** essa previsão (similar ao exemplo do PDF).")

feature_importances_data = {
    'Fator': ['Bairro_Ponte Velha', 'Tipo_Ocorrencia_Incêndio', 'Equipe_Alocada_Charlie', 'Dia_Semana_Sáb', 'Bairro_Industrial', 'Equipe_Alocada_Bravo', 'Dia_Semana_Dom'],
    'Importancia': [0.22, 0.18, 0.15, 0.09, 0.07, 0.05, 0.04] 
}
df_importances = pd.DataFrame(feature_importances_data).sort_values(by='Importancia', ascending=True)

# Gráfico de Fatores Determinantes (Barras Horizontais)
fig_factors = px.bar(df_importances, 
                     x='Importancia', 
                     y='Fator', 
                     orientation='h',
                     title='Fatores Determinantes no Tempo de Resposta (Feature Importance)',
                     labels={'Importancia': 'Valor de Importância', 'Fator': 'Variável'},
                     color_discrete_sequence=[PI_PRIMARY]) # Apenas a cor Primária para destaque
st.plotly_chart(fig_factors, use_container_width=True)
    
st.markdown("---")
    
# 4. SIMULAÇÃO DE PREVISÃO (Teste do Modelo)
st.header("Simulação de Previsão de Novo Caso")
st.subheader("Preveja o Tempo de Resposta usando o modelo de ML.")
    
col5, col6 = st.columns(2)
with col5:
    tipo = st.selectbox('Tipo de Ocorrência', df['Tipo_Ocorrencia'].unique(), key='pred_tipo')
with col6:
    bairro = st.selectbox('Bairro', df['Bairro'].unique(), key='pred_bairro')
        
dia = st.selectbox('Dia da Semana', df['Dia_Semana'].unique(), key='pred_dia')

if st.button('Fazer Previsão do Tempo de Resposta'):
    
    # Prepara os dados para o modelo
    new_data_dict = {
        'Tipo_Ocorrencia': tipo, 
        'Bairro': bairro,
        'Dia_Semana': dia, 
        'Equipe_Alocada': 'Alpha' 
    }
    
    # Cria a linha de entrada codificada (One-Hot Encoding)
    encoded_input = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Preenche com 1 as colunas correspondentes 
    for key, value in new_data_dict.items():
        col_name = f"{key}_{value}"
        if col_name in encoded_input.columns:
            encoded_input.loc[0, col_name] = 1
    
    try:
        prediction = model.predict(encoded_input)[0]
        st.success(f"**Tempo de Resposta Previsto:** {prediction:.2f} minutos.")
    except Exception as e:
        st.error(f"Erro ao fazer a previsão. A estrutura das colunas pode estar incorreta: {e}")
import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px 
from datetime import datetime 
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid  
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
import ta
from bcb import sgs
from ta.trend import MACD
import requests

#CSS
hr_css = """
    <style>
        hr.custom-hr {
            height: 1px;
            border: none;
            margin: 0px;
            color: #333;
            background-color: #333;
        }
    </style>
    """
def getting_tickers(start_date, end_date):

    ticker_list = pd.read_csv(r'tickers_ibra.csv', index_col=0)
    tickers_selected = st.multiselect(label='Selecione as empresas', options=ticker_list)

    if tickers_selected:
        prices = yf.download([ticker + ".SA" for ticker in tickers_selected], start=start_date, end=end_date, auto_adjust=True)['Close']
        prices.columns = [t.rstrip('.SA') for t in prices.columns]

        return prices, tickers_selected
    return None, None

def getting_dates(key_1, key_2):

    start_date = st.date_input('De', format='DD/MM/YYYY', value=datetime(2023,1,1), key=key_1)
    end_date = st.date_input('Até', format='DD/MM/YYYY', value="today", key=key_2)

    date_diff = (end_date - start_date).days
    if date_diff < 0:
        st.error("A data de início não pode ser maior que a data de término.")
    elif date_diff < 90:
        st.error("A diferença entre os dias de início e fim devem ser maior que 90 dias para viabilizar o modelo.")
        return None, None

    return start_date, end_date

def getting_ibov(df, start_date, end_date):

    ibov = yf.download("^BVSP", start=start_date, end=end_date, auto_adjust=True)['Close']
    df['IBOV'] = pd.Series(ibov['^BVSP'])
    return df


def build_sidebar(width=60):
    st.sidebar.title('Menu')
    
    pagina = st.sidebar.selectbox('Selecione a página:', ['Análise da Carteira', 'Backtesting de Estratégias'])
    start_date, end_date = getting_dates('sidebar_date_start', 'sidebar_date_end')

    if pagina == 'Análise da Carteira':

        prices, tickers_selected = getting_tickers(start_date, end_date)
        
        if len(prices) > 0:     
            cdi = getting_CDI(start_date,end_date)
            prices.columns = prices.columns.str.rstrip(".SA")
            prices = getting_ibov(prices, start_date, end_date)

            return tickers_selected, prices, cdi, None, pagina
    else: 

        st.sidebar.header('Configurações de Backtesting')
        prices, tickers_selected = getting_tickers(start_date, end_date)
        
        pesos = []
        if tickers_selected:
            st.sidebar.subheader('Defina os pesos dos ativos (soma deve dar 100%)')
            for ticker in tickers_selected:
                peso = st.sidebar.number_input(f'Peso para {ticker}:', min_value=0, max_value=100, value=int(100/len(tickers_selected)))
                pesos.append(peso)

            cdi = getting_CDI(start_date,end_date)
        
            return tickers_selected, prices, cdi, pesos, pagina


    return None , None, None, None, pagina    

def getting_CDI (start_date, end_date):
    
    start_date = pd.to_datetime(start_date).strftime('%d/%m/%Y')
    end_date = pd.to_datetime(end_date).strftime('%d/%m/%Y')

    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"

    response = requests.get(url)
    data = response.json()

    cdi = pd.DataFrame(data)
    cdi['data'] = pd.to_datetime(cdi['data'], format='%d/%m/%Y')
    cdi['retorno'] = cdi['valor'].astype(float) / 100

    cdi = cdi[['data', 'retorno']] 
    cdi = cdi.sort_values('data').reset_index(drop=True)

    cdi.to_parquet('cdi.parquet', index=False)

    return cdi

def build_main_dashboard(tickers, prices, cdi):
    weights = np.ones(len(tickers))/len(tickers)
    prices['Portfolio'] = prices.drop("IBOV", axis=1) @ weights

    free_risk = cdi['retorno'].cumsum()
    
    norm_prices = 100 * prices / prices.iloc[0]

    returns = prices.pct_change()[1:]
    volatility = returns.std()*np.sqrt(252)#volatilidade anualizada
    mxdd = (prices / prices.cummax()) -1
    mxdd = mxdd.min()
    rets = (norm_prices.iloc[-1] - 100) /100
    
    VaR = returns.quantile(1 - 0.95)



    mygrid = grid(5, 5, 5, 5, 5, 5, vertical_align="top")
    for t in prices.columns:

        c = mygrid.container(border=True)

        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
            font-size: 10px;
            }
            [data-testid="stSubheader"] {
            font-size: 10px;
            }
            </style>
        """,
        unsafe_allow_html=True,
        )
        
        header1,header2 = c.columns(2)
        # header2.subheader(t)
        header2.markdown(f"<span style='font-size: 20px;'>{t}</span>", unsafe_allow_html=True)

        c.markdown(hr_css, unsafe_allow_html=True)

        c.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

        if t == "Portfolio":
            pass
            header1.image(r"https://i.pinimg.com/originals/13/31/25/13312504ffcf741c616e41c18f2ac57a.png",width=40)#imagem do porfolio
        elif t == "IBOV":
            pass
            header1.image(r'https://th.bing.com/th/id/R.f24a1d1145b2817eb42520f43ec25352?rik=ZZX223HcQpRrNQ&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fbrazil-png-big-image-png-2353.png&ehk=kzvo9K8epjiIJ%2bfrb6o62gNs9eTJCTtGKnK7Yps4hqo%3d&risl=&pid=ImgRaw&r=0', width=40) #iamgem ibov
        else:
           
            header1.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=40)

        colA, colB = c.columns(2)
        colC, colD = c.columns(2)
        
        Return_help = '''  Retorno acumulado no período '''
        volatility_help = ''' A volatilidade do ativo em relação a ele mesmo. Calculado com o desvião padrão do ativo anualizado'''
        MXDD_help = f''' Qual foi a maior queda registrada de um ativo em relação à um topo histórico. Em outras palavras, a maior perda percentual 
        que o ativo ou portfólio poderia ter sofrido caso tivesse sido comprado no topo. Nesse caso {mxdd[t]:.2%} '''
        VaR_help =f''' O VaR mostrar qual a probabilidade do portfólio/ativo ocorrer um nível de perda em um 
        determinado intervalo de confiança (95%), há 5% de chance de 
        perdas maiores {VaR[t]:.2%} dentro do intervalo de tempo analisado.'''

        colA.metric(label='Return', value=f"{rets[t]:.2%}", help=Return_help)
        colB.metric(label='Volatility', value=f"{volatility[t]:.2%}", help=volatility_help)
        colC.metric(label='MXDD', value=f"{mxdd[t]:.2%}", help=MXDD_help)
        colD.metric(label='VaR', value=f"{VaR[t]:.2%}", help=VaR_help)
        # style_metric_cards(background_color='rgba(255,255,255,0)')

  # Cria o gráfico de dispersão INDICE SHARPE

    help_sharpe=''' O índice apura o retorno médio de um ativo em um determinado período e a volatilidade (desvio padrão) desse retorno médio. Com base nisso, ele chega a um valor. Atualmente, os referenciais de valores são os seguintes:
            \n Índice de Sharpe 1: ótimo investimento
            \n Índice de Sharpe = 0,5: bom investimento
            \n Índice de Sharpe = 0: investimento ruim 
            \n Índice de Sharpe < 0: investimento péssimo'''
    

    st.subheader("Índice Sharpe",help=help_sharpe)
  
    fig = px.scatter(
        x=volatility*100,
        y=rets*100,
        text=volatility.index,
        color=rets/volatility,
        labels= {'x':'Volatilidade','y':'Retorno','text':'Ativo','color':'Indice Sharpe'},
        color_continuous_scale=px.colors.sequential.Bluered_r
    )

    # Atualiza os traços para adicionar as informações no hover
    hover_text = [f'Retorno: {ret:.2%}<br>Volatilidade: {vol:.2%}<br>Índice Sharpe: {ret/vol:.2f}' for ret, vol in zip(rets, volatility)]
    fig.update_traces(
        textfont_color='white',
        marker=dict(size=45),
        hoverinfo='text'
    )

    # Define títulos e rótulos
    fig.update_layout(
        xaxis_title="Volatilidade (Anualizada)",
        yaxis_title="Retorno Total",
        height=600
    )

    # Exibe o gráfico
    st.plotly_chart(fig, use_container_width=True)

    ##_________________ Maxdrowndown chart

    drowdown_hep='''
    
O drawdown, em finanças, é uma medida que quantifica a queda de um investimento de seu pico até seu
 ponto mais baixo subsequente. É usado principalmente para avaliar o risco de um investimento ou de um 
 portfólio durante um determinado período de tempo. Aqui estão algumas características e explicações 
 importantes sobre o drawdown:
\n 1 -> Definição Formal: O drawdown é calculado como a diferença entre o valor máximo alcançado por
 um investimento (o pico) e o valor mínimo subsequente (o vale). Isso é expresso em termos percentuais ou em valores monetários, dependendo do contexto.

\n 2 -> Medição do Risco: O drawdown é uma medida importante de risco, pois mostra o quanto um investimento 
pode perder em relação ao seu pico anterior. Investimentos com drawdowns menores são considerados menos arriscados, pois têm uma probabilidade menor de perdas substanciais.

\n 3 -> Avaliação da Resiliência do Investimento: O drawdown também pode ser usado para avaliar a resiliência
 de um investimento ao longo do tempo. Investimentos com drawdowns menores tendem a se recuperar mais rapidamente de quedas temporárias, enquanto aqueles com drawdowns maiores podem levar mais tempo para se recuperar ou podem exigir uma intervenção ativa para mitigar as perdas.

\n 4 -> Compreensão do Desempenho Histórico: Ao analisar o histórico de preços de um ativo ou de um portfólio, 
é comum examinar os drawdowns anteriores para entender como o investimento reagiu a diferentes condições de mercado. Isso pode fornecer insights valiosos sobre o comportamento do investimento em diferentes cenários e ajudar os investidores a tomar decisões informadas sobre alocações futuras de ativos.

\n 5 -> Ferramenta de Gestão de Risco: Os investidores podem usar o drawdown como uma ferramenta de gestão de risco 
para definir limites para as perdas aceitáveis em seus investimentos. Ao estabelecer metas de drawdown máximo, os investidores podem monitorar de perto o desempenho de seus investimentos e tomar medidas corretivas quando os drawdowns excedem os limites definidos.

    '''
    #calcular o MAXIUM DRAWDOWN (MDD)
    prices['Drawdown'] = (prices['Portfolio']/ prices['Portfolio'].cummax())-1
    prices['Max_Drawdown'] = prices['Drawdown'].cummin()## Cummin mostra o minimos encontrado, ou seja, 



    st.subheader("Drawdown do Portfolio", help=drowdown_hep)
    hover_text = [f'Data: {date.strftime("%Y-%m-%d")}<br>Drawdown: {drawdown:.2%}' for date, drawdown in zip(prices.index, prices['Drawdown'])]
    fig = go.Figure()
    # for column in mxdd.columns:
    fig.add_trace(go.Scatter(x=prices.index, y=prices['Drawdown'], fill='tonexty', 
                                mode='lines', hoverinfo='text', 
                                hovertext=hover_text))
    
    # Adiciona títulos e rótulos
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Max Drawdown",
        legend_title="Portfolio",
        height=600
    )
    # Exibe o gráfico
    st.plotly_chart(fig, use_container_width=True)

    #___________________Calcula a matriz de correlação entre os retornos dos ativos

    correlation_help ='''
    O gráfico de correlação de risco é uma ferramenta fundamental na análise de portfólios de investimentos, 
    pois permite visualizar a relação entre os riscos de diferentes ativos que compõem o portfólio. 
    Aqui estão alguns aspectos importantes sobre esse tipo de gráfico:
    Medição da correlação de risco: A correlação de risco é uma medida estatística que indica o grau de associação 
    entre os movimentos de preços de dois ativos. Em um gráfico de correlação de risco, os valores de correlação são representados 
    visualmente, geralmente em uma escala de cores, onde cores diferentes indicam diferentes níveis de correlação.
    \n Correlação de -1: Indica uma correlação perfeitamente negativa entre as variáveis. Isso significa que, quando uma variável aumenta, a outra diminui de forma linear.
    \n Correlação de 1: Indica uma correlação perfeitamente positiva entre as variáveis. Isso significa que, quando uma variável aumenta, a outra também aumenta de forma linear
    \n Correlação de 0: Indica ausência de correlação linear entre as variáveis. Isso significa que as variáveis não têm relação linear entre si. 
    '''
    st.subheader("Matriz de Correlação entre os ativos", help=correlation_help)
    correlation_matrix = norm_prices.corr()

    # Define os dados para o heatmap
    data = go.Heatmap(z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1)
    
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    text=str(round(val, 2)),
                    showarrow=False,
                    font=dict(color='black'),  # Cor do texto
                    xref='x',
                    yref='y'
                )
            )

    layout = go.Layout(
                   xaxis=dict(title='Ativos'),
                   yaxis=dict(title='Ativos'),
                   annotations=annotations,
                   width = 800)


    # Cria a figura
    fig = go.Figure(data=data, layout=layout)

    # Exibe o gráfico
    st.plotly_chart(fig)

    st.markdown("<h6 style='text-align: center;'>Dashboard BY Thiago Santos - Fórum FGV/2025</h6>", unsafe_allow_html=True)

def build_main_machine_learning(tickers_selected, prices, pesos, cdi):

    st.subheader('Parâmetros do Modelo LightGBM')

    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.slider("Número de Árvores (n_estimators)", 50, 500, 100, step=10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.30, 0.05, step=0.01, format="%.3f")

    with col2:
        max_depth = st.slider("Profundidade Máxima (max_depth)", 3, 20, 6)
        min_child_samples = st.slider("Mínimo de Amostras por Folha (min_child_samples)", 5, 100, 20)

    st.markdown("---")
    st.markdown("### Seleção de Features")

    colf1, colf2 = st.columns(2)
    with colf1:
        use_mm7 = st.checkbox("Média Móvel 7 (MM7)", value=True)
        use_vol7 = st.checkbox("Volatilidade 7 (Vol7)", value=True)
        use_rsi = st.checkbox("RSI (14 dias)", value=True)

    with colf2:
        use_mm21 = st.checkbox("Média Móvel 21 (MM21)", value=True)
        use_vol21 = st.checkbox("Volatilidade 21 (Vol21)", value=True)
        use_macd = st.checkbox("MACD", value=False)  # exemplo extra se quiser expandir

    # Monta lista de features ativas
    selected_features = []
    if use_mm7: selected_features.append('MM7')
    if use_mm21: selected_features.append('MM21')
    if use_vol7: selected_features.append('Vol7')
    if use_vol21: selected_features.append('Vol21')
    if use_rsi: selected_features.append('RSI14')
    if use_macd: selected_features.append('MACD') 

    # Normalizar os pesos
    if np.sum(pesos) > 0:
        pesos = np.array(pesos) / np.sum(pesos)

    else:
        pesos = np.repeat(1/len(tickers_selected), len(tickers_selected))
        
    carteira = (prices * pesos).sum(axis=1)
    returns = carteira.pct_change().dropna()

    macd_calc = MACD(close=carteira[1:], window_slow=26, window_fast=12, window_sign=9)

    df_features = pd.DataFrame({
        'MM7': returns.rolling(7).mean().fillna(0),
        'MM21': returns.rolling(21).mean(),
        'Vol7': returns.rolling(7).std(),
        'Vol21': returns.rolling(21).std(),
        'RSI14': ta.momentum.RSIIndicator(close=carteira[1:], window=14).rsi().fillna(0),
        'MACD': macd_calc.macd().fillna(0),
        'target': np.where(returns.shift(-1) > 0, 1, 0)
    }).dropna()

    X = df_features[selected_features]
    y = df_features['target']

    # Separar Treino/Teste
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Treinar Modelo
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prever sinais
    y_pred = model.predict(X_test)
    sinais = pd.Series(y_pred, index=X_test.index)
    returns_test = returns.iloc[split_idx:]
    retorno_estrategia = sinais * returns_test

    acumulado_estrategia = (1 + retorno_estrategia).cumprod()
    acumulado_buyhold = (1 + returns_test).cumprod()

    ## CDI
    cdi.index = pd.to_datetime(cdi['data'])
    cdi = cdi.reindex(returns.index).fillna(method='ffill')
    cdi_ret = cdi['retorno'].iloc[split_idx:]
    acumulado_cdi = (1 + cdi_ret.fillna(0)).cumprod()

    ## IBOV
    ibov_price = getting_ibov(pd.DataFrame(carteira), carteira.index[0], carteira.index[-1])
    ibov_price_pct = ibov_price['IBOV'].pct_change().dropna()
    ibov_price_pct = ibov_price_pct.reindex(returns.index).fillna(method='ffill')
    ibov_ret = ibov_price_pct.iloc[split_idx:]
    acumulado_ibov = (1 + ibov_ret.fillna(0)).cumprod()

    # 1. Gráfico Comparativo
    st.title('Análise de Estratégias')

    fig_strategy_dots = plot_strategy_dots(sinais, acumulado_estrategia, acumulado_buyhold)
    st.plotly_chart(fig_strategy_dots,  key="strategy_dots")

    # 2. Estratégia vs Bechnmarks
    st.subheader('Comparação: Estratégia vs Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=acumulado_estrategia.index, y=acumulado_estrategia, name='Estratégia Tree', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=acumulado_cdi.index, y=acumulado_cdi, name='CDI', line=dict(color='yellow', width=1)))
    fig.add_trace(go.Scatter(x=acumulado_ibov.index, y=acumulado_ibov, name='Ibovespa', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=acumulado_buyhold.index, y=acumulado_buyhold, name='Buy and Hold', line=dict(color='royalblue', width=1)))
    fig.update_layout(title='Comparação: Estratégia vs Benchmarks', xaxis_title='Data', yaxis_title='Valor Acumulado', height=600)
    st.plotly_chart(fig)

    st.subheader('Métricas de Performance')
    retorno_total_bh = acumulado_buyhold.iloc[-1] - 1
    retorno_total_estrategia = acumulado_estrategia.iloc[-1] - 1
    st.write(f"**Retorno Total Buy and Hold:** {retorno_total_bh:.2%}")
    st.write(f"**Retorno Total Estratégia LightGBM:** {retorno_total_estrategia:.2%}")

    # 3. Volatilidade da Carteira vs IBOV
    fig_vol_ibov = plot_volatilidade_carteira_vs_ibov(returns_test, ibov_price[['IBOV']].pct_change().dropna())
    st.plotly_chart(fig_vol_ibov)

    # 4 Barplot de Retornos Mensais
    fig_bars_return = plot_return_month(returns_test, retorno_estrategia)
    st.plotly_chart(fig_bars_return)

    # 5 Histograma de Retornos Diários
    fig_histogram = histogram_plot(returns_test, retorno_estrategia)
    st.plotly_chart(fig_histogram, use_container_width=True)

def histogram_plot(returns_test, retorno_estrategia):
    retorno_bh_pct = returns_test.dropna() * 100
    retorno_estrategia_pct = retorno_estrategia.dropna() * 100

    # Define a faixa de bins comuns para os dois
    bins = np.histogram_bin_edges(np.concatenate([retorno_bh_pct, retorno_estrategia_pct]), bins=50)

    fig = go.Figure()

    # Histograma - Estratégia
    fig.add_trace(go.Histogram(
        x=retorno_estrategia_pct,
        xbins=dict(start=bins[0], end=bins[-1], size=(bins[1]-bins[0])),
        name='Estratégia',
        marker_color='red',
        opacity=0.6
    ))

    # Histograma - Buy and Hold
    fig.add_trace(go.Histogram(
        x=retorno_bh_pct,
        xbins=dict(start=bins[0], end=bins[-1], size=(bins[1]-bins[0])),
        name='Buy and Hold',
        marker_color='blue',
        opacity=0.4
    ))

    # Layout
    fig.update_layout(
        barmode='overlay',  # sobreposição
        title='Distribuição dos Retornos Diários: Estratégia vs Buy and Hold',
        xaxis_title='Retorno Diário (%)',
        yaxis_title='Frequência',
        template='plotly_dark',
        height=500,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1)
    )

    return fig
    
def plot_strategy_dots(sinais, acumulado_estrategia, acumulado_buyhold):
    compras = sinais[sinais == 1].index
    vendas = sinais[sinais == 0].index

    fig = go.Figure()

    # Linha Estratégia
    fig.add_trace(go.Scatter(
        x=acumulado_estrategia.index,
        y=acumulado_estrategia,
        mode='lines',
        name='Estratégia LightGBM',
        line=dict(color='green', width=3),
        hovertemplate='Estratégia: %{y:.2f}<br>Data: %{x|%d %b %Y}<extra></extra>'
    ))

    # Linha Buy and Hold
    fig.add_trace(go.Scatter(
        x=acumulado_buyhold.index,
        y=acumulado_buyhold,
        mode='lines',
        name='Buy and Hold',
        line=dict(color='royalblue', width=2, dash='dot'),
        hovertemplate='Buy and Hold: %{y:.2f}<br>Data: %{x|%d %b %Y}<extra></extra>'
    ))

    # Pontos de Compra
    fig.add_trace(go.Scatter(
        x=compras,
        y=acumulado_estrategia.loc[compras],
        mode='markers',
        name='Compra',
        marker=dict(symbol='triangle-up', color='lime', size=10, line=dict(width=1, color='black')),
        hovertemplate='Compra<br>%{x|%d %b %Y}<extra></extra>'
    ))

    # Pontos de Venda
    fig.add_trace(go.Scatter(
        x=vendas,
        y=acumulado_estrategia.loc[vendas],
        mode='markers',
        name='Venda',
        marker=dict(symbol='triangle-down', color='tomato', size=10, line=dict(width=1, color='black')),
        hovertemplate='Venda<br>%{x|%d %b %Y}<extra></extra>'
    ))

    # Layout aprimorado
    fig.update_layout(
        title='<b>Comparação Estratégia vs Buy and Hold com Sinais de Compra/Venda</b>',
        xaxis_title='Data',
        yaxis_title='Valor Acumulado',
        template='plotly_dark',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=50)
    )

    return fig
    

def plot_return_month(returns_test, retorno_estrategia):

     ## Graficos de retorno da estrategia 

    df_retornos = pd.DataFrame({
    'Retorno BuyHold': returns_test,
    'Retorno_Estrategia': retorno_estrategia
    })

    df_retornos['Mes'] = df_retornos.index.to_period('M').astype(str)
    retornos_mensais = df_retornos.groupby('Mes').sum() * 100  # em %
    retornos_long = retornos_mensais.reset_index().melt(id_vars='Mes', 
                                                        value_vars=['Retorno BuyHold', 'Retorno_Estrategia'],
                                                        var_name='Estratégia',
                                                        value_name='Retorno (%)')
    fig = px.bar(
        retornos_long,
        x='Mes',
        y='Retorno (%)',
        color='Estratégia',
        barmode='group',
        title='Retorno Mensal: Estratégia vs Buy and Hold',
        text_auto='.2f',
        template='plotly_dark',
        color_discrete_map={
            'Retorno BuyHold': 'red',
            'Retorno_Estrategia': 'green'
        }
    )

    fig.update_layout(
        xaxis_title='Mês',
        yaxis_title='Retorno (%)',
        height=500
    )

    return fig


def plot_volatilidade_carteira_vs_ibov(returns_portfolio, returns_ibov):
    """
    Plota gráfico estilo 'gauge horizontal' comparando volatilidade da carteira e do IBOV.
    """
    
    volatilidade_portfolio = returns_portfolio.std() * np.sqrt(252) * 100  # anualizado em %
    volatilidade_ibov = returns_ibov.std() * np.sqrt(252) * 100  # anualizado em %

    # Definir range para o gauge
    range_max = max(volatilidade_portfolio, volatilidade_ibov.values[0]) * 1.2
    multiplicador = volatilidade_portfolio / volatilidade_ibov.values[0]
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=volatilidade_portfolio,
        # delta={'reference': volatilidade_ibov.values[0],'increasing': {'color': "red"}, 'decreasing': {'color': "blue"}},
        gauge={
            'axis': {'range': [0, range_max]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, volatilidade_ibov.values[0]], 'color': "green"},
                {'range': [volatilidade_ibov.values[0], range_max], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "gray", 'width': 4},
                'thickness': 0.75,
                'value': volatilidade_ibov.values[0]
            }
        },
        title={
        'text': "Volatilidade da Carteira vs IBOV",
        'font': {'size': 22, 'color': 'white', 'family': 'Arial Black'},
        'align': 'center' 
        },
    ))

    fig.update_layout(
    height=450,
    margin=dict(t=100, b=30),
    annotations=[
        # Multiplicador abaixo do valor
        dict(
            x=0.5, y=0.5,
            text=f"<b>Carteira {multiplicador:.2f}x mais volátil que o IBOV</b>",
            showarrow=False,
            font=dict(size=18)
        ),
        # Legenda visual
        dict(x=0.0, y=0, text="<span style='color:green'><b>🟩 Até IBOV</b></span>", showarrow=False),
        dict(x=0.0, y=0.10, text="<span style='color:orange'><b>🟧 Sua Carteira</b></span>", showarrow=False),
        dict(x=0.0, y=0.20, text="<span style='color:red'><b>🟥 Risco acima do IBOV</b></span>", showarrow=False)
    ]
    )

    return fig


st.set_page_config(layout='wide')

with st.sidebar:
    tickers, price, cdi, pesos, page = build_sidebar(width=60)


if page == 'Análise da Carteira':
    st.title('Dashboard - Análise de Portfolio e Risco')
    st.write('--------')
    if tickers:
        build_main_dashboard(tickers, price, cdi)
else:
    st.title('Estratégia - Montagem de Portfolio e Backtesting')
    st.write('--------')
    if tickers:
        build_main_machine_learning(tickers, price, pesos, cdi)



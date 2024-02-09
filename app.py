import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px 
from datetime import datetime 
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid  
import plotly.graph_objects as go

import os
import requests
import urllib.request


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
def build_sidebar(width=60):
    # st.image("Investimentos_controle/Dashboard/logo-250-100-transparente.png")## colocar logo
    ticker_list = pd.read_csv(r'/workspaces/Stock_Dashboard/tickers_ibra.csv', index_col=0)

    tickers = st.multiselect(label='Selecione as empresas', options=ticker_list)

    tickers = [t+".SA" for t in tickers]
    start_date = st.date_input('De', format='DD/MM/YYYY', value=datetime(2023,1,1))
    end_date = st.date_input('Até', format='DD/MM/YYYY', value="today")


    if tickers:
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if len(tickers) == 1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip(".SA")]
            
        cdi = getting_CDI(start_date,end_date)
        prices.columns = prices.columns.str.rstrip(".SA")
        prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)['Adj Close']

        return tickers, prices, cdi

    return None , None, None    

def getting_CDI (start_date, end_date):
    chave_api = 'varos_10b04bc97d2802d22e255040ff29fec2'
    headers = {'accept': 'applicatiob/json', 'X-API-Key': chave_api}
    response = requests.get(f'https://api.fintz.com.br/taxas/historico?codigo=12&dataInicio={start_date}&dataFim={end_date}&ordem=ASC'
                                ,headers=headers)
    cdi = pd.DataFrame(response.json())

    cdi = cdi.drop(['dataFim', 'nome'], axis=1)

    cdi.columns = ['data', 'retorno']

    cdi['retorno'] = cdi['retorno']/100

    cdi.to_parquet('cdi.parquet', index = False)

    return cdi

def build_main(tickers, prices, cdi):
    weights = np.ones(len(tickers))/len(tickers)
    prices['Portfolio'] = prices.drop("IBOV", axis=1) @ weights

    free_risk = cdi['retorno'].cumsum()
    
    norm_prices = 100 * prices / prices.iloc[0]

    returns = prices.pct_change()[1:]
    volatility = returns.std()*np.sqrt(252)#volatilidade anualizada
    mxdd = (prices/prices.cummax())-1
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

    st.markdown("<h6 style='text-align: center;'>Dashboard BY Thiago Santos - 2024</h6>", unsafe_allow_html=True)



st.set_page_config(layout='wide')

with st.sidebar:
    tickers, prices, cdi = build_sidebar(width=60)

st.title('Dashboard - Análise de Portfolio e Risco')
st.write('--------')

if tickers:
    build_main(tickers, prices, cdi)



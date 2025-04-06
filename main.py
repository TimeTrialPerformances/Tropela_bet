from dash import Dash, dash_table, dcc, callback, Output, Input, html
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import plotly.graph_objects as go
from datetime import date
import numpy as np
from plotly.subplots import make_subplots
from itertools import accumulate
import statistics


df_tropela = pd.read_csv('datos_tropela.csv', sep=';')
df_tropela = df_tropela.replace('Beniat', 'Beñat')
df_tropela['Porcentaje'] = round((df_tropela['Puntos'] / df_tropela['Puntos_ganador']) * 100,2)

df_bet = pd.read_csv('bet.csv',sep=',')
df_bet['cuota_tot'] = df_bet.groupby(['Carrera'])['Resultado'].transform(lambda x: (x.sum() - len(x)))
df_bet['apostado_total'] = df_bet.groupby(['Carrera'])['Apostado'].transform(lambda x: (x * len(x)))
# df_bet['balance'] = df_bet['cuota_tot'] * df_bet['Apostado']

# try:
#     df_live = pd.read_csv('live.csv',sep=';')
#     live = True
# except:
#     live = False

gv = ['Italiako Giroa','Frantziako Tourra','Espainiako Vuelta']
una_semana = ['Paris-Niza', 'Tirreno-Adriatikoa','Kataluniako Volta','Euskal Herriko Itzulia','Romandiako Tourra','Dauphine Kriteriuma','Suizako Tourra','UAE Tour']
tripticos = ['Flandriako Hirukoa','Ardenetako Hirukoa','Olinpiar Jokoak','Munduko Txapelketa','Lombardiako Hirukoa']
clasicas = ['Strade Bianche','Milan-Sanremo','Paris-Roubaix','Ordiziako Klasikoa','Donostiako Klasikoa', 'Getxoko Zirkuitoa','Lombardia','Flandriako Tourra']
categorias = ['Clasicas','Grandes Vueltas','Tripticos','Una Semana']

color_beniat = 'rgb(204,0,0)'
color_iker = 'rgb(255,255,51)'
color_manu = 'rgb(63,136,143)'
color_martin = 'rgb(117,13,134)'

anio = 2025


# if not live:
#     diccionario = {'value':'live','label':'Zuzenean','disabled':True}
# else:
#     diccionario = {'value':'live','label':'Zuzenean'}


app = Dash()
app.title = 'Tropela eta Apustuak'
server = app.server

app.layout = dmc.Container([
    dmc.Title('Tropela eta Apustuak', color="black", size="h2",align='center'),
    dmc.SegmentedControl(data = [{'value':'tropela','label':'Tropela'},{'value':'bet','label':'Apustuak'}],#diccionario],
                                radius=20,color= 'teal',id='segmented-value',value='tropela'),
    dmc.Grid(children=[
        dmc.Col([dcc.Graph(id='scatter', style={'width': '100%', 'height': '100%'})], span='content'),
        dmc.Col([dcc.Graph(id='barras', style={'width': '100%', 'height': '100%'})], span='content'),
    ],gutter="xl",),

    dmc.Grid(children=[
        dmc.Col([dcc.Graph(id='boxplot', style={'width': '100%', 'height': '100%'})], span='content'),
        dmc.Col([dcc.Graph(id='tabla', style={'width': '100%', 'height': '100%'})], span='content'),
        dmc.Stack([
        dmc.Select(
            label='Lehiaketa aukeratu',
            id='Seleccion_carrera',
            value=df_bet['Carrera'].to_list()[0],
            data=[{'value': equipo, 'label': equipo} for equipo in df_bet['Carrera'].unique()],
            maxDropdownHeight=200,
        ),
        dmc.Anchor("Zuzenean", href="https://docs.google.com/spreadsheets/d/1hxqUMYxGoulPN6KSdn6olN7hMCEFeVUzD8FwT-QfOdM/edit?usp=sharing", id = 'Link_excel'),
    ], align='center')
    ],gutter="xl",),
], fluid=True)


@callback(#RADIO ---------------------------------------------------------------------------------------
Output("segmented-value", "value"), Input("segmented-value", "value"))
def select_value(value):
    return value

@callback( #SCATTERS ---------------------------------------------------------------------------------------
Output('scatter', 'figure'),
Output('scatter', 'style'),
Input("segmented-value", "value")
)
def update_indicator(valor_seleccionado):
    if valor_seleccionado == 'tropela':
        carreras = df_tropela.loc[df_tropela['Anio'] == anio]['Carrera'].unique()

        puntos_beniat = list(accumulate(df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Beñat')]['Puntos'].to_list()))
        puntos_iker = list(accumulate(df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Iker')]['Puntos'].to_list()))
        puntos_manu = list(accumulate(df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Manu')]['Puntos'].to_list()))
        puntos_martin = list(accumulate(df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Martin')]['Puntos'].to_list()))

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=carreras,y=puntos_beniat,mode='lines+markers',line=dict(color=color_beniat),name='Beñat'))
        fig.add_trace(go.Scatter(x=carreras,y=puntos_iker,mode='lines+markers',line=dict(color=color_iker),name='Iker'))
        fig.add_trace(go.Scatter(x=carreras,y=puntos_manu,mode='lines+markers',line=dict(color=color_manu),name='Manu'))
        fig.add_trace(go.Scatter(x=carreras,y=puntos_martin,mode='lines+markers',line=dict(color=color_martin),name='Martin'))
        width = '900px'
    elif valor_seleccionado == 'bet':
        carreras_bet = np.insert(df_bet['Carrera'].unique(), 0, 'Hasiera', axis=0)
        # balances = list(accumulate(np.insert(df_bet['balance'].unique(), 0, 375, axis=0)))
        balances = np.insert(df_bet.groupby('Carrera', as_index=False,sort=False).first()['total'], 0, 375, axis=0)
        # balances[np.where(carreras_bet == "Australia")[0][0]] = 3.75
        # balances[np.where(carreras_bet == "Etoile de Besseges 5")[0][0]] = 4.21
        # balances[np.where(carreras_bet == "UAE 2")[0][0]] = 5.68  
        # balances[np.where(carreras_bet == "Algarve 5")[0][0]] = 6.62
        balances_25 = [round(x / 15,2) for x in balances]
        balances_50 = [round(x / 7.5,2) for x in balances]
        apostado_total = df_bet['apostado_total'].unique()
        # Rentabilidad = [round(((balances[i] - balances[i - 1])/ apostado_total[i-1])*100,2) for i in range(1, len(balances))]
        Rentabilidad = np.insert(df_bet.groupby('Carrera', as_index=False,sort=False).first()['Rentabilidad'],0, 0, axis= 0)
        # Rentabilidad[np.where(carreras_bet == "UAE 2")[0][0]] = 34.5
        # Rentabilidad[np.where(carreras_bet == "Algarve 5")[0][0]] = 16.54

        fig = go.Figure()

        fig.add_trace(go.Scatter(x = carreras_bet, y = balances_25,mode='lines+markers',line=dict(color='seagreen'),name='1.Taldea'))

        fig.add_trace(go.Scatter(x = carreras_bet, y = balances_50,mode='lines+markers',line=dict(color='darkviolet'),name='2.Taldea'))

        fig.add_trace(go.Bar(x = carreras_bet, y = Rentabilidad, name = 'Errent.', text = Rentabilidad, 
                            marker_color='indianred',textposition='outside'))
        width = '1180px'


    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=10),
    )

    return fig, {'width': width, 'height': '330px'}

@callback( #BARRAS ---------------------------------------------------------------------------------------
Output('barras', 'figure'),
Output('barras', 'style'),
Input("segmented-value", "value")
)
def update_indicator(valor_seleccionado):
    if valor_seleccionado == 'tropela':
        nombres = df_tropela.loc[df_tropela['Anio'] == anio].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Quien',ascending=False)['Quien'].tolist()
        pts_gv = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Carrera'].isin(gv))].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Quien',ascending=False)['Puntos'].tolist()
        pts_semana = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Carrera'].isin(una_semana))].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Quien',ascending=False)['Puntos'].tolist()
        pts_triptico = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Carrera'].isin(tripticos))].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Quien',ascending=False)['Puntos'].tolist()
        pts_clasica = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Carrera'].isin(clasicas))].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Quien',ascending=False)['Puntos'].tolist()

        fig = go.Figure()

        fig.add_trace(go.Bar(y = nombres, x = pts_gv, name = 'Itzuli handiak', orientation = 'h', text = pts_gv,
                            marker = dict(color = 'rgba(246, 78, 139, 0.6)', line = dict(color = 'rgba(246, 78, 139, 1.0)', width = 3))))

        fig.add_trace(go.Bar(y = nombres, x = pts_semana, name = 'Aste batekoak', orientation = 'h', text = pts_semana,
                            marker = dict(color = 'rgba(81, 220, 53, 0.6)', line = dict(color = 'rgba(81, 220, 53, 1.0)', width = 3))))

        fig.add_trace(go.Bar(y = nombres, x = pts_triptico, name = 'Hirukoak', orientation = 'h', text = pts_triptico,
                            marker = dict(color = 'rgba(59, 220, 226, 0.6)', line = dict(color = 'rgba(59, 220, 226, 1.0)', width = 3))))

        fig.add_trace(go.Bar(y = nombres, x = pts_clasica, name = 'Klasikak', orientation = 'h', text = pts_clasica,
                            marker = dict(color = 'rgba(231, 116, 15, 0.6)', line = dict(color = 'rgba(231, 116, 15, 1.0)', width = 3))))
        width = '500px'
        
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        barmode='stack'
        )

    elif valor_seleccionado == 'bet':
        carreras_bet = np.insert(df_bet['Carrera'].unique(), 0, 'Hasiera', axis=0)
        balances = np.insert(df_bet.groupby('Carrera', as_index=False,sort=False).first()['total'], 0, 375, axis=0)
        # balances = list(accumulate(np.insert(df_bet['balance'].unique(), 0, 375, axis=0)))
        # balances[np.where(carreras_bet == "Australia")[0][0]] = 3.75
        # balances[np.where(carreras_bet == "Etoile de Besseges 5")[0][0]] = 4.21
        # balances[np.where(carreras_bet == "UAE 2")[0][0]] = 5.68  
        # balances[np.where(carreras_bet == "Algarve 5")[0][0]] = 6.62
        apostado_total = df_bet['apostado_total'].unique()
        # Rentabilidad = [round(((balances[i] - balances[i - 1])/ apostado_total[i-1])*100,2) for i in range(1, len(balances))]
        # Rentabilidad[np.where(carreras_bet == "UAE 2")[0][0]-1] = 34.5
        # Rentabilidad[np.where(carreras_bet == "Algarve 5")[0][0]-1] = 16.54
        Rentabilidad = np.insert(df_bet.groupby('Carrera', as_index=False,sort=False).first()['Rentabilidad'],0, 0, axis= 0)
        rent_pos = round(statistics.mean([x for x in Rentabilidad if x > 0]),2)
        rent_neg = round(statistics.mean([x for x in Rentabilidad if x < 0]),2)
        fig = make_subplots(rows=3, cols=2)

        fig.add_trace(go.Indicator(
        mode = "number+delta",
        delta = {'reference': 25},
        value = round(balances[-1] / 15,2),
        number = {'suffix': "€","font":{"size":25}},
        title = {'text': "1.Taldea", 'font': {'size':12}},
        domain = {'row': 0, 'column': 0}))
        
        fig.add_trace(go.Indicator(
        mode = "number+delta",
        delta = {'reference': 50},
        value = round(balances[-1] / 7.5,2),
        number = { 'suffix': "€" ,"font":{"size":25}},
        title = {'text': "2.Taldea", 'font': {'size':12}},
        domain = {'row': 0, 'column': 1}))
        
        fig.add_trace(go.Indicator(
        mode = "number",
        value = round(((balances[-1] - 375) / 375)*100,2),
        number = { 'suffix': "%" ,"font":{"size":25}},
        title = {'text': "Errentagarritasuna", 'font': {'size':12}},
        domain = {'row': 1, 'column': 0}))
        
        fig.add_trace(go.Indicator(
        mode = "number",
        value = round(((len(df_bet.index) - len(df_bet.loc[df_bet['Resultado'] == 0].index)) / len(df_bet.index)*100),2),
        number = { 'suffix': "%" ,"font":{"size":25}},
        title = {'text': "Igartze tasa", 'font': {'size':12}},
        domain = {'row': 1, 'column': 1}))

        fig.add_trace(go.Indicator(
        mode = "number",
        value = rent_pos,
        number = { 'suffix': "%" ,"font":{"size":25}},
        title = {'text': "Batez besteko<br /> errentagarritasuna<br /> irabaztean", 'font': {'size':12}},
        domain = {'row': 2, 'column': 0}))
        
        fig.add_trace(go.Indicator(
        mode = "number",
        value = rent_neg,
        number = { 'suffix': "%" ,"font":{"size":25}},
        title = {'text': "Batez besteko<br /> errentagarritasuna<br /> galtzean", 'font': {'size':12}},
        domain = {'row': 2, 'column': 1}))

        width = '260px'
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        grid = {'rows': 3, 'columns': 2, 'pattern': "independent"},
        )

    return fig, {'width': width, 'height': '300px'}

@callback( #BOXPLOT ---------------------------------------------------------------------------------------
Output('boxplot', 'figure'),
Output('boxplot', 'style'),
Input("segmented-value", "value"),
Input('Seleccion_carrera', "value")
)
def update_indicator(valor_seleccionado,carrera_seleccionada):
    if valor_seleccionado == 'tropela':

        fig = go.Figure()

        carreras = df_tropela.loc[df_tropela['Anio'] == anio]['Carrera'].unique()

        fig.add_trace(go.Box(y = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Beñat')]['Porcentaje'].tolist(), 
                            name='Beñat', boxpoints='all', marker_color = color_beniat, line_color = color_beniat,
                            customdata = np.stack((carreras, df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Beñat')]['Porcentaje'].tolist()),axis=-1),
                            hovertemplate= "Lasterketa: %{customdata[0]}<br>" + "Portzentaia: %{customdata[1]}<br>"))

        fig.add_trace(go.Box(y = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Iker')]['Porcentaje'].tolist(), 
                            name='Iker', boxpoints='all', marker_color = color_iker, line_color = color_iker,
                            customdata = np.stack((carreras, df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Iker')]['Porcentaje'].tolist()),axis=-1),
                            hovertemplate= "Lasterketa: %{customdata[0]}<br>" + "Portzentaia: %{customdata[1]}<br>"))

        fig.add_trace(go.Box(y = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Manu')]['Porcentaje'].tolist(),
                            name='Manu', boxpoints='all', marker_color = color_manu, line_color = color_manu,
                            customdata = np.stack((carreras, df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Manu')]['Porcentaje'].tolist()),axis=-1),
                            hovertemplate= "Lasterketa: %{customdata[0]}<br>" + "Portzentaia: %{customdata[1]}<br>"))

        fig.add_trace(go.Box(y = df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Martin')]['Porcentaje'].tolist(),
                            name='Martin', boxpoints='all', marker_color = color_martin, line_color = color_martin,
                            customdata = np.stack((carreras, df_tropela.loc[(df_tropela['Anio'] == anio) & (df_tropela['Quien'] == 'Martin')]['Porcentaje'].tolist()),axis=-1),
                            hovertemplate= "Lasterketa: %{customdata[0]}<br>" + "Portzentaia: %{customdata[1]}<br>"))
        
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        barmode='stack'
        )
        width = '800px'
        height = '250px'
        display = 'block'


    elif valor_seleccionado == 'bet':
        df_giro7 = df_bet.loc[df_bet['Carrera'] == carrera_seleccionada]
        colors = [['lime' if val != 0 else 'tomato' for val in df_giro7['Resultado']] for _ in range(9)]
        fig = go.Figure(data=[go.Table(
        columnwidth = [70,100,35,35,100,100,40,30,60],
        header=dict(values=['Lehiaketa', 'A Txirrindularia', 'A Kuota', 'B Kuota', 'B Txirrindularia', 'Igarpena', 'Portzentaia', 'Emaitza', 'Jokatutako dirua'],
                    fill_color='khaki',
                    align='left'),
        cells=dict(values=[df_giro7.Carrera, df_giro7.Corredor_1, df_giro7.Cuota_1, df_giro7.Cuota_2, df_giro7.Corredor_2, df_giro7.Prediccion, df_giro7.Porcentaje, df_giro7.Resultado, df_giro7.Apostado],
                fill_color=colors,
                align='left'))
        ])

        fig.update_layout(
        margin=dict(l=0,r=15,b=0,t=0),
        barmode='stack'
        )
        width = '1175px'
        height = '250px'
        display = 'block'


    return fig, {'width': width, 'height': height, 'display': display}


@callback( #TABLA ---------------------------------------------------------------------------------------
Output('tabla', 'figure'),
Output('tabla', 'style'),
Input("segmented-value", "value")
)
def update_indicator(valor_seleccionado):

    df_tabla = df_tropela.loc[df_tropela['Anio'] == anio].groupby('Quien').sum('Puntos').reset_index().sort_values(by='Puntos',ascending=False)
    df_tabla['Ranking'] = range(1, len(df_tabla) + 1)
    df_tabla['Ranking'] = pd.to_numeric(df_tabla['Ranking'])
    puntos_lider = max(df_tabla['Puntos'])
    puntos_segundo = max(df_tabla.loc[df_tabla['Ranking'] > int(1)]['Puntos'].tolist())
    df_tabla['Al_lider'] = df_tabla['Puntos'] - puntos_lider
    df_tabla['Al_segundo'] = df_tabla['Puntos'] - puntos_segundo

    if valor_seleccionado == 'tropela':
        fig = go.Figure(data=[go.Table(
            columnwidth = [60,125,80,195,195],
            header=dict(values=['Postua','Lehiakidea','Puntuak','Liderrarekiko distantzia','Bigarrenarekiko distantzia'],
                        fill_color='khaki',
                        align='left',
                        height=23),
            cells=dict(values=[df_tabla.Ranking, df_tabla.Quien, df_tabla.Puntos,df_tabla.Al_lider,
                                df_tabla.Al_segundo],
                    fill_color = [['lime','lawngreen','tomato','red']]*4,
                    align='center',
                    height=23))
        ])

        width = '600px'
        height = '250px'
        display = 'block'
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

    elif valor_seleccionado == 'bet':
        fig = go.Figure(go.Box(x = [0,1,2,3,4,5], marker_color = 'lightseagreen', name= '',boxpoints='all'))

        width = '0px'
        height = '0px'
        display = 'none'

        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

    return fig, {'width': width, 'height': height, 'display': display}

@callback( Output('Seleccion_carrera', 'style'), [Input('segmented-value', 'value')] ) 
def update_dropdown_visibility(slider_value): 
    if slider_value == 'bet': return {'display': 'block'} 
    else: return {'display': 'none'}

@callback( Output('Link_excel', 'style'), [Input('segmented-value', 'value')] ) 
def update_link_visibility(slider_value): 
    if slider_value == 'bet': return {'display': 'block'} 
    else: return {'display': 'none'}


if __name__ == '__main__':
    app.run(debug=True)
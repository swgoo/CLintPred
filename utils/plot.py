import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
from plotly.subplots import make_subplots

rangeLabel_col = ["MW_range", "PSA_range", "NRB_range", "HBA_range", "HBD_range", "LogP_range"]
superscript_map = {
    0: "\u2070",
    1: "\u00B9",
    2: "\u00B2",
    3: "\u00B3",
    4: "\u2074",
    5: "\u2075",
    6: "\u2076",
    7: "\u2077",
    8: "\u2078",
    9: "\u2079"
}


def draw_plot(result_path:str, save_path:str):
    ## -- predict plot save -- ##
    # df_plot = pd.read_csv(os.path.join("../", result_path))
    df_plot = pd.read_csv(result_path)
    labels, predict = df_plot['Clint'], df_plot['predict']
    x_ax = list(range(len(labels)))

    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Scatter(x=x_ax, y=labels,
                        mode='lines',
                        name='labels',
                        legendgroup="1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_ax, y=predict,
                        mode='lines',
                        name='predicted',
                        legendgroup="1"), row=1, col=1)


    ## -- Compare model result Scatter plot -- ##
    model = LinearRegression().fit(np.array(labels).reshape(-1,1), (np.array(predict)))
    y_hat = model.predict(np.array(labels).reshape(-1,1))

    fig.add_trace(go.Scatter(x=labels, y=predict,
                            mode='markers',
                            legendgroup="2"), row=2, col=1)

    fig.add_trace(go.Scatter(x=labels, y=y_hat,
                            mode="lines",
                            name="trend line",
                            line=dict(shape="linear", color="red", width=2, dash="dot"),
                            legendgroup="2"), row=2, col=1)


    ideal_list = list(range(-1, int(max(labels)) + 3))
    ideal_x, ideal_y = ideal_list, ideal_list

    ideal_x_rev = ideal_x[::-1]
    ideal_y_uper,ideal_y_lower = [val+1 for val in ideal_y], [val-1 for val in ideal_y]
    ideal_y_lower = ideal_y_lower[::-1]

    ideal_y_uper_dot,ideal_y_lower_dot = [val+0.5 for val in ideal_y], [val-0.5 for val in ideal_y]
    ideal_y_lower_dot = ideal_y_lower_dot[::-1]

    fig.add_trace(go.Scatter(x=ideal_x, y=ideal_y,
                            mode="lines",
                            name="ideal",
                            line=dict(shape="linear", color="black", width=2, dash="dot"),
                            legendgroup="2"), row=2, col=1)

    fig.add_trace(go.Scatter(x=ideal_x + ideal_x_rev, y=ideal_y_uper + ideal_y_lower,
                            mode="lines",
                            name="margin 1",
                            line=dict(shape="linear", color='rgba(231,107,243,0.8)', width=2, dash="dashdot"),
                            legendgroup="2"), row=2, col=1)

    fig.add_trace(go.Scatter(x=ideal_x + ideal_x_rev, y=ideal_y_uper_dot + ideal_y_lower_dot,
                            mode="lines",
                            name="margin 2",
                            line=dict(shape="linear", color='rgba(0,176,246,0.8)', width=2, dash="dash"),
                            legendgroup="2"), row=2, col=1)

    fig.update_layout(
        height=800, 
        width=800, 
        title_text="Compare Observed Clint to Prediction results", 
        xaxis1_title = 'Data Length',
        yaxis1_title = 'Clint_logScale',
        xaxis2_title = 'Observed',
        yaxis2_title = 'Predicted',
        legend_tracegroupgap = 320,
        xaxis2_range=[-0.5, int(max(labels)) + 1],
        yaxis2_range=[-0.5, int(max(labels)) + 1],
    )
    
    # fig.write_image(os.path.join("../", save_path))
    fig.write_image(save_path)


def draw_boundaryplot(df_plot:pd.DataFrame, save_path:str, df_duplData:pd.DataFrame = None):
    labels, predict = np.expm1(df_plot["Clint"]), np.expm1(df_plot["predict"])
    labels, predict = np.log10(labels), np.log10(predict)

    # labels, predict = df_plot["Clint"], df_plot["predict"]
        
    variance_list =[abs(data - labels[idx])/predict.mean() for idx, data in enumerate(predict)]
    df_plot["variance"] = variance_list

    variance_boundary = 0.5
    in_labels, in_predict = df_plot[df_plot["variance"] <= variance_boundary]['Clint'], df_plot[df_plot["variance"] <= variance_boundary]['predict']
    out_labels, out_predict = df_plot[df_plot["variance"] > variance_boundary]['Clint'], df_plot[df_plot["variance"] > variance_boundary]['predict']

    x_extended = np.linspace(df_plot['Clint'].min() - 1, df_plot['Clint'].max() + 1, 200)

    grad, bias = np.polyfit(labels, predict, 1)  
    yhat = grad* x_extended + bias

    in_grad, in_bias = np.polyfit(in_labels, in_predict, 1)  
    in_yhat = in_grad* x_extended + in_bias

    ## -- y축: 예측치, x축: 관측치 -- ##
    scatter = go.Scatter(x=labels, y=predict,
                            mode='markers',
                            name="Predict_point")

    out_scatter = go.Scatter(x=out_labels, y=out_predict,
                            mode='markers',
                            name="out_bound",
                            )

    trend_line = go.Scatter(x=x_extended, y=yhat,
                            mode="lines",
                            name="trend line",
                            line=dict(shape="linear", color="red", width=2, dash="dot"))

    in_trend_line = go.Scatter(x=x_extended, y=in_yhat,
                            mode="lines",
                            name="in-variance trend line",
                            line=dict(shape="linear", color="olive", width=2, dash="dot"))

    ## -- log scale된 데이터의 Ideal 기울기와 fold 기울기 -- ##
    ideal_list = list(range(-1, int(max(labels)) + 3))
    ideal_x, ideal_y = ideal_list, ideal_list

    ideal_line = go.Scatter(x=ideal_x, y=ideal_y,
                                mode="lines",
                                name="ideal",
                                line=dict(shape="linear", color="black", width=2))
        
    ## -- 2-fold 기울기 : 관측치 대비 예측치가 /2 ~ *2인 범위
        
    fold2_top = [x + np.log10(2.0) for x in ideal_list]
    fold2_bottom = [x - np.log10(2.0) for x in ideal_list]

    fold2_topline = go.Scatter(x=ideal_list, y=fold2_top,
                            mode="lines",
                            name="2-fold",
                            line=dict(shape="linear", color="black", width=2, dash="dash"))
    fold2_bottomline = go.Scatter(x=ideal_list, y=fold2_bottom,
                            mode="lines",
                            name="2-fold",
                            showlegend =False,
                            line=dict(shape="linear", color="black", width=2, dash="dash"))


    fold5_top = [x + np.log10(5.0) for x in ideal_list]
    fold5_bottom = [x - np.log10(5.0) for x in ideal_list]

    fold5_topline = go.Scatter(x=ideal_x, y=fold5_top,
                            mode="lines",
                            name="5-fold",
                            line=dict(shape="linear", color="black", width=2, dash="dot"))
    fold5_bottomline = go.Scatter(x=ideal_x, y=fold5_bottom,
                            mode="lines",
                            name="5-fold",
                            showlegend =False,
                            line=dict(shape="linear", color="black", width=2, dash="dot"))

    fig = go.Figure(data=[scatter, out_scatter, trend_line, in_trend_line, ideal_line, 
                        fold2_topline, fold2_bottomline, fold5_topline, fold5_bottomline])

    axis_map = [f'10{superscript_map[0]}', f'10{superscript_map[1]}', f'10{superscript_map[2]}', f'10{superscript_map[3]}']

    # x축 제목과 폰트 설정
    fig.update_xaxes(title_text='Observed Clint', title_font=dict(size=18))

    # y축 제목과 폰트 설정
    fig.update_yaxes(title_text='Predicted Clint', title_font=dict(size=18))

    fig.update_layout(
        height=800, 
        width=900, 
        title_text="", 
        legend=dict(font=dict(size=16)),
        legend_tracegroupgap = 320,
        xaxis_range=[0.0, int(max(labels)) + 1],
        yaxis_range=[0.0, int(max(labels)) + 1],
        plot_bgcolor='white',  # Set the background color to white
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey', 
            tickfont=dict(size=18),
            tickvals = [0.0, 1.0, 2.0, 3.0],
            ticktext = axis_map,
            gridwidth=1, 
            griddash='dash',  # Set the x-axis grid lines to light grey dashed
            linecolor='black',  # Set x-axis line color to black
            linewidth=2,  # Set x-axis line width
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey', 
            gridwidth=1, 
            tickfont=dict(size=18),
            tickvals = [0.0, 1.0, 2.0, 3.0],
            ticktext = axis_map,
            griddash='dash',  # Set the y-axis grid lines to light grey dashed
            linecolor='black',  # Set y-axis line color to black
            linewidth=2,  # Set y-axis line width
        )
    )

    fig.write_image(save_path)
    
    return grad, in_grad
    


def draw_boxplot(result_path:str, save_path:str):
    ## -- predict plot save -- ##
    x_axisName = ["molecular weight", "PSA", "Number of Rotatable Bonds",
                  "Number of Hydrogen-Bond Acceptors", "Number of Hydrogen-Bond Donors", "LogP"]
    
    df_plot = pd.read_csv(result_path)
    y_labels, y_predict = df_plot['Clint'], df_plot['predict']
    fig = make_subplots(rows=3, cols=2)

    for idx, axisName in enumerate(x_axisName):
        x_labels, x_predict = df_plot[rangeLabel_col[idx]], df_plot[rangeLabel_col[idx]]
        row_numb, col_numb = (idx // 2) + 1, (idx % 2) + 1

        fig.add_trace(go.Box(x=x_labels, y=y_labels,
                            name="Label",
                            jitter=0.3,
                            pointpos=-1.8,
                            boxpoints='all', # represent all points
                            marker_color='rgb(7,40,89)',
                            line_color='rgb(7,40,89)',
                            legendgroup="1"), row=row_numb, col=col_numb)
        
        fig.add_trace(go.Scatter(x=x_predict, y=y_predict,
                                 name="Prediction",
                                 jitter=0.3,
                                 pointpos=-1.8,
                                 boxpoints='all', # represent all points
                                 marker_color='rgb(7,40,89)',
                                 line_color='rgb(7,40,89)',
                                 legendgroup="1"), row=row_numb, col=col_numb)
            
        fig.update_xaxes(title_text=x_axisName[idx], row=row_numb, col=col_numb)
        fig.update_yaxes(title_text='Intrinsic CL(Log Scale)', row=row_numb, col=col_numb)


    fig.update_layout(
        height=800, 
        width=800, 
        title_text="Compare Observed Clint to Prediction results", 
        legend_tracegroupgap = 320,
    )
    
    # fig.write_image(os.path.join("../", save_path))
    fig.write_image(save_path)
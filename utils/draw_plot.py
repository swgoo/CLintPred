import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
from plotly.subplots import make_subplots

rangeLabel_col = ["MW_range", "PSA_range", "NRB_range", "HBA_range", "HBD_range", "LogP_range"]

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
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_distributionPlot(result_path:str, save_path:str):
    df_plot = pd.read_csv(result_path)
    labels, predict = df_plot['Clint'], df_plot['predict']
    x_ax = list(range(len(labels)))
# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import constants as c

from statsmodels.tsa.seasonal import seasonal_decompose

class Data:

    def __init__(self) -> None:
        self.hdfc_five_min_df = None
        self.reliance_five_min_df = None
        self.sunpharma_five_min_df = None
        

    def read_data(self) -> str:
        # Reading data of all selected stocks

        # 5 min data of HDFC Bank from the 2015 to 2022
        self.hdfc_five_min_df = pd.read_csv('/source/data/HDFCBANK_with_indicators_.csv')#[chosen_columns]
        # 5 min data of Reliance Industries from the 2015 to 2022
        self.reliance_five_min_df = pd.read_csv('/source/data/RELIANCE_with_indicators_.csv')#[chosen_columns]#[['date', 'close']]
        # 5 min data of Sun Pharmaceutical Industries from the 2015 to 2022
        self.sunpharma_five_min_df = pd.read_csv('/source/data/SUNPHARMA_with_indicators_.csv')#[chosen_columns]#[['date', 'close']]

        return (f"HDFC Bank: {self.hdfc_five_min_df.shape}\nReliance Industries: {self.reliance_five_min_df.shape}\nSun Pharmaceutical Industries:\
            self.{self.sunpharma_five_min_df.shape}")
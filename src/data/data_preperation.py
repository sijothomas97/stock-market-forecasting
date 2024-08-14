import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


class Data_preperation:

    def __init__(self, obj_dpp) -> None:
        self.monthly_hdfc_data = None
        self.monthly_reliance_data = None
        self.monthly_sunpharma_data = None
        self.obj_dpp = obj_dpp


    def seasonal_dcompose(self, close, arg_model, arg_period) -> tuple[pd.series]:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(close, model=arg_model, period=arg_period)  # Assuming monthly data (period=12)

        # Extract and return the trend, seasonal, and residual components
        return decomposition.trend, decomposition.seasonal, decomposition.resid


    def decomposition(self) -> None:
        hdfc_trend, hdfc_seasonal, hdfc_residual = self.seasonal_dcompose(self.obj_dpp.monthly_hdfc_data.close, 'additive', 12)
        reliance_trend, reliance_seasonal, reliance_residual = self.seasonal_dcompose(self.obj_dpp.monthly_reliance_data.close, 'additive', 12)
        sunpharma_trend, sunpharma_seasonal, sunpharma_residual = self.seasonal_dcompose(self.obj_dpp.monthly_sunpharma_data.close, 'additive', 12)

    
    def test_stationarity(self, time_series) -> list:
        # Calculate rolling statistics
        rolling_mean = time_series.rolling(window=12).mean()
        rolling_std = time_series.rolling(window=12).std()

        # Perform ADF test and return test result
        return adfuller(time_series, autolag='AIC')

        # Print ADF test results
        # print('ADF Statistic:', adf_test[0])
        # print('p-value:', adf_test[1])
        # print('Critical Values:')
        # for key, value in adf_test[4].items():
        #     print(f'   {key}: {value}')

        # -- for main.py --
        # Assuming 'monthly_hdfc_data' is your time series DataFrame
        # test_stationarity(monthly_hdfc_data.close)

    def first_diff(self, df) -> pd.Series:
        return df['close'].diff()
    
        # -- for main.py --
        #  monthly_hdfc_data['first_difference'] = monthly_hdfc_data['close'].diff()
        # monthly_reliance_data['first_difference'] = monthly_reliance_data['close'].diff()
        # monthly_sunpharma_data['first_difference'] = monthly_sunpharma_data['close'].diff()     

    def drop_rows(self, df) -> pd.DataFrame:
        return df.dropna(how='any', axis=0)  # Removes rows with missing values
    
        # for -- main.py --
        # monthly_hdfc_data = obj.drop_rows(monthly_hdfc_data)
        # monthly_reliance_data = obj.drop_rows(monthly_reliance_data)
        # monthly_sunpharma_data = obj.drop_rows(monthly_sunpharma_data)
    

    def prepare_data(self) -> str:

        self.monthly_hdfc_data['first_difference'] = self.first_diff(self.obj_dpp.monthly_hdfc_data)
        self.monthly_reliance_data['first_difference'] = self.first_diff(self.obj_dpp.monthly_reliance_data)
        self.monthly_sunpharma_data['first_difference'] = self.first_diff(self.obj_dpp.monthly_sunpharma_data)

        self.monthly_hdfc_data = self.drop_rows(self.monthly_hdfc_data)
        self.monthly_reliance_data = self.drop_rows(self.monthly_reliance_data)
        self.monthly_sunpharma_data = self.drop_rows(self.monthly_sunpharma_data)

        return 'success'
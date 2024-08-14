import itertools
# import warningsz
import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import *
from statsmodels.tsa.api import ARDL
from sklearn.model_selection import train_test_split
import arch  # Import the arch library
import pandas as pd


class Train:

    def __init__(self, obj_dpp) -> None:
        self.arima_mdl_hdfc = None
        self.arima_mdl_rel = None
        self.arima_mdl_sun = None

        self.ardl_mdl_hdfc = None
        self.ardl_mdl_rel = None
        self.ardl_mdl_sun = None

        self.arch_mdl_hdfc = None
        self.arch_mdl_rel = None
        self.arch_mdl_sun = None

        self.garch_mdl_hdfc = None
        self.garch_mdl_rel = None
        self.garch_mdl_sun = None

        self.ln_column_names = None
        self.obj_dpp = obj_dpp


    def first_diff_correlations(self, df) -> list:
        # Calculate correlations between 'close' and other columns with different lags
        df_copy = self.obj_dpp.df.copy()
        # monthly_hdfc_data_copy = self.obj_dpp.monthly_hdfc_data.copy()
        first_diff_correlations = {}
        corr_cols = ['close']

        for col in self.obj_dpp.monthly_hdfc_data.columns:
            if col != 'close':
                df_copy[f'{col}_first_diff'] = df_copy['close'].diff()
                correlation = df_copy['close'].corr(df_copy[f'{col}_first_diff'])
                # monthly_hdfc_data_copy[f'{col}_first_diff'] = monthly_hdfc_data_copy['close'].diff()
                # correlation = monthly_hdfc_data_copy['close'].corr(monthly_hdfc_data_copy[f'{col}_first_diff'])
                first_diff_correlations[f'{col}_first_diff'] = correlation
                corr_cols.append(f'{col}_first_diff')


        # Sort the lagged variables by correlation
        return sorted(first_diff_correlations.items(), key=lambda x: abs(x[1]), reverse=True), df_copy
        # sorted_ln_variables = sorted(first_diff_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        # Print the top 10 lagged variables with the highest absolute correlation
        # for ln_variable, correlation in sorted_ln_variables[:10]:
        #     ret_lst.append(f'{ln_variable}: {correlation}')

        # return ret_lst
        # Create a heatmap to visualize the correlation matrix
        # plt.figure(figsize=(16, 10))
        # sns.heatmap(monthly_hdfc_data_copy[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        # plt.title('Correlation Matrix')
        # plt.show()


    def init_exog(self, df_copy):
        # Extract column names
        # ln_column_names = [item[0] for item in sort_ln_vars[:5]]
        df_copy = df_copy.dropna()
        # ln_column_names

        for col in self.self.ln_column_names:
            self.obj_dpp.test_stationarity(df_copy[col])

        return df_copy[self.ln_column_names]
        # exog = monthly_hdfc_data_copy[ln_column_names]


    def train_test_split(self, df, alg):
        #split data into train and training set
        if alg is 'arima':
            return df[3:-11], df[-12:]
        elif alg is 'ardl':
            test_size = 0.2  # You can adjust the test size as needed
            exog = self.init_exog(df)
            return train_test_split(exog, df.close, test_size=test_size, shuffle=False)
        
    
    def returns_vltlty(self, df) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['actual_volatility'] = df['returns'].rolling(window=20).std()  # Calculate rolling 20-day volatility
        df['volatility'] = df['returns'].std()

        return df.dropna()
    

    def train_arima(self, train_df, test_df):
        
        y_to_train = train_df['first_difference'] # dataset to train
        y_to_test = test_df['first_difference'] # last X months for test
        y_to_val = y_to_test

        # warnings.filterwarnings("ignore") # specify to ignore warning messages

        p = d = q = range(0, 2)
        seasonal_period = 12
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]


        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y_to_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_invertibility=False)
                    results_hdfc = mod.fit()
                except Exception as ex:
                    continue

        order = (0, 0, 1) #Using Values from the previous step
        seasonal_order = (1, 1, 1, 12)
        model = sm.tsa.statespace.SARIMAX(y_to_train,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_invertibility=False)
        
        return model.fit()
        # results_hdfc = model.fit()
        # return (results_hdfc.summary())
    

    def train_ardl(self, df) -> None:
        sort_ln_vars, df_copy = self.first_diff_correlations(df)
        self.ln_column_names = [item[0] for item in sort_ln_vars[:5]]

        # Split your data into a training set and a testing set
        X_train, X_test, y_train, y_test = self.train_test_split(df_copy, 'ardl')
        # X_train, X_test, y_train, y_test = train_test_split(exog, monthly_hdfc_data_copy, test_size=test_size, shuffle=False)

        # Train the ARDL model on the training data
        model = ARDL(y_train, lags=2, exog=X_train, order=(2, 0))
        return model.fit()
        # ardl_model = model.fit()
        # return ardl_model.summary()
    

    def train_arch(self, df):
        # Calculate daily returns
        df = self.returns_vltlty(df)

        # Create an ARCH model
        model = arch.arch_model(df['returns'], vol='ARCH', p=1)

        # Fit the model
        return model.fit()
        # results = model.fit()
    

    def train_garch(self, df):
        # Calculate daily returns
        df = self.returns_vltlty(df)

        # Create an ARCH model
        model = arch.arch_model(df['returns'], vol='Garch', p=1, q=1)

        # Fit the model
        return model.fit()
        

    def train(self) -> str:
        # Model Trainin and Fitting -- ARIMA --
        algo_nme = 'arima'
        train_hdfc, test_hdfc = self.train_test_split(self.obj_dpp.monthly_hdfc_data, algo_nme)
        train_rel, test_rel = self.train_test_split(self.obj_dpp.monthly_reliance_data, algo_nme)
        train_sun, test_sun = self.train_test_split(self.obj_dpp.monthly_sunpharma_data, algo_nme)

        self.arima_mdl_hdfc = self.train_arima(train_hdfc, test_hdfc)
        self.arima_mdl_rel = self.train_arima(train_rel, test_rel)
        self.arima_mdl_sun = self.train_arima(train_sun, test_sun)

        # Model Trainin and Fitting -- ARDL --
        # algo_nme = 'ardl'
        # train_hdfc, test_hdfc = self.train_test_split(self.obj_dpp.monthly_hdfc_data, algo_nme)
        # train_rel, test_rel = self.train_test_split(self.obj_dpp.monthly_reliance_data, algo_nme)
        # train_sun, test_sun = self.train_test_split(self.obj_dpp.monthly_sunpharma_data, algo_nme)

        self.ardl_mdl_hdfc = self.train_ardl(self.obj_dpp.monthly_hdfc_data)
        self.ardl_mdl_rel = self.train_ardl(self.obj_dpp.monthly_reliance_data)
        self.ardl_mdl_sun = self.train_ardl(self.obj_dpp.monthly_sunpharma_data)

        # Model Trainin and Fitting -- ARCH --

        self.arch_mdl_hdfc = self.train_arch(self.obj_dpp.monthly_hdfc_data)
        self.arch_mdl_rel = self.train_arch(self.obj_dpp.monthly_reliance_data)
        self.arch_mdl_sun = self.train_arch(self.obj_dpp.monthly_sunpharma_data)

        # Model Trainin and Fitting -- GARCH --

        self.garch_mdl_hdfc = self.train_arch(self.obj_dpp.monthly_hdfc_data)
        self.garch_mdl_rel = self.train_arch(self.obj_dpp.monthly_reliance_data)
        self.garch_mdl_sun = self.train_arch(self.obj_dpp.monthly_sunpharma_data)

        return 'success'

import numpy as np
import constants as c

class Forecast:

    def __init__(self, obj_train, obj_dpp) -> None:
        self.frcst_rslt_garch_hdfc = None
        self.frcst_cp_garch_hdfc = None
        self.frcst_rslt_garch_rel = None
        self.frcst_cp_garch_rel = None
        self.frcst_rslt_garch_sun = None
        self.frcst_cp_garch_sun = None

        self.frcst_rslt_arch_hdfc = None
        self.frcst_cp_arch_hdfc = None
        self.frcst_rslt_arch_rel = None
        self.frcst_cp_arch_rel = None
        self.frcst_rslt_arch_sun = None
        self.frcst_cp_arch_sun = None

        self.obj_train = obj_train
        self.obj_dpp = obj_dpp

    
    def close_price(self, df, frcst):
        # Get the last observed volatility
        last_volatility = df['returns'].rolling(window=20).std().dropna()

        # Generate a simulated return series based on the forecasted volatility
        simulated_returns = np.random.normal(scale=np.sqrt(frcst.variance.values[-1, :]), size=c.FORECAST_HORIZON)

        # Calculate and return the forecasted close price
        return df['close'].iloc[-1] * np.exp(simulated_returns.cumsum())
        # forecasted_close_price = df['close'].iloc[-1] * np.exp(simulated_returns.cumsum())


    def forecast_garch(self, mdl):
        # Forecast volatility
        # c.FORECAST_HORIZON = 20  # Number of days to forecast
        return mdl.forecast(start=None, horizon=c.FORECAST_HORIZON)
        # pred_variances = frcst_rslt.variance.values[-1, :]

    def forecast_ardl(self, mdl, ytr, yts, xts):
        # Use the trained model to make predictions on the testing data
        return mdl.predict(start=len(ytr), end=len(ytr) + len(yts) - 1, exog_oos=xts)
        # y_pred = ardl_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog_oos=X_test)


    def forecast(self):

        # -- Forecasting with GARCH --
        self.frcst_rslt_garch_hdfc = self.forecast_garch(self.obj_train.garch_mdl_hdfc)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_garch_hdfc = self.close_price(self.obj_dpp.monthly_hdfc_data, self.frcst_rslt_garch_hdfc)
        # Extract and return the forecasted conditional variances
        pred_variance_garch_hdfc = self.frcst_rslt_garch_hdfc.variance.values[-1, :]

        self.frcst_rslt_garch_rel = self.forecast_garch(self.obj_train.garch_mdl_rel)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_garch_rel = self.close_price(self.obj_dpp.monthly_reliance_data, self.frcst_rslt_garch_rel)
        # Extract and return the forecasted conditional variances
        pred_variance_garch_rel = self.frcst_rslt_garch_rel.variance.values[-1, :]

        self.frcst_rslt_garch_sun = self.forecast_garch(self.obj_train.garch_mdl_sun)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_garch_sun = self.close_price(self.obj_dpp.monthly_sunpharma_data, self.frcst_rslt_garch_sun)
        # Extract and return the forecasted conditional variances
        pred_variance_garch_sun = self.frcst_rslt_garch_sun.variance.values[-1, :]

        # -- Forecasting with ARCH --

        self.frcst_rslt_arch_hdfc = self.forecast_garch(self.obj_train.arch_mdl_hdfc)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_arch_hdfc = self.close_price(self.obj_dpp.monthly_hdfc_data, self.frcst_rslt_arch_hdfc)
        # Extract and return the forecasted conditional variances
        pred_variance_arch_hdfc = self.frcst_rslt_arch_hdfc.variance.values[-1, :]

        self.frcst_rslt_arch_rel = self.forecast_garch(self.obj_train.arch_mdl_rel)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_arch_rel = self.close_price(self.obj_dpp.monthly_reliance_data, self.frcst_rslt_arch_rel)
        # Extract and return the forecasted conditional variances
        pred_variance_arch_rel = self.frcst_rslt_arch_rel.variance.values[-1, :]

        self.frcst_rslt_arch_sun = self.forecast_garch(self.obj_train.arch_mdl_sun)
        # Get forecasted closed price based on forecasted volatility.
        self.frcst_cp_arch_sun = self.close_price(self.obj_dpp.monthly_sunpharma_data, self.frcst_rslt_arch_sun)
        # Extract and return the forecasted conditional variances
        pred_variance_arch_sun = self.frcst_rslt_arch_sun.variance.values[-1, :]

        # -- Forecasting with ARDL --

        sort_ln_vars, df_copy = self.obj_train.first_diff_correlations(self.obj_dpp.monthly_hdfc_data)
        X_train, X_test, y_train, y_test = self.obj_train.train_test_split(df_copy, 'ardl')
        self.forecast_ardl(self.obj_train.ardl_mdl_hdfc, y_train, y_test, X_test)



        
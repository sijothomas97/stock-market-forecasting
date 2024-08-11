import pandas as pd
import constants as c

class Preprocessing:

    def __init__(self, obj_data) -> None:
        self.monthly_hdfc_data = None
        self.monthly_reliance_data = None
        self.monthly_sunpharma_data = None
        self.obj_data = obj_data

    def feature_selection(self) -> list[str]:
        # Delete unwanted features from the dataset
        drop_cols = c.DROP_COLS
        self.obj_data.hdfc_five_min_df = self.obj_data.hdfc_five_min_df.drop(columns = drop_cols)
        self.obj_data.reliance_five_min_df = self.obj_data.reliance_five_min_df.drop(columns = drop_cols)
        self.obj_data.sunpharma_five_min_df = self.obj_data.sunpharma_five_min_df.drop(columns = drop_cols)

        return (self.obj_data.hdfc_five_min_df.columns)

    def del_time_zone(self) -> str:
        # Deleting the the time zone part from the date column
        # HDFC Bank
        self.obj_data.hdfc_five_min_df['date'] = pd.to_datetime(self.obj_data.hdfc_five_min_df['date'], format='%Y-%m-%d')
        # dt.tz_convert method is a part of the datetime module and is used to convert the time zone of a datetime object to a different time zone.
        self.obj_data.hdfc_five_min_df['date'] = self.obj_data.hdfc_five_min_df['date'].dt.tz_convert(None)
        self.obj_data.hdfc_five_min_df['date'] = self.obj_data.hdfc_five_min_df.set_index('date', inplace=True)

        # Reliance Industries
        self.obj_data.reliance_five_min_df['date'] = pd.to_datetime(self.obj_data.reliance_five_min_df['date'], format='%Y-%m-%d')
        # dt.tz_convert method is a part of the datetime module and is used to convert the time zone of a datetime object to a different time zone.
        self.obj_data.reliance_five_min_df['date'] = self.obj_data.reliance_five_min_df['date'].dt.tz_convert(None)
        self.obj_data.reliance_five_min_df['date'] = self.obj_data.reliance_five_min_df.set_index('date', inplace=True)

        # Sun Pharmaceutical Industries
        self.obj_data.sunpharma_five_min_df['date'] = pd.to_datetime(self.obj_data.sunpharma_five_min_df['date'], format='%Y-%m-%d')
        # dt.tz_convert method is a part of the datetime module and is used to convert the time zone of a datetime object to a different time zone.
        self.obj_data.sunpharma_five_min_df['date'] = self.obj_data.sunpharma_five_min_df['date'].dt.tz_convert(None)
        self.obj_data.sunpharma_five_min_df['date'] = self.obj_data.sunpharma_five_min_df.set_index('date', inplace=True)

        # data.set_index('date', inplace=True)
        return (self.obj_data.hdfc_five_min_df.head(1).index, self.obj_data.reliance_five_min_df.head(1).index, self.obj_data.sunpharma_five_min_df.head(1).index)

    def change_interval(self) -> str:
        # Resampling data from 5 min intreval to 1 day interval

        # Specify the columns you want to resample
        columns_to_resample = c.COLS_TO_RSML
        columns_to_agg = c.COLS_TO_AGG

        # Resample the DataFrame to 1-day intervals (OHLCV data)
        self.monthly_hdfc_data = self.obj_data.hdfc_five_min_df[columns_to_resample].resample('M').agg(columns_to_agg)

        # Resample the DataFrame to 1-day intervals (OHLCV data)
        self.monthly_reliance_data = self.obj_data.reliance_five_min_df[columns_to_resample].resample('M').agg(columns_to_agg)

        # Resample the DataFrame to 1-day intervals (OHLCV data)
        self.monthly_sunpharma_data = self.obj_data.sunpharma_five_min_df[columns_to_resample].resample('M').agg(columns_to_agg)

        return (self.monthly_hdfc_data.shape, self.monthly_reliance_data.shape, self.monthly_sunpharma_data.shape)

    def add_year_month(self) -> str:
        # Extracting year and month from column date and adding them as new features to the stock market dataframe.
        self.monthly_hdfc_data['year'] = self.monthly_hdfc_data.index.year
        self.monthly_hdfc_data['month'] = self.monthly_hdfc_data.index.month

        self.monthly_reliance_data['year'] = self.monthly_reliance_data.index.year
        self.monthly_reliance_data['month'] = self.monthly_reliance_data.index.month

        self.monthly_sunpharma_data['year'] = self.monthly_sunpharma_data.index.year
        self.monthly_sunpharma_data['month'] = self.monthly_sunpharma_data.index.month

        return (self.monthly_hdfc_data.shape, self.monthly_reliance_data.shape, self.monthly_sunpharma_data.shape)
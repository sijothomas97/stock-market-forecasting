from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Data_analysis:

    def __init__(self, obj_dpp) -> None:
        self.obj_dpp = obj_dpp


    def candle_stick_analysis(self) -> None:
        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=("HDFC Bank","Reliance Industries", "Sun Pharmaceutical Industries", "Tata Steel", "TCS"))

        fig.add_trace(go.Candlestick(x = self.obj_dpp.monthly_hdfc_data.index,
                        open = self.obj_dpp.monthly_hdfc_data['open'],
                        high = self.obj_dpp.monthly_hdfc_data['high'],
                        low = self.obj_dpp.monthly_hdfc_data['low'],
                        close = self.obj_dpp.monthly_hdfc_data['close'], name="HDFC Bank"),
                    row=1, col=1)

        fig. add_trace(go.Candlestick(x=self.obj_dpp.monthly_reliance_data.index,
                        open = self.obj_dpp.monthly_reliance_data['open'],
                        high = self.obj_dpp.monthly_reliance_data['high'],
                        low = self.obj_dpp.monthly_reliance_data['low'],
                        close = self.obj_dpp.monthly_reliance_data['close'], name="Reliance Industries"),
                    row=2, col=1)

        fig.add_trace(go.Candlestick(x = self.obj_dpp.monthly_sunpharma_data.index,
                        open = self.obj_dpp.monthly_sunpharma_data['open'],
                        high = self.obj_dpp.monthly_sunpharma_data['high'],
                        low = self.obj_dpp.monthly_sunpharma_data['low'],
                        close = self.obj_dpp.monthly_sunpharma_data['close'], name="Sun Pharmaceutical Industries"),
                    row=3, col=1)


        fig.update_layout(height=800, width=800,
                        title_text="Stock prices")
        fig.update_xaxes(rangeslider= {'visible':False}, row=1, col=1)
        fig.update_xaxes(rangeslider= {'visible':False}, row=2, col=1)
        fig.update_xaxes(rangeslider= {'visible':False}, row=3, col=1)

        fig.show()

        # from plotly.subplots import make_subplots
        # import plotly.graph_objects as go


    def vol_analysis(self) -> None:
        fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 8))

        fig.suptitle('Volume of stocks')
        axs[0].set_title("HDFC Bank")
        axs[0].bar(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['volume'], width=15)
        axs[1].set_title("Reliance Industries")
        axs[1].bar(self.obj_dpp.monthly_reliance_data.index, self.obj_dpp.monthly_reliance_data['volume'], width=15)
        axs[2].set_title("Sun Pharmaceutical Industries")
        axs[2].bar(self.obj_dpp.monthly_sunpharma_data.index, self.obj_dpp.monthly_sunpharma_data['volume'], width=15)

        # Show Plot
        plt.show()

        # Circular pie chart
        column_to_analyze = "volume"  # Change to "Falls within" if needed

        hdfc_volume_df = self.obj_dpp.monthly_hdfc_data.groupby('year').sum()['volume']
        reliance_volume_df = self.obj_dpp.monthly_reliance_data.groupby('year').sum()['volume']
        spharma_volume_df = self.obj_dpp.monthly_sunpharma_data.groupby('year').sum()['volume']

        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))

        fig.suptitle('Distribution of Volume over years from 2015 to 2022')
        axs[0][0].set_title("HDFC Bank")
        axs[0][0].pie(hdfc_volume_df, labels=hdfc_volume_df.index, autopct='%1.1f%%', startangle=140)
        axs[0][1].set_title("Reliance Industries")
        axs[0][1].pie(reliance_volume_df, labels=reliance_volume_df.index, autopct='%1.1f%%', startangle=140)
        axs[0][2].set_title("Sun Pharmaceutical Industries")
        axs[0][2].pie(spharma_volume_df, labels=spharma_volume_df.index, autopct='%1.1f%%', startangle=140)

        # plt.axis('equal')
        plt.tight_layout()
        # Show Plot
        plt.show()


    def sma_analysis(self) -> None:
        # Calculate a longer-term SMA (e.g., 30-day SMA)
        self.obj_dpp.monthly_hdfc_data['SMA5'] = self.obj_dpp.monthly_hdfc_data['volume'].rolling(window=5).mean()
        self.obj_dpp.monthly_hdfc_data['SMA10'] = self.obj_dpp.monthly_hdfc_data['volume'].rolling(window=10).mean()
        self.obj_dpp.monthly_hdfc_data['SMA15'] = self.obj_dpp.monthly_hdfc_data['volume'].rolling(window=15).mean()
        self.obj_dpp.monthly_hdfc_data['SMA20'] = self.obj_dpp.monthly_hdfc_data['volume'].rolling(window=20).mean()
        self.obj_dpp.monthly_hdfc_data['SMA30'] = self.obj_dpp.monthly_hdfc_data['volume'].rolling(window=30).mean()

        fig, axs = plt.subplots(5, sharex=True, sharey=True, figsize=(10, 10))

        fig.suptitle('Volume of stocks')
        axs[0].set_title("HDFC Bank")
        axs[0].bar(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['volume'], width=15)
        axs[1].set_title("Reliance Industries")
        axs[1].bar(self.obj_dpp.monthly_reliance_data.index, self.obj_dpp.monthly_reliance_data['volume'], width=15)
        axs[2].set_title("Sun Pharmaceutical Industries")
        axs[2].bar(self.obj_dpp.monthly_sunpharma_data.index, self.obj_dpp.monthly_sunpharma_data['volume'], width=15)
        axs[3].set_title("Tata Steel")

        # Show Plot
        plt.show()

        # fig.show()
        # Create a plot to visualize volume trends
        plt.figure(figsize=(12, 6))
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['volume'], label='Volume', color='black', alpha=0.5)
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['SMA5'], label='5-day SMA', color='blue')
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['SMA10'], label='10-day SMA', color='orange')
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['SMA15'], label='15-day SMA', color='green')
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['SMA20'], label='20-day SMA', color='red')
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['SMA30'], label='30-day SMA', color='yellow')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Trading Volume and Moving Averages')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def macd_analysis(self) -> None:
        # Assuming you have a DataFrame with MACD data including 'date', 'macd', 'signal' columns
        # Create a plot to visualize volume trends
        plt.figure(figsize=(12, 6))
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['macd520'], label='MACD', color='blue', alpha=0.5)
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['macd1020'], label='Signal', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('MACD and Signal Line Chart')
        plt.legend()
        plt.grid(True)
        plt.show()


    def rsi_analysis(self) -> None:
        # Assuming you have a DataFrame with MACD data including 'date', 'macd', 'signal' columns
        # Create a plot to visualize volume trends
        plt.figure(figsize=(12, 6))
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['RSI14'], label='RSI (14)', color='blue', alpha=0.5)
        plt.plot(self.obj_dpp.monthly_hdfc_data.index, self.obj_dpp.monthly_hdfc_data['RSI8'], label='RSI (8)', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('RSI Chart')
        plt.legend()
        plt.grid(True)
        plt.show()

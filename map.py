import yfinance as yf
from datetime import datetime as dt, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearAxis, Range1d

def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return data

def create_stock_chart(df, predicted_dates, predicted_prices):
    # 創建 Bokeh 圖表
    df.index = pd.to_datetime(df.index)  
    p = figure(x_axis_type='datetime', width=1000, height=400, sizing_mode='stretch_width')
    p.title.text = "Stock CandleStick, Volume, 20-day MA, and Predicted Close Price"
    p.grid.grid_line_alpha = 0.35

    # 製作蠟燭圖
    dframe = df.copy()
    dframe["Status"] = ["Increase" if c > o else "Decrease" if c < o else "Equal" for c, o in zip(dframe.Close, dframe.Open)]
    dframe["Height"] = abs(dframe.Open - dframe.Close)
    dframe["Middle"] = (dframe.Open + dframe.Close) / 2
    hour12 = 12 * 60 * 60 * 1000

    # 使用綠色和紅色表示蠟燭圖
    p.segment(x0=dframe.index[dframe.Status == "Increase"], y0=dframe.Low[dframe.Status == "Increase"],
              x1=dframe.index[dframe.Status == "Increase"], y1=dframe.High[dframe.Status == "Increase"],
              color="green", line_width=3)
    p.segment(x0=dframe.index[dframe.Status == "Decrease"], y0=dframe.Low[dframe.Status == "Decrease"],
              x1=dframe.index[dframe.Status == "Decrease"], y1=dframe.High[dframe.Status == "Decrease"],
              color="red", line_width=3)

    source_inc = ColumnDataSource(dframe[dframe.Status == "Increase"])
    source_dec = ColumnDataSource(dframe[dframe.Status == "Decrease"])
    p.rect(x='index', y='Middle', width=hour12, height='Height', source=source_inc, fill_color="green", line_color="black")
    p.rect(x='index', y='Middle', width=hour12, height='Height', source=source_dec, fill_color="red", line_color="black")

    # 添加交易量
    p.extra_y_ranges = {"volume": Range1d(start=0, end=max(dframe['Volume']) * 1.2)}
    p.add_layout(LinearAxis(y_range_name="volume"), 'right')
    p.vbar(x='index', top='Volume', width=hour12, source=ColumnDataSource(dframe),
           y_range_name="volume", line_color="black", fill_color="purple", line_alpha=0.2)

    # 添加20-day MA
    dframe['MA20'] = dframe['Close'].rolling(window=20).mean()
    p.line(x=dframe.index, y=dframe['MA20'], line_color='orange', legend_label='20-day MA')

    # 添加預測的收盤價
    p.line(x=predicted_dates, y=predicted_prices, line_color='blue', legend_label='Predicted Close Price')

    return p


def linear_regression_predict(df, days=3):
    model = LinearRegression()

    # 使用日期的 Ordinal 表示進行訓練
    df['Date_Ordinal'] = df.index.to_numpy().astype(np.int64) // 10**9 // 86400  # 將納秒轉換為天

    # 切割數據集
    X_train = df[['Date_Ordinal']].values
    y_train = df['Close'].values
    model.fit(X_train, y_train)

    # 預測未來收盤價
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([(date - dt(1970, 1, 1)).total_seconds() // 86400 for date in future_dates]).reshape(-1, 1)

    future_predictions = model.predict(future_dates_ordinal)

    return future_dates, future_predictions


# 設定股票代碼和時間範圍
symbol = 'AAPL'
start_date = dt(2022, 1, 1)
end_date = dt(2022, 12, 31)

# 獲取股票數據
stock_data = fetch_stock_data(symbol, start_date, end_date)

# 繪製股票圖表
predicted_dates, predicted_prices = linear_regression_predict(stock_data)
plot = create_stock_chart(stock_data, predicted_dates, predicted_prices)

# 顯示 Bokeh 圖表
show(plot)
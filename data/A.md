
1. **Ashare**：
   Ashare 是一个轻量级的股票数据接口库，支持获取A股的实时行情数据，包括日线、分钟线、历史数据等。它使用新浪和腾讯的数据源，自动切换数据源，返回 pandas 的 DataFrame 格式，适合量化分析和自动化交易。它的用法非常简单，只需一行代码即可获取数据，例如：
   ```python
   from Ashare import *
   df = get_price('sh000001', frequency='1d', count=5)  # 获取上证指数最近5天的日线数据
   print(df)
   ```

2. **AKShare**：
   AKShare 是一个功能强大的开源财经数据接口库，支持获取A股、期货、外汇等数据。它的优势在于简单易用，且支持全球多个市场的数据。获取 A 股数据的代码如下：
   ```python
   import akshare as ak
   stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20210101", end_date="20211231")
   print(stock_data)
   ```

3. **Tushare**：
   Tushare 是另一个广泛使用的免费开源库，专注于中国市场，支持获取实时行情、历史行情和财务数据等。使用它可以方便地进行股票数据分析。例如：
   ```python
   import tushare as ts
   pro = ts.pro_api('your_api_token')
   df = pro.daily(ts_code='000001.SZ', start_date='20210101', end_date='20211231')
   print(df)
   ```

4. **Baostock**：
   Baostock 提供免费的A股历史数据和实时行情数据，适合做数据分析和回测。获取实时股票数据的例子如下：
   ```python
   import baostock as bs
   bs.login()
   rs = bs.query_history_k_data_plus("sh.600000", "date, code, open, high, low, close", start_date='2020-01-01', end_date='2020-12-31')
   print(rs.get_data())
   bs.logout()
   ```

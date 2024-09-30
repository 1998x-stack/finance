import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

class RandomWalkAnalysis:
    """随机游走模型分析类，用于检验股票价格是否符合随机游走模型。

    Attributes:
        tickers (List[str]): 股票代码列表。
        start_date (str): 数据起始日期。
        end_date (str): 数据结束日期。
        data (pd.DataFrame): 股票收盘价数据。
        returns (pd.DataFrame): 股票收益率数据。
        adf_results (Dict[str, Dict[str, float]]): ADF 检验结果。
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """初始化随机游走分析类。

        Args:
            tickers (List[str]): 股票代码列表。
            start_date (str): 数据起始日期，格式 'YYYY-MM-DD'。
            end_date (str): 数据结束日期，格式 'YYYY-MM-DD'。
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.adf_results = {}

    def fetch_data(self):
        """获取股票收盘价数据。"""
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        print("已成功获取股票数据。")

    def calculate_returns(self):
        """计算股票的对数收益率。"""
        self.returns = np.log(self.data / self.data.shift(1)).dropna()
        print("已计算对数收益率。")

    def adf_test(self):
        """对每只股票进行 ADF 单位根检验。"""
        for ticker in self.tickers:
            result = adfuller(self.data[ticker].dropna())
            self.adf_results[ticker] = {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Used Lag': result[2],
                'Number of Observations': result[3],
                'Critical Values': result[4],
                'Confidence Level': self._interpret_p_value(result[1])
            }
            print(f"{ticker} 的 ADF 检验完成。")

    def _interpret_p_value(self, p_value: float) -> str:
        """根据 p-value 给出置信度解释。

        Args:
            p_value (float): 检验的 p-value。

        Returns:
            str: 置信度解释。
        """
        if p_value < 0.01:
            return "99% 置信水平拒绝单位根假设"
        elif p_value < 0.05:
            return "95% 置信水平拒绝单位根假设"
        elif p_value < 0.10:
            return "90% 置信水平拒绝单位根假设"
        else:
            return "无法拒绝单位根假设"

    def plot_prices(self):
        """绘制股票价格走势。"""
        self.data.plot(figsize=(12, 6))
        plt.title('股票收盘价走势')
        plt.xlabel('日期')
        plt.ylabel('收盘价')
        plt.legend(self.tickers)
        plt.grid(True)
        plt.show()

    def plot_returns(self):
        """绘制股票收益率走势。"""
        self.returns.plot(figsize=(12, 6))
        plt.title('股票对数收益率走势')
        plt.xlabel('日期')
        plt.ylabel('对数收益率')
        plt.legend(self.tickers)
        plt.grid(True)
        plt.show()

    def plot_acf(self, lags: int = 20):
        """绘制股票收益率的自相关图。

        Args:
            lags (int, optional): 滞后阶数。默认为 20。
        """
        from statsmodels.graphics.tsaplots import plot_acf

        for ticker in self.tickers:
            plot_acf(self.returns[ticker], lags=lags, title=f'{ticker} 收益率自相关图')
            plt.show()

    def print_adf_results(self):
        """打印 ADF 检验结果。"""
        for ticker, result in self.adf_results.items():
            print(f"\n{ticker} 的 ADF 检验结果：")
            print(f"检验统计量：{result['Test Statistic']:.4f}")
            print(f"p-value：{result['p-value']:.4f}")
            print(f"使用的滞后阶数：{result['Used Lag']}")
            print(f"观测值数量：{result['Number of Observations']}")
            print(f"临界值：")
            for key, value in result['Critical Values'].items():
                print(f"  {key}: {value:.4f}")
            print(f"结论：{result['Confidence Level']}")

    def run(self):
        """执行完整的随机游走分析流程。"""
        self.fetch_data()
        self.calculate_returns()
        self.adf_test()
        self.print_adf_results()
        self.plot_prices()
        self.plot_returns()
        self.plot_acf()

if __name__ == "__main__":
    # 设置参数
    tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=1)

    rwa = RandomWalkAnalysis(
        tickers=tickers_list,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    rwa.run()
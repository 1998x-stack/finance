import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List, Dict

class SecurityMarketLine:
    """证券市场线类，用于计算并绘制 SML。

    Attributes:
        stock_tickers (List[str]): 股票代码列表。
        market_ticker (str): 市场指数代码。
        start_date (str): 数据起始日期。
        end_date (str): 数据结束日期。
        risk_free_rate (float): 无风险利率。
        stock_returns (pd.DataFrame): 股票收益率数据。
        market_returns (pd.Series): 市场指数收益率数据。
        betas (Dict[str, float]): 股票的 β 系数字典。
        expected_returns (Dict[str, float]): 股票的预期回报率字典。
    """

    def __init__(self, stock_tickers: List[str], market_ticker: str,
                 start_date: str, end_date: str, risk_free_rate: float):
        """初始化证券市场线类。

        Args:
            stock_tickers (List[str]): 股票代码列表。
            market_ticker (str): 市场指数代码。
            start_date (str): 数据起始日期，格式 'YYYY-MM-DD'。
            end_date (str): 数据结束日期，格式 'YYYY-MM-DD'。
            risk_free_rate (float): 无风险利率，年化利率（如 0.01 表示 1%）。
        """
        self.stock_tickers = stock_tickers
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.stock_returns = pd.DataFrame()
        self.market_returns = pd.Series()
        self.betas = {}
        self.expected_returns = {}
        self.market_premium = 0.0

    def fetch_data(self):
        """获取股票和市场指数数据，并计算收益率。"""
        # 获取股票数据
        stock_data = yf.download(self.stock_tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.stock_returns = stock_data.pct_change().dropna()

        # 获取市场指数数据
        market_data = yf.download(self.market_ticker, start=self.start_date, end=self.end_date)['Adj Close']
        self.market_returns = market_data.pct_change().dropna()

        # 对齐日期索引
        self.stock_returns = self.stock_returns.loc[self.market_returns.index]

    def calculate_betas(self):
        """计算每只股票的 β 系数。"""
        for ticker in self.stock_tickers:
            stock_ret = self.stock_returns[ticker]
            market_ret = self.market_returns

            # 添加常数项用于回归
            X = sm.add_constant(market_ret)
            Y = stock_ret

            # 线性回归
            model = sm.OLS(Y, X).fit()
            beta = model.params[1]
            self.betas[ticker] = beta

    def calculate_expected_returns(self):
        """计算每只股票的预期回报率。"""
        # 计算市场的平均收益率（年化）
        market_mean_return = self.market_returns.mean() * 252

        # 计算市场风险溢价
        self.market_premium = market_mean_return - self.risk_free_rate

        for ticker in self.stock_tickers:
            beta = self.betas[ticker]
            expected_return = self.risk_free_rate + beta * self.market_premium
            self.expected_returns[ticker] = expected_return

    def plot_sml(self):
        """绘制证券市场线和股票位置。"""
        # 绘制 SML 线
        betas = np.array(list(self.betas.values()))
        min_beta = betas.min() - 0.5
        max_beta = betas.max() + 0.5
        beta_values = np.linspace(min_beta, max_beta, 100)
        sml_values = self.risk_free_rate + beta_values * self.market_premium

        plt.figure(figsize=(10, 6))
        plt.plot(beta_values, sml_values, label='证券市场线 (SML)')

        # 绘制每只股票的位置
        for ticker in self.stock_tickers:
            beta = self.betas[ticker]
            expected_return = self.expected_returns[ticker]
            plt.plot(beta, expected_return, 'o', label=f'{ticker}')
            plt.text(beta, expected_return, f' {ticker}', fontsize=9)

        plt.xlabel('β 系数')
        plt.ylabel('预期年化回报率')
        plt.title('证券市场线 (SML)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_results(self):
        """打印每只股票的 β 系数和预期回报率。"""
        print("股票的 β 系数和预期年化回报率：")
        for ticker in self.stock_tickers:
            beta = self.betas[ticker]
            expected_return = self.expected_returns[ticker]
            print(f"{ticker}: β = {beta:.4f}, 预期年化回报率 = {expected_return:.2%}")

    def run(self):
        """执行完整的 SML 分析流程。"""
        self.fetch_data()
        self.calculate_betas()
        self.calculate_expected_returns()
        self.print_results()
        self.plot_sml()

if __name__ == "__main__":
    # 设置参数
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    market_ticker = '^GSPC'  # 标普500指数
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    risk_free_rate = 0.01  # 无风险利率

    sml = SecurityMarketLine(
        stock_tickers=stock_tickers,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate
    )
    sml.run()
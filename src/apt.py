import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from typing import List, Dict
import warnings

from config.config import FONT_PATH, BASIC_IMAGE_DIR

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'

warnings.filterwarnings('ignore')

class ArbitragePricingTheoryModel:
    """套利定价理论模型类，用于估计资产的预期回报率。

    Attributes:
        asset_ticker (str): 目标资产的股票代码。
        factor_tickers (List[str]): 因子资产的股票代码列表。
        start_date (str): 起始日期。
        end_date (str): 结束日期。
        asset_returns (pd.Series): 目标资产的收益率序列。
        factor_returns (pd.DataFrame): 因子资产的收益率数据框。
        factor_loadings (Dict[str, float]): 因子载荷（敏感度）。
    """

    def __init__(self, asset_ticker: str, factor_tickers: List[str], start_date: str, end_date: str):
        """初始化套利定价理论模型类。

        Args:
            asset_ticker (str): 目标资产的股票代码。
            factor_tickers (List[str]): 因子资产的股票代码列表。
            start_date (str): 起始日期，格式为'YYYY-MM-DD'。
            end_date (str): 结束日期，格式为'YYYY-MM-DD'。
        """
        self.asset_ticker = asset_ticker
        self.factor_tickers = factor_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.asset_returns = pd.Series()
        self.factor_returns = pd.DataFrame()
        self.factor_loadings = {}

    def fetch_data(self):
        """获取目标资产和因子资产的收盘价数据。"""
        # 获取目标资产数据
        asset_data = yf.download(self.asset_ticker, start=self.start_date, end=self.end_date)
        # 获取因子资产数据
        factor_data = yf.download(self.factor_tickers, start=self.start_date, end=self.end_date)

        # 计算每日收益率
        self.asset_returns = asset_data['Adj Close'].pct_change().dropna()
        self.factor_returns = factor_data['Adj Close'].pct_change().dropna()

        # 对齐日期索引
        combined_data = pd.concat([self.asset_returns, self.factor_returns], axis=1, join='inner')
        self.asset_returns = combined_data.iloc[:, 0]
        self.factor_returns = combined_data.iloc[:, 1:]

    def estimate_factor_loadings(self):
        """估计资产对各个风险因素的敏感度（因子载荷）。"""
        # 准备回归模型的数据
        X = self.factor_returns.values
        Y = self.asset_returns.values

        # 线性回归模型
        reg = LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_

        # 保存因子载荷
        self.factor_loadings = dict(zip(self.factor_tickers, coefficients))

        print("估计的因子载荷（敏感度）：")
        for factor, loading in self.factor_loadings.items():
            print(f"{factor}: {loading:.4f}")

    def calculate_expected_return(self, risk_free_rate: float = 0.02):
        """计算资产的预期回报率。

        Args:
            risk_free_rate (float, optional): 无风险利率，默认为0.02（2%）。
        """
        # 计算各个因子的风险溢价（年化）
        factor_premiums = self.factor_returns.mean() * 252 - risk_free_rate

        # 计算预期回报率
        expected_return = risk_free_rate
        for factor in self.factor_tickers:
            expected_return += self.factor_loadings[factor] * factor_premiums[factor]

        print(f"根据 APT 计算的预期年化回报率为: {expected_return:.2%}")

        self.expected_return = expected_return

    def plot_factor_loadings(self):
        """绘制因子载荷的柱状图。"""
        factors = list(self.factor_loadings.keys())
        loadings = list(self.factor_loadings.values())

        plt.figure(figsize=(10, 6))
        plt.bar(factors, loadings, color='skyblue')
        plt.xlabel('风险因素')
        plt.ylabel('因子载荷（敏感度）')
        plt.title(f'{self.asset_ticker} 的因子载荷')
        plt.grid(True)
        plt.savefig(
            os.path.join(BASIC_IMAGE_DIR, f'{self.asset_ticker}_factor_loadings.png')
        )

    def run(self, risk_free_rate: float = 0.02):
        """执行APT分析的完整流程。

        Args:
            risk_free_rate (float, optional): 无风险利率，默认为0.02（2%）。
        """
        self.fetch_data()
        self.estimate_factor_loadings()
        self.calculate_expected_return(risk_free_rate)
        self.plot_factor_loadings()

if __name__ == "__main__":
    # 设置参数
    asset_ticker = 'AAPL'          # 目标资产代码
    # 选择一些行业指数或大型公司的股票作为风险因素
    factor_tickers = ['^GSPC', '^IXIC', '^DJI']  # 标普500、纳斯达克综合指数、道琼斯指数
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(months=3)).strftime('%Y-%m-%d')

    apt_model = ArbitragePricingTheoryModel(asset_ticker, factor_tickers, start_date, end_date)
    apt_model.run(risk_free_rate=0.02)
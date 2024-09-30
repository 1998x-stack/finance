import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import warnings

warnings.filterwarnings('ignore')

class MMTheoremAnalysis:
    """MM定理分析类，用于研究资本结构与公司价值的关系。

    Attributes:
        tickers (List[str]): 公司股票代码列表。
        financial_data (pd.DataFrame): 财务数据，包括债务和权益。
        market_data (pd.DataFrame): 市场数据，包括市值和企业价值。
        data (pd.DataFrame): 合并后的数据，用于分析。
    """

    def __init__(self, tickers: List[str]):
        """初始化 MM 定理分析类。

        Args:
            tickers (List[str]): 公司股票代码列表。
        """
        self.tickers = tickers
        self.financial_data = self._get_financial_data()
        self.market_data = self._get_market_data()
        self.data = pd.DataFrame()

    def _get_financial_data(self) -> pd.DataFrame:
        """获取公司的财务数据。

        Returns:
            pd.DataFrame: 包含债务、权益和债务权益比的数据。
        """
        financials = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            # 获取最近的资产负债表
            balance_sheet = stock.quarterly_balance_sheet
            if balance_sheet.empty or 'Total Debt' not in balance_sheet.index or 'Total Stockholder Equity' not in balance_sheet.index:
                continue
            total_debt = balance_sheet.loc['Total Debt'][0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'][0]
            if total_equity == 0:
                debt_to_equity = np.nan
            else:
                debt_to_equity = total_debt / total_equity
            financials[ticker] = {
                'Total Debt': total_debt,
                'Total Equity': total_equity,
                'Debt to Equity Ratio': debt_to_equity
            }
        financial_data = pd.DataFrame.from_dict(financials, orient='index')
        return financial_data

    def _get_market_data(self) -> pd.DataFrame:
        """获取公司的市场数据。

        Returns:
            pd.DataFrame: 包含市值和企业价值的数据。
        """
        market_data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', np.nan)
            enterprise_value = info.get('enterpriseValue', np.nan)
            market_data[ticker] = {
                'Market Cap': market_cap,
                'Enterprise Value': enterprise_value
            }
        market_data = pd.DataFrame.from_dict(market_data, orient='index')
        return market_data

    def analyze(self):
        """合并财务数据和市场数据，准备进行分析。"""
        self.data = pd.concat([self.financial_data, self.market_data], axis=1)
        # 删除缺失数据的行
        self.data.dropna(inplace=True)

    def plot_debt_to_equity_vs_value(self):
        """绘制债务权益比与公司价值的关系图。"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Debt to Equity Ratio'], self.data['Market Cap'] / 1e9, color='blue', label='市值')
        plt.scatter(self.data['Debt to Equity Ratio'], self.data['Enterprise Value'] / 1e9, color='green', label='企业价值')
        plt.xlabel('债务权益比')
        plt.ylabel('公司价值（十亿美元）')
        plt.title('债务权益比与公司价值的关系')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_data(self):
        """打印分析使用的数据。"""
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print("公司财务和市场数据：")
        print(self.data)

if __name__ == "__main__":
    # 选定科技行业中的一些公司
    tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'IBM', 'ORCL']
    mm_analysis = MMTheoremAnalysis(tickers_list)

    # 进行数据分析
    mm_analysis.analyze()
    mm_analysis.print_data()

    # 绘制可视化图表
    mm_analysis.plot_debt_to_equity_vs_value()

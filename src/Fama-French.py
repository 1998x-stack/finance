import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import requests
import statsmodels.api as sm
from zipfile import ZipFile
import os

from util.log_utils import logger
from config.config import FONT_PATH, BASIC_IMAGE_DIR, BASIC_DATA_DIR

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'

class FamaFrenchModel:
    """Fama-French 三因素模型类，用于估计股票的预期回报率。

    Attributes:
        stock_ticker (str): 股票代码。
        start_date (str): 起始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
        stock_returns (pd.Series): 股票收益率序列。
        factors (pd.DataFrame): 三因素数据。
        results (Optional[sm.OLS]): 回归结果。
    """

    def __init__(self, stock_ticker: str, start_date: str, end_date: str):
        """初始化 Fama-French 三因素模型。

        Args:
            stock_ticker (str): 股票代码。
            start_date (str): 起始日期。
            end_date (str): 结束日期。
        """
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_returns = pd.Series()
        self.factors = pd.DataFrame()
        self.results = None

    def fetch_stock_data(self):
        """获取股票数据并计算收益率。"""
        stock_data = yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)
        self.stock_returns = stock_data['Adj Close'].pct_change().dropna() * 100  # 转换为百分比
        logger.log_info(f"已获取 {self.stock_ticker} 的股票数据。")

    def fetch_fama_french_factors(self):
        """从 Kenneth French 网站获取 Fama-French 三因素数据，保存文件并从保存的位置读取。"""
        flag = False
        if os.path.exists(os.path.join(BASIC_DATA_DIR, 'F-F_Research_Data_Factors_daily.CSV')):
            flag = True
        else:
            url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
            response = requests.get(url)
            flag = response.ok
        if flag:
            # 保存 ZIP 文件
            zip_path = os.path.join(BASIC_DATA_DIR, 'fama_french_factors.zip')
            # 解压 ZIP 文件并保存 CSV
            csv_path = os.path.join(BASIC_DATA_DIR, 'F-F_Research_Data_Factors_daily.CSV')
            if not os.path.exists(zip_path):
                with open(zip_path, 'wb') as zip_file:
                    zip_file.write(response.content)
                logger.log_info(f"已保存 ZIP 文件到 {zip_path}")
            if not os.path.exists(csv_path):
                with ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extract(csv_path)
                logger.log_info(f"已解压并保存 CSV 文件到 {csv_path}")

            # 从保存的 CSV 文件读取数据
            data = pd.read_csv(csv_path, skiprows=3)
            data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            data = data[:-1]  # 删除最后的无关行
            data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
            data.set_index('Date', inplace=True)
            self.factors = data.apply(pd.to_numeric)
            logger.log_info("已从保存的文件读取 Fama-French 三因素数据。")
        else:
            logger.log_info("无法从网站获取 Fama-French 数据。")

    def prepare_data(self):
        """准备数据，合并股票收益率和三因素数据。"""
        self.fetch_stock_data()
        self.fetch_fama_french_factors()

        # 合并数据
        self.data = pd.concat([self.stock_returns, self.factors], axis=1, join='inner')
        self.data.dropna(inplace=True)
        self.data.rename(columns={'Adj Close': 'Stock_Return'}, inplace=True)
        logger.log_info("数据已合并完成。")

    def run_regression(self):
        """运行多元线性回归，估计模型参数。"""
        Y = self.data['Stock_Return'] - self.data['RF']  # 超额收益率
        X = self.data[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        self.results = model.fit()
        logger.log_info("回归分析已完成。")
        logger.log_info(self.results.summary())

    def plot_actual_vs_fitted(self):
        """绘制实际收益率与模型拟合收益率的对比图。"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data['Stock_Return'] - self.data['RF'], label='实际超额收益率')
        plt.plot(self.data.index, self.results.fittedvalues, label='模型预测收益率')
        plt.xlabel('日期')
        plt.ylabel('超额收益率 (%)')
        plt.title(f'{self.stock_ticker} 实际与预测超额收益率对比')
        plt.legend()
        plt.savefig(os.path.join(BASIC_IMAGE_DIR, f'{self.stock_ticker}_actual_vs_fitted.png'))
        plt.close()

    def plot_factor_loadings(self):
        """绘制因子载荷的柱状图。"""
        params = self.results.params[1:]  # 排除常数项
        params.plot(kind='bar', figsize=(8, 6))
        plt.title(f'{self.stock_ticker} 的因子载荷')
        plt.ylabel('因子载荷系数')
        plt.savefig(os.path.join(BASIC_IMAGE_DIR, f'{self.stock_ticker}_factor_loadings.png'))
        plt.close()

    def run(self):
        """执行完整的模型分析流程。"""
        self.prepare_data()
        self.run_regression()
        self.plot_actual_vs_fitted()
        self.plot_factor_loadings()

if __name__ == "__main__":
    # ���置参数
    stock_ticker = 'AAPL'  # 股票代码
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=3)

    fama_french_model = FamaFrenchModel(
        stock_ticker=stock_ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    fama_french_model.run()
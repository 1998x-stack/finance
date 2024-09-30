# coding=utf-8
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

# 字体路径，用于支持中文显示
FONT_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..',
    'config/SimHei.ttf'
)


BASIC_IMAGE_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..',
    'figures',
)


BASIC_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..',
    'data',
)


# 定义要分析的股票代码
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
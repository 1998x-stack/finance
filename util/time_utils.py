import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import time
from functools import wraps
from threading import Lock
from datetime import datetime
from typing import Callable, Any

from util.log_utils import Log

logger = Log("time_utils")


class FuncUsage:
    """
    记录函数使用的统计信息，包括调用次数、执行时间等。
    """
    def __init__(self):
        self.usage_stats = {}  # 用于记录函数使用信息的字典
        self.lock = Lock()  # 用于线程安全

    def update_usage(self, func_key: str, execution_time: float) -> None:
        """
        更新函数的使用统计信息。
        :param func_key: 函数标识符
        :param execution_time: 函数执行时间（毫秒）
        """
        with self.lock:
            usage = self.usage_stats.get(
                func_key,
                {
                    'count': 0,
                    'last_used': None,
                    'total_time': 0,
                    'mean_time': 0,
                    'max_time': 0,
                }
            )
            usage['count'] += 1
            usage['last_used'] = str(datetime.now())
            usage['total_time'] += execution_time
            usage['mean_time'] = usage['total_time'] / usage['count']
            usage['max_time'] = max(usage['max_time'], execution_time)
            self.usage_stats[func_key] = usage


func_usage_tracker = FuncUsage()


def truncate_args(args, kwargs, limit=30):
    """
    截断参数和关键字参数的字符串表示形式。
    :param args: 函数参数
    :param kwargs: 关键字参数
    :param limit: 截断长度
    :return: 截断后的参数和关键字参数表示形式
    """
    truncated_args = [str(arg)[:limit] for arg in args]
    truncated_kwargs = {k: str(v)[:limit] for k, v in kwargs.items()}
    return truncated_args, truncated_kwargs


def calculate_execution_time(func_id: str = 'default') -> Callable:
    """
    计算函数执行时间的装饰器。
    :param func_id: 函数的唯一标识符
    :return: 包装后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_key = f'【{func_id}】{func.__module__}.{func.__qualname__}: '  # 使用__qualname__处理类和函数
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # 捕获异常并记录日志
                if logger:
                    logger.log_info(f"[=== Exception in {func_key}: {str(e)} ===]")
                raise  # 重新抛出异常
            finally:
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # 将秒转换为毫秒

                # 记录执行时间和更新统计信息
                if logger:
                    truncated_args, truncated_kwargs = truncate_args(args, kwargs)
                    logger.log_info(f"[=== Execution time of {func_key}: {round(execution_time, 4)} ms, "
                                    f"args: {truncated_args}, kwargs: {truncated_kwargs} ===]")
                func_usage_tracker.update_usage(func_key, execution_time)

                # 打印使用统计信息
                if logger:
                    usage = func_usage_tracker.usage_stats[func_key]
                    logger.log_info(f"[=== {func_key} - MEAN TIME: {round(usage['mean_time'], 4)} ms, "
                                    f"MAX TIME: {round(usage['max_time'], 4)} ms, "
                                    f"TOTAL TIME: {round(usage['total_time'], 4)} ms, "
                                    f"USAGE COUNTS: {usage['count']} ===]")

            return result
        return wrapper
    return decorator
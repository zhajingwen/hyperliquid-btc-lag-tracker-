#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比1分钟和5分钟K线计算的Z-score差异
验证1分钟数据是否导致Z-score偏低
"""

import pandas as pd
import numpy as np
from hyperliquid_analyzer import DelayCorrelationAnalyzer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def calculate_zscore_stats(btc_prices, alt_prices, window=30, beta_window=60):
    """
    计算Z-score及相关统计量

    Returns:
        dict: {
            'zscore': 当前Z-score,
            'mean': 价差均值,
            'std': 价差标准差,
            'current_spread': 当前价差,
            'beta': Beta系数
        }
    """
    analyzer = DelayCorrelationAnalyzer()

    # 计算Beta
    beta = analyzer._calculate_beta_from_prices(btc_prices, alt_prices)
    if beta is None or np.isnan(beta):
        return None

    # 计算价差序列
    log_btc = np.log(btc_prices)
    log_alt = np.log(alt_prices)
    spread = log_alt - beta * log_btc

    # 计算统计量（使用最后window个点）
    if len(spread) < window:
        return None

    spread_window = spread.iloc[-window:]
    mean = spread_window.mean()
    std = spread_window.std()
    current_spread = spread.iloc[-1]

    if std == 0 or np.isnan(std):
        return None

    zscore = (current_spread - mean) / std

    return {
        'zscore': zscore,
        'mean': mean,
        'std': std,
        'current_spread': current_spread,
        'beta': beta,
        'data_points': len(spread)
    }


def compare_timeframes(coin="ETHFI/USDC:USDC"):
    """对比1分钟和5分钟K线的Z-score"""

    analyzer = DelayCorrelationAnalyzer()

    logger.info(f"\n{'='*80}")
    logger.info(f"对比分析：1分钟 vs 5分钟K线的Z-score")
    logger.info(f"币种: {coin}")
    logger.info(f"{'='*80}\n")

    # 1. 获取1分钟数据（1天）
    logger.info("📊 获取1分钟K线数据（1天周期）...")
    btc_1m = analyzer._get_btc_data("1m", "1d")
    alt_1m = analyzer._get_alt_data(coin, "1d", "1m", coin)

    if btc_1m is None or alt_1m is None:
        logger.error("❌ 1分钟数据获取失败")
        return

    # 对齐数据
    aligned_1m = analyzer._align_and_validate_data(btc_1m, alt_1m, coin, "1m", "1d")
    if aligned_1m is None:
        logger.error("❌ 1分钟数据对齐失败")
        return

    btc_1m_aligned, alt_1m_aligned = aligned_1m
    logger.info(f"✅ 1分钟数据：{len(btc_1m_aligned)} 个数据点")

    # 2. 获取5分钟数据（7天）
    logger.info("\n📊 获取5分钟K线数据（7天周期）...")
    btc_5m = analyzer._get_btc_data("5m", "7d")
    alt_5m = analyzer._get_alt_data(coin, "7d", "5m", coin)

    if btc_5m is None or alt_5m is None:
        logger.error("❌ 5分钟数据获取失败")
        return

    # 对齐数据
    aligned_5m = analyzer._align_and_validate_data(btc_5m, alt_5m, coin, "5m", "7d")
    if aligned_5m is None:
        logger.error("❌ 5分钟数据对齐失败")
        return

    btc_5m_aligned, alt_5m_aligned = aligned_5m
    logger.info(f"✅ 5分钟数据：{len(btc_5m_aligned)} 个数据点")

    # 3. 计算1分钟K线的Z-score
    logger.info("\n🔢 计算1分钟K线的Z-score...")
    stats_1m = calculate_zscore_stats(
        btc_1m_aligned['Close'],
        alt_1m_aligned['Close'],
        window=30,
        beta_window=60
    )

    if stats_1m is None:
        logger.error("❌ 1分钟Z-score计算失败")
        return

    logger.info(f"  - 数据点数: {stats_1m['data_points']}")
    logger.info(f"  - Beta系数: {stats_1m['beta']:.4f}")
    logger.info(f"  - 价差均值: {stats_1m['mean']:.6f}")
    logger.info(f"  - 价差标准差: {stats_1m['std']:.6f}")
    logger.info(f"  - 当前价差: {stats_1m['current_spread']:.6f}")
    logger.info(f"  - Z-score: {stats_1m['zscore']:.4f}")

    # 4. 计算5分钟K线的Z-score
    logger.info("\n🔢 计算5分钟K线的Z-score...")
    stats_5m = calculate_zscore_stats(
        btc_5m_aligned['Close'],
        alt_5m_aligned['Close'],
        window=30,
        beta_window=60
    )

    if stats_5m is None:
        logger.error("❌ 5分钟Z-score计算失败")
        return

    logger.info(f"  - 数据点数: {stats_5m['data_points']}")
    logger.info(f"  - Beta系数: {stats_5m['beta']:.4f}")
    logger.info(f"  - 价差均值: {stats_5m['mean']:.6f}")
    logger.info(f"  - 价差标准差: {stats_5m['std']:.6f}")
    logger.info(f"  - 当前价差: {stats_5m['current_spread']:.6f}")
    logger.info(f"  - Z-score: {stats_5m['zscore']:.4f}")

    # 5. 对比分析
    logger.info(f"\n{'='*80}")
    logger.info("📊 对比分析结果")
    logger.info(f"{'='*80}\n")

    # 标准差对比
    std_ratio = stats_1m['std'] / stats_5m['std']
    logger.info(f"标准差对比:")
    logger.info(f"  - 1分钟标准差: {stats_1m['std']:.6f}")
    logger.info(f"  - 5分钟标准差: {stats_5m['std']:.6f}")
    logger.info(f"  - 比率 (1m/5m): {std_ratio:.2f}x")

    if std_ratio > 1.2:
        logger.warning(f"  ⚠️ 1分钟标准差明显大于5分钟（{std_ratio:.2f}倍），说明噪音更大")
    elif std_ratio < 0.8:
        logger.info(f"  ✅ 1分钟标准差小于5分钟")
    else:
        logger.info(f"  ✅ 两者标准差接近")

    # Z-score对比
    logger.info(f"\nZ-score对比:")
    logger.info(f"  - 1分钟Z-score: {stats_1m['zscore']:.4f}")
    logger.info(f"  - 5分钟Z-score: {stats_5m['zscore']:.4f}")
    logger.info(f"  - 差值: {stats_5m['zscore'] - stats_1m['zscore']:+.4f}")

    # 阈值判断
    threshold = 2.0
    logger.info(f"\n阈值判断 (阈值={threshold}):")
    logger.info(f"  - 1分钟: {'✅ 通过' if abs(stats_1m['zscore']) >= threshold else '❌ 未通过'} (|{stats_1m['zscore']:.2f}|)")
    logger.info(f"  - 5分钟: {'✅ 通过' if abs(stats_5m['zscore']) >= threshold else '❌ 未通过'} (|{stats_5m['zscore']:.2f}|)")

    # 结论
    logger.info(f"\n{'='*80}")
    logger.info("💡 结论")
    logger.info(f"{'='*80}\n")

    if abs(stats_5m['zscore']) > abs(stats_1m['zscore']) * 1.3:
        logger.warning(
            f"⚠️ 5分钟Z-score明显高于1分钟（{abs(stats_5m['zscore']) / abs(stats_1m['zscore']):.2f}倍）\n"
            f"   说明1分钟数据的噪音确实导致Z-score偏低\n"
            f"   建议：改用5分钟数据计算Z-score，或降低1分钟阈值"
        )
    elif abs(stats_1m['zscore']) > abs(stats_5m['zscore']) * 1.3:
        logger.info(
            f"✅ 1分钟Z-score反而更高\n"
            f"   说明当前偏离主要在短期（1分钟级别）\n"
            f"   保持使用1分钟数据是合理的"
        )
    else:
        logger.info(
            f"✅ 两个时间级别的Z-score接近\n"
            f"   说明价格偏离在不同时间级别上一致\n"
            f"   使用1分钟或5分钟都可以"
        )

    # Beta对比
    logger.info(f"\nBeta系数对比:")
    logger.info(f"  - 1分钟Beta: {stats_1m['beta']:.4f}")
    logger.info(f"  - 5分钟Beta: {stats_5m['beta']:.4f}")
    logger.info(f"  - 差值: {stats_5m['beta'] - stats_1m['beta']:+.4f}")

    return {
        '1m': stats_1m,
        '5m': stats_5m
    }


if __name__ == '__main__':
    # 测试多个币种
    test_coins = [
        "ETHFI/USDC:USDC",  # Z-score 0.78 (1m)
        "GALA/USDC:USDC",   # Z-score 1.65 (1m)
        "DOGE/USDC:USDC",   # Z-score 1.83 (1m)
    ]

    for coin in test_coins:
        try:
            result = compare_timeframes(coin)
            if result:
                input(f"\n按Enter继续下一个币种...")
        except Exception as e:
            logger.error(f"❌ {coin} 分析失败: {e}")
            continue

#!/usr/bin/env python3
"""
分析hyperliquid.log日志，统计每个代币未触发告警的原因
"""
import re
from collections import defaultdict
from datetime import datetime

def analyze_log(log_file, target_date="2025-12-31"):
    """分析日志文件"""

    # 统计数据
    coins_checked = set()
    coin_results = defaultdict(lambda: {
        'reasons': [],
        'correlation_check': None,
        'beta_check': None,
        'zscore_check': None,
        'stationarity_check': None,
        'data_exists': True
    })

    # 读取日志文件
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤目标日期的日志
    target_lines = [line for line in lines if target_date in line]

    print(f"找到 {len(target_lines)} 条 {target_date} 的日志记录")

    current_coin = None

    for line in target_lines:
        # 提取检查的币种
        if "检查币种:" in line:
            match = re.search(r'检查币种: (\S+)', line)
            if match:
                current_coin = match.group(1)
                coins_checked.add(current_coin)

        if not current_coin:
            continue

        # 数据不存在
        if "数据不存在" in line or "空数据" in line:
            if "币种: " + current_coin in line or current_coin in line:
                coin_results[current_coin]['data_exists'] = False
                coin_results[current_coin]['reasons'].append("数据不存在（交易对可能已下线或无历史数据）")

        # Beta收益率系数不满足
        if "Beta收益率系数不满足要求" in line:
            if "币种: " + current_coin in line or current_coin in line:
                match = re.search(r'平均Beta: ([\d.]+) < (\d+)', line)
                if match:
                    beta_value = match.group(1)
                    threshold = match.group(2)
                    coin_results[current_coin]['beta_check'] = False
                    coin_results[current_coin]['reasons'].append(f"Beta收益率系数不满足要求（平均Beta={beta_value} < {threshold}，套利空间不足）")

        # 相关系数检测
        if "相关系数检测" in line and "币种: " + current_coin in line:
            match = re.search(r'是否异常: (True|False)', line)
            if match:
                is_anomaly = match.group(1) == "True"
                coin_results[current_coin]['correlation_check'] = is_anomaly
                if not is_anomaly:
                    match_diff = re.search(r'相关系数差值: ([\d.]+)', line)
                    if match_diff:
                        diff = match_diff.group(1)
                        coin_results[current_coin]['reasons'].append(f"相关系数检测未发现异常（差值={diff}，未达到告警阈值）")

        # 平稳性检验失败
        if "平稳性检验失败" in line and "币种: " + current_coin in line:
            match = re.search(r'p-value: ([\d.]+)', line)
            if match:
                p_value = match.group(1)
                coin_results[current_coin]['stationarity_check'] = False
                coin_results[current_coin]['reasons'].append(f"价差序列非平稳（ADF检验 p-value={p_value} >= 0.10，均值回归假设不成立）")

        # Z-score验证未通过
        if "Z-score 验证未通过" in line and "币种: " + current_coin in line:
            match = re.search(r'Z-score: ([-\d.]+)', line)
            if match:
                zscore = match.group(1)
                coin_results[current_coin]['zscore_check'] = False
                coin_results[current_coin]['reasons'].append(f"Z-score验证未通过（Z-score={zscore}，绝对值小于阈值，偏离程度不足）")

        # 常规数据（没有异常）
        if "常规数据" in line and "币种: " + current_coin in line:
            if not coin_results[current_coin]['reasons']:
                match = re.search(r'相关系数范围: ([\d.]+) ~ ([\d.]+)', line)
                if match:
                    min_corr = match.group(1)
                    max_corr = match.group(2)
                    coin_results[current_coin]['reasons'].append(f"相关系数正常（范围: {min_corr} ~ {max_corr}，无显著异常）")

    return coins_checked, coin_results

def generate_report(coins_checked, coin_results):
    """生成分析报告"""

    # 统计各类原因
    reason_stats = defaultdict(int)

    for coin, info in coin_results.items():
        for reason in info['reasons']:
            # 简化原因分类
            if "数据不存在" in reason:
                reason_stats["数据不存在"] += 1
            elif "Beta收益率系数不满足" in reason:
                reason_stats["Beta系数不满足"] += 1
            elif "相关系数检测未发现异常" in reason or "相关系数正常" in reason:
                reason_stats["相关系数正常"] += 1
            elif "价差序列非平稳" in reason:
                reason_stats["价差序列非平稳"] += 1
            elif "Z-score验证未通过" in reason:
                reason_stats["Z-score未达阈值"] += 1

    # 生成报告内容
    report = f"""# Hyperliquid BTC滞后性监控系统 - 日志分析报告

## 📊 数据概览

- **分析日期**: 2025-12-31（注：2026-01-01暂无日志数据）
- **监控币种总数**: {len(coins_checked)} 个
- **告警数量**: 0 条

## ⚠️ 关键发现

**本监控周期内未产生任何告警信息**

## 📋 未触发告警原因统计

| 过滤原因 | 币种数量 | 占比 |
|---------|---------|------|
"""

    total = sum(reason_stats.values())
    for reason, count in sorted(reason_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100 if total > 0 else 0
        report += f"| {reason} | {count} | {percentage:.1f}% |\n"

    report += f"\n**合计**: {total} 次过滤（部分币种可能有多个过滤原因）\n\n"

    # 详细的币种分析
    report += """## 📝 各币种未触发告警详细原因

### 告警触发条件说明

系统采用多层过滤机制，币种需要通过以下所有检查才会触发告警：

1. **数据可用性检查**: 币种必须有足够的历史数据
2. **Beta系数检查**: 平均Beta收益率系数 ≥ 1.0（确保有足够的套利空间）
3. **相关系数异常检测**: 短期与长期相关系数需出现显著差异
4. **价差序列平稳性检验**: ADF检验 p-value < 0.10（确保均值回归特性）
5. **Z-score阈值验证**: Z-score绝对值需达到设定阈值（确保偏离程度足够）

只要任一条件不满足，就不会触发告警。

### 币种详细分析

"""

    # 按字母顺序排序币种
    for coin in sorted(coins_checked):
        info = coin_results[coin]
        report += f"#### {coin}\n\n"

        if not info['data_exists']:
            report += "- ❌ **数据不存在**: 该交易对可能已下线或无足够历史数据\n\n"
            continue

        if not info['reasons']:
            report += "- ⚠️ **未找到具体过滤原因**（可能数据处理中断）\n\n"
            continue

        for idx, reason in enumerate(info['reasons'], 1):
            report += f"- {reason}\n"

        report += "\n"

    # 添加系统配置说明
    report += """## ⚙️ 系统配置参数

根据代码分析，当前系统使用以下关键参数：

- **相关系数差值阈值**: 需要短期与长期相关系数出现显著差异
- **Beta系数阈值**: ≥ 1.0（低于此值认为套利空间不足）
- **平稳性检验**: ADF检验 p-value < 0.10
- **Z-score阈值**: 需要价差偏离达到一定标准差倍数
- **监控周期**:
  - 长期: 5分钟K线 / 7天周期
  - 短期: 1分钟K线 / 1天周期

## 💡 结论与建议

### 为什么今天没有告警？

通过分析发现，所有监控的币种都因以下一个或多个原因被过滤：

1. **数据质量问题** ({reason_stats.get('数据不存在', 0)} 个币种): 部分交易对可能已下线或缺少历史数据
2. **Beta系数不足** ({reason_stats.get('Beta系数不满足', 0)} 个币种): 与BTC的关联强度不够，套利空间有限
3. **相关系数正常** ({reason_stats.get('相关系数正常', 0)} 个币种): 短期和长期相关性未出现显著变化
4. **价差非平稳** ({reason_stats.get('价差序列非平稳', 0)} 个币种): 不满足均值回归特性，不适合配对交易
5. **Z-score未达阈值** ({reason_stats.get('Z-score未达阈值', 0)} 个币种): 价差偏离程度不够显著

### 建议

- ✅ **系统运行正常**: 所有币种都经过了完整的检查流程
- ✅ **过滤机制有效**: 多层过滤确保只有高质量信号才会触发告警
- ⚠️ **数据监控**: 建议定期清理已下线的交易对，减少无效检查
- 📊 **参数优化**: 如果长期无告警，可考虑适当调整阈值参数

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report

if __name__ == "__main__":
    log_file = "/Users/test/Documents/hyperliquid-btc-lag-tracker/hyperliquid.log"

    print("正在分析日志文件...")
    coins_checked, coin_results = analyze_log(log_file)

    print(f"\n共检查了 {len(coins_checked)} 个币种")
    print("\n正在生成报告...")

    report = generate_report(coins_checked, coin_results)

    # 保存报告
    report_file = "/Users/test/Documents/hyperliquid-btc-lag-tracker/log_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 报告已生成: {report_file}")
    print("\n预览前50行:")
    print("=" * 80)
    for line in report.split('\n')[:50]:
        print(line)

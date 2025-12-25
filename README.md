# Hyperliquid BTC 滞后性追踪器

一个用于分析 Hyperliquid 交易所中山寨币与 BTC 相关性的量化分析工具,通过识别短期低相关但长期高相关的异常币种,发现潜在的时间差套利机会。

## 项目背景与目标

### 核心需求

本项目旨在研究 **Hyperliquid 交易所**上具有以下特征的山寨币：

1. **强跟随 BTC**：长期（如7天）与 BTC 价格走势高度相关
2. **存在滞后性**：短期（如1天）存在明显的价格发现延迟
3. **波动幅度大**：相对 BTC 有更大的价格波动（Beta系数 > 1.0）

### 项目定位

- **当前阶段**：研究与探索阶段，专注于识别和验证异常币种特征
- **长期目标**：为后期构建**全自动化套利系统**提供理论基础和早期数据支撑
- **核心价值**：通过量化分析发现时间差套利机会，自动化告警潜在目标币种

### 应用场景

- **研究分析**：识别具有套利潜力的币种，建立候选池
- **风险评估**：通过 Beta 系数评估波动风险，过滤高风险标的
- **策略验证**：为后续自动化交易策略提供历史数据和回测依据
- **实时监控**：自动化检测和告警，减少人工筛选成本

## 项目简介

基于原项目 [related_corrcoef_abnormal_alert](https://github.com/zhajingwen/related_corrcoef_abnormal_alert) 改进开发。

本项目通过计算不同时间周期和时间延迟下山寨币与 BTC 的皮尔逊相关系数,自动识别存在时间差套利空间的异常币种。核心原理是寻找**短期滞后但长期跟随 BTC 走势**的币种,这类币种可能存在价格发现延迟,从而产生套利机会。

### 核心功能

- **多时间周期分析**: 支持 5分钟/7天 和 1分钟/1天 两种K线组合
- **时间延迟优化**: 自动寻找最优延迟 τ*,识别滞后性套利机会
- **Beta 系数计算**: 衡量山寨币相对 BTC 的波动幅度,评估风险
- **异常值处理**: 采用 Winsorization 方法提高统计分析的稳健性
- **实时告警通知**: 通过飞书机器人推送异常币种检测结果
- **定时调度**: 支持按时间或周期自动运行分析任务

## 技术原理

### 相关系数分析

项目通过计算不同延迟 τ 下的皮尔逊相关系数来识别异常模式:

```
ρ(τ) = corr(BTC_returns[t], ALT_returns[t+τ])
```

当满足以下条件时,判定为异常模式(存在套利机会):

1. **长期高相关**: 7天周期相关系数 > 0.6
2. **短期低相关**: 1天周期相关系数 < 0.4
3. **显著差异**: 相关系数差值 > 0.38
4. **存在延迟**: 最优延迟 τ* > 0

### Beta 系数

Beta 系数用于衡量山寨币收益率相对 BTC 的跟随幅度:

```
β = Cov(BTC_returns, ALT_returns) / Var(BTC_returns)
```

- **β > 1.0**: 山寨币波动幅度大于 BTC (高风险)
- **β = 1.0**: 与 BTC 同步波动
- **β < 1.0**: 波动幅度小于 BTC (相对稳健)

项目设定 Beta 阈值为 1.0,低于此值的币种不会触发告警。

### 异常值处理

采用 Winsorization 方法处理极端收益率:

- 下分位数: 0.1%
- 上分位数: 99.9%
- 将超出范围的值限制在分位数边界内

这种方法可以有效降低极端价格波动对统计分析的影响。

## 项目结构

```
hyperliquid-btc-lag-tracker/
├── hyperliquid_analyzer.py    # 核心分析模块
├── utils/                      # 工具模块
│   ├── config.py              # 环境配置
│   ├── lark_bot.py            # 飞书机器人集成
│   ├── scheduler.py           # 定时调度器
│   ├── redisdb.py             # Redis 数据库工具
│   └── spider_failed_alert.py # 爬虫失败告警
├── pyproject.toml             # 项目依赖配置
├── README.md                  # 项目文档
└── hyperliquid.log            # 运行日志
```

## 快速开始

### 环境要求

- Python >= 3.12
- Redis (可选,用于数据缓存)

### 安装依赖

使用 uv (推荐):
```bash
uv sync
```

或使用 pip:
```bash
pip install -r requirements.txt
```

### 主要依赖

- **ccxt**: 加密货币交易所 API 统一接口
- **numpy**: 数值计算
- **pandas**: 数据分析
- **matplotlib**: 数据可视化
- **pyinform**: 信息论分析 (可选)
- **retry**: 自动重试机制

### 环境变量配置

创建 `.env` 文件或设置环境变量:

```bash
# 飞书机器人 Webhook ID (必需,用于接收告警通知)
export LARKBOT_ID="your-lark-bot-webhook-id"

# 环境标识 (可选,默认为 local)
export ENV="production"  # 生产环境启用定时调度

# Redis 配置 (可选)
export REDIS_HOST="127.0.0.1"
export REDIS_PASSWORD="your-redis-password"
```

### 运行分析

直接运行主程序:

```bash
python hyperliquid_analyzer.py
```

程序将自动:
1. 连接 Hyperliquid 交易所
2. 获取所有 USDC 永续合约交易对
3. 分析每个币种与 BTC 的相关性
4. 检测异常模式并通过飞书推送告警

## 核心模块说明

### DelayCorrelationAnalyzer

主要分析器类,包含以下核心方法:

#### 初始化参数

```python
analyzer = DelayCorrelationAnalyzer(
    exchange_name="hyperliquid",  # 交易所名称
    timeout=30000,                # 请求超时(毫秒)
    default_combinations=[        # K线组合
        ("5m", "7d"),  # 5分钟K线,7天数据
        ("1m", "1d")   # 1分钟K线,1天数据
    ]
)
```

#### 关键阈值配置

```python
# 相关系数阈值
LONG_TERM_CORR_THRESHOLD = 0.6   # 长期相关系数阈值
SHORT_TERM_CORR_THRESHOLD = 0.4  # 短期相关系数阈值
CORR_DIFF_THRESHOLD = 0.38       # 差值阈值

# Beta 系数配置
AVG_BETA_THRESHOLD = 1.0         # 平均 Beta 阈值

# 异常值处理配置
WINSORIZE_LOWER_PERCENTILE = 0.1    # 下分位数 (0.1%)
WINSORIZE_UPPER_PERCENTILE = 99.9   # 上分位数 (99.9%)
```

### 数据下载与缓存

```python
# 下载历史数据 (带自动重试)
df = analyzer.download_ccxt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m"
)

# 获取 BTC 数据 (带缓存)
btc_df = analyzer._get_btc_data(timeframe="5m", period="7d")

# 获取山寨币数据 (带缓存)
alt_df = analyzer._get_alt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m",
    coin="ETH"
)
```

### 相关性分析

```python
# 寻找最优延迟
tau_star, corrs, max_corr, beta = DelayCorrelationAnalyzer.find_optimal_delay(
    btc_ret=btc_returns,    # BTC 收益率数组
    alt_ret=alt_returns,    # 山寨币收益率数组
    max_lag=3,              # 最大延迟
    enable_outlier_treatment=True,  # 启用异常值处理
    enable_beta_calc=True   # 计算 Beta 系数
)

# 分析单个币种
is_anomaly = analyzer.one_coin_analysis("ETH/USDC:USDC")

# 批量分析所有币种
analyzer.run()
```

## 输出示例

### 控制台日志

```
2025-12-25 18:00:00 - __main__ - INFO - 启动分析器 | 交易所: hyperliquid | K线组合: [('5m', '7d'), ('1m', '1d')]
2025-12-25 18:00:05 - __main__ - INFO - 发现 150 个 USDC 永续合约交易对
2025-12-25 18:05:30 - __main__ - INFO - 发现异常币种 | 交易所: hyperliquid | 币种: DOGE/USDC:USDC | 差值: 0.42
2025-12-25 18:05:30 - __main__ - INFO - 分析进度: 37/150 (24%)
...
2025-12-25 18:30:00 - __main__ - INFO - 分析完成 | 交易所: hyperliquid | 总数: 150 | 异常: 8 | 跳过: 12 | 耗时: 1800.5s | 平均: 12.00s/币种
```

### 飞书告警通知

```
hyperliquid

DOGE/USDC:USDC 相关系数分析结果
相关系数  时间周期  数据周期  最优延迟  Beta系数
  0.8500      5m      7d       0     1.35
  0.4200      1m      1d       2     1.28

差值: 0.43
⚠️ 中等波动：平均Beta=1.32
```

## 定时调度

### 使用方式

在代码中使用 `@scheduled_task` 装饰器:

```python
from utils.scheduler import scheduled_task

# 每天 09:00 执行
@scheduled_task(start_time="09:00")
def daily_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()

# 每周二、周四、周六 14:30 执行
@scheduled_task(start_time="14:30", weekdays=[1, 3, 5])
def weekly_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()

# 每隔 3600 秒执行一次
@scheduled_task(duration=3600)
def hourly_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()
```

### 调度模式

1. **周几的几点执行**: 提供 `start_time` 和 `weekdays` 参数
2. **每天的几点执行**: 只提供 `start_time` 参数
3. **每隔 N 秒执行**: 只提供 `duration` 参数

注意: 在本地环境 (`ENV=local`) 下会直接执行,跳过定时调度。

## 配置说明

### 阈值调优建议

根据市场环境和策略需求,可以调整以下阈值:

```python
# 保守策略 (降低误报)
LONG_TERM_CORR_THRESHOLD = 0.7
SHORT_TERM_CORR_THRESHOLD = 0.3
CORR_DIFF_THRESHOLD = 0.45
AVG_BETA_THRESHOLD = 1.2

# 激进策略 (增加机会发现)
LONG_TERM_CORR_THRESHOLD = 0.5
SHORT_TERM_CORR_THRESHOLD = 0.5
CORR_DIFF_THRESHOLD = 0.30
AVG_BETA_THRESHOLD = 0.8
```

### K线组合选择

可以根据交易频率调整 K线组合:

```python
# 高频交易
combinations = [("1m", "1d"), ("5m", "3d")]

# 中长线交易
combinations = [("15m", "14d"), ("1h", "30d")]

# 超短线交易
combinations = [("1m", "6h"), ("5m", "1d")]
```

## 性能优化

### 缓存机制

- **BTC 数据缓存**: 避免重复下载相同周期的 BTC 数据
- **山寨币数据缓存**: 缓存已下载的山寨币数据

### 限流控制

- **请求间隔**: 1.5秒 (避免触发 Hyperliquid 限流)
- **币种间隔**: 2秒
- **启用速率限制**: `enableRateLimit=True`

### 数据验证

- **最小数据点**: 50 个数据点才进行分析
- **相关系数计算**: 至少需要 10 个数据点
- **Beta 系数计算**: 至少需要 10 个数据点

## 风险提示

1. **回测局限性**: 历史相关性不代表未来表现
2. **市场变化**: 市场结构变化可能导致相关性失效
3. **滑点成本**: 实际交易中需考虑滑点和手续费
4. **流动性风险**: 小市值币种可能存在流动性不足
5. **技术风险**: API 限流、网络延迟等技术问题

## 故障排查

### 常见问题

1. **飞书通知未发送**
   - 检查 `LARKBOT_ID` 环境变量是否正确
   - 验证 Webhook URL 是否有效
   - 查看日志中的错误信息

2. **数据下载失败**
   - 检查网络连接
   - 验证交易所 API 可用性
   - 查看是否触发限流 (增加请求间隔)

3. **分析结果为空**
   - 检查数据量是否足够 (>50 个数据点)
   - 验证交易对是否存在
   - 查看日志中的跳过原因

### 日志分析

日志文件位于 `hyperliquid.log`,采用轮转策略:
- 单文件最大 10MB
- 保留 5 个备份文件
- UTF-8 编码

日志级别说明:
- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息 (进度、统计)
- **WARNING**: 警告信息 (数据不足、跳过币种)
- **ERROR**: 错误信息 (下载失败、计算异常)

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

本项目基于原项目开发,请遵守相关开源许可。

## 致谢

- 原项目: [related_corrcoef_abnormal_alert](https://github.com/zhajingwen/related_corrcoef_abnormal_alert)
- CCXT 库: 提供统一的交易所 API 接口
- Hyperliquid 交易所: 数据来源

## 联系方式

如有问题或建议,请通过以下方式联系:

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**免责声明**: 本工具仅供学习和研究使用,不构成任何投资建议。使用本工具进行交易产生的风险和损失由使用者自行承担。

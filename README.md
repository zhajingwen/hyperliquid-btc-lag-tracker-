# Hyperliquid BTC 滞后性追踪器

一个用于分析 Hyperliquid 交易所中山寨币与 BTC 相关性的量化分析工具,通过识别短期低相关但长期高相关的异常币种,发现潜在的时间差套利机会。

## 项目背景与目标

### 理论基础

本项目是 **统计套利 (Statistical Arbitrage)** 在加密货币市场的创新应用，核心思想是通过识别山寨币与 BTC 之间的**时间延迟相关性**来发现套利机会。n

**🎯 策略分类**

- **主策略**：时间延迟配对交易 (Time-Lagged Pair Trading)
  - 以 BTC 为基准资产，山寨币为配对资产
  - 引入时间延迟维度 τ，捕捉价格传导的时间差

- **理论支撑**：
  - **协整理论 (Cointegration)**：长期相关性表明两个资产存在协整关系
  - **均值回归 (Mean Reversion)**：短期相关性破裂后预期回归至长期水平

**🔬 技术实现的三个维度**

1. **跨周期分析 (Cross-Timeframe Analysis)**
   - **长期视角**：5分钟K线 × 7天数据 → 识别稳定的跟随关系
   - **短期视角**：1分钟K线 × 1天数据 → 捕捉即时的价格延迟
   - **套利信号**：长期高相关（>0.6）但短期低相关（<0.4）

2. **延迟优化 (Lag Optimization)**
   - 搜索最优延迟 τ* ∈ [0, 3]，使相关系数最大化
   - **τ* > 0** 表明山寨币滞后于 BTC，存在时间差套利窗口
   - 通过延迟时间预判山寨币价格走势

3. **波动性评估 (Volatility Assessment)**
   - 计算 Beta 系数衡量山寨币相对 BTC 的波动幅度
   - **β ≥ 1.0** 保证足够的套利空间（波动幅度大于 BTC）
   - 使用 Winsorization 方法处理极端值，提高分析稳健性

### 核心需求

本项目旨在研究 **Hyperliquid 交易所**上满足以下三个维度特征的山寨币：

1. **跨周期特征**：长期（7天）高相关 + 短期（1天）低相关
   - 长期协整：ρ(7d) > 0.6，表明存在稳定的跟随关系
   - 短期破裂：ρ(1d) < 0.4，表明存在相关性破裂窗口

2. **延迟特征**：价格传导存在时间差
   - 最优延迟 τ* > 0，山寨币滞后于 BTC
   - 可预判性：通过 BTC 价格变动预测山寨币走势

3. **波动特征**：波动幅度大于 BTC
   - Beta 系数 β ≥ 1.0，保证足够的套利空间
   - 稳健性：异常值处理后的 Beta 仍满足阈值

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

**三维分析引擎**
- **跨周期分析**: 对比 5分钟/7天 与 1分钟/1天 两种K线组合，发现相关性差异
- **延迟优化**: 自动搜索最优延迟 τ* ∈ [0, 3]，捕捉价格传导时差
- **波动评估**: 计算 Beta 系数和异常值处理（Winsorization），确保分析稳健性

**自动化运维**
- **实时监控**: 持续跟踪所有 USDC 永续合约，自动识别异常模式
- **智能告警**: 通过飞书机器人推送套利机会，减少人工筛选成本
- **定时调度**: 支持自定义执行时间和周期，适配不同交易策略

## 技术原理

本章节详细介绍如何将上述理论转化为可执行的量化分析算法。

### 相关系数分析

通过计算不同延迟 τ 下的皮尔逊相关系数，量化山寨币与 BTC 的跟随关系：

```
ρ(τ) = corr(BTC_returns[t], ALT_returns[t+τ])
```

**异常模式识别（套利机会判定）**

同时满足以下任一组合条件时，判定存在套利机会：

**组合1：跨周期相关性破裂**
- 长期高相关：7天周期 ρ(0) > 0.6
- 短期低相关：1天周期 ρ(0) < 0.4
- 显著差异：Δρ = ρ_long - ρ_short > 0.38
- 波动充足：平均 β ≥ 1.0

**组合2：延迟传导模式**
- 长期高相关：7天周期 ρ(0) > 0.6
- 存在延迟：1天周期最优延迟 τ* > 0
- 波动充足：平均 β ≥ 1.0

### Beta 系数

Beta 系数用于衡量山寨币收益率相对 BTC 的跟随幅度:

```
β = Cov(BTC_returns, ALT_returns) / Var(BTC_returns)
```

- **β > 1.0**: 山寨币波动幅度大于 BTC (高风险)
- **β = 1.0**: 与 BTC 同步波动
- **β < 1.0**: 波动幅度小于 BTC (相对稳健)

项目设定 Beta 阈值为 1.0,低于此值的币种不会触发告警。

### Z-score（标准分数）

Z-score 是统计套利策略中的核心指标，用于量化当前价差相对于历史均值的偏离程度，从而判断套利机会的信号强度和交易方向。

#### 基本概念

Z-score 通过构建价差序列并计算其标准化偏离来衡量套利信号：

```
对数价差序列: spread = log(alt_prices) - β × log(btc_prices)
Z-score = (当前价差 - 历史均值) / 历史标准差
```

其中：
- **log(alt_prices)**: 山寨币对数价格序列
- **log(btc_prices)**: BTC 对数价格序列
- **β**: Beta 系数（基于对数价格计算，使用 `_calculate_beta_from_prices()`）
- **历史均值/标准差**: 基于滚动窗口（默认 20 个数据点）计算

**为什么使用对数价格？**
1. **消除价格量级差异**：BTC 价格 50000 vs 山寨币 0.01，直接相减无意义
2. **统计性质更稳定**：对数价格差分接近收益率，更符合平稳性假设
3. **符合协整理论**：这是学术研究和实践中的标准做法
4. **比例缩放不变性**：对数价差不受价格绝对值影响

**数据周期选择**

Z-score 计算使用 **1分钟K线 × 1天数据（1m/1d）**，而非 5分钟K线数据。这一设计决策基于以下考虑：

1. **更高的时间分辨率**：1分钟K线提供更细粒度的价格信息，能够更快地捕捉到价格异常和偏离
2. **更敏感的异常检测**：短期异常往往在分钟级别出现，使用1分钟数据能更及时地发现套利机会
3. **与跨周期分析逻辑一致**：
   - **5m/7d**（长期视角）：用于识别稳定的跟随关系，判断是否存在协整关系
   - **1m/1d**（短期视角）：用于捕捉即时价格延迟和异常，这正是Z-score的核心目标
4. **短期异常检测的定位**：Z-score 是短期异常检测工具，需要与"短期低相关"的检测周期保持一致，使用1分钟数据更符合这一设计理念

**为什么不使用5分钟K线？**
- 时间分辨率较低，可能错过短期的价格异常
- 对价格偏离的敏感度相对较低
- 与"短期低相关"检测逻辑不一致（系统通过对比长期5m/7d高相关与短期1m/1d低相关来识别套利机会）

#### Z-score 的含义

Z-score 表示当前价差偏离历史均值的标准差倍数：

| Z-score 范围 | 含义 | 套利信号强度 |
|-------------|------|------------|
| \|Z\| > 3 | 极强偏离 | 强套利信号 |
| 2 < \|Z\| ≤ 3 | 显著偏离 | 中等套利信号 |
| 1 < \|Z\| ≤ 2 | 中等偏离 | 弱套利信号 |
| \|Z\| ≤ 1 | 正常波动 | 无套利信号 |

#### 交易方向判断

Z-score 的正负值直接指示交易方向：

- **Z-score > 0（正数）**：
  - 含义：当前价差高于历史均值，山寨币相对 BTC 偏高
  - 预期：价差会回归均值，山寨币相对 BTC 会下跌
  - **交易方向**：做空山寨币 / 做多 BTC

- **Z-score < 0（负数）**：
  - 含义：当前价差低于历史均值，山寨币相对 BTC 偏低
  - 预期：价差会回归均值，山寨币相对 BTC 会上涨
  - **交易方向**：做多山寨币 / 做空 BTC

#### 实际应用示例

假设收到以下告警：

```
AR/USDC:USDC 相关系数分析结果
相关系数 时间周期 数据周期  最优延迟   Beta系数 
0.651925   5m   7d     0 1.344139 
0.336622   1m   1d     1 0.664220 

差值: 0.32
Beta系数: 1.00
📊 中等套利信号：Z-score=2.41（偏离2.4倍标准差）
📌 交易方向：做空AR/做多BTC
```

**解读**：
- Z-score = 2.41（正数），表示当前价差偏高
- 偏离历史均值 2.4 个标准差，属于显著偏离
- 预期价差会回归，AR 相对 BTC 会下跌
- 交易策略：做空 AR，同时做多 BTC（构建市场中性组合）

#### 配置参数

Z-score 相关配置位于 `DelayCorrelationAnalyzer` 类中：

```python
# Z-score 相关配置（类常量）
ENABLE_ZSCORE_CHECK = True        # 是否启用 Z-score 检查
ZSCORE_THRESHOLD = 2.0            # Z-score 阈值，超过此值才触发告警
ZSCORE_WINDOW = 20                # 滚动窗口大小（建议 20-30）

# 平稳性检验配置（新增）
ENABLE_STATIONARITY_CHECK = True        # 是否启用平稳性检验
STATIONARITY_STRONG_THRESHOLD = 0.05    # 强平稳阈值（p-value < 0.05）
STATIONARITY_WEAK_THRESHOLD = 0.10      # 弱平稳阈值（p-value < 0.10）
ENABLE_WEAK_SIGNAL_FEISHU = True        # 是否发送弱信号告警
```

#### 平稳性检验的重要性

配对交易的核心假设是**价差均值回归**，这要求价差序列必须是**平稳的**。

**什么是平稳性？**
- **无趋势**：价差不会持续上涨或下跌
- **方差稳定**：价差的波动幅度相对恒定
- **均值回归**：价差会围绕固定均值波动

**ADF 检验（Augmented Dickey-Fuller Test）**

系统使用 ADF 检验来验证价差序列的平稳性：

- **原假设（H0）**：序列是非平稳的（有单位根）
- **判断标准**：如果 p-value < 0.05，拒绝原假设，认为序列是平稳的
- **验证流程**：只有通过平稳性检验的价差才会计算 Z-score

**为什么需要平稳性检验？**

如果价差序列非平稳（持续漂移），Z-score 的均值回归假设不成立，可能导致：
- 价差持续扩大，但 Z-score 仍提示"回归"
- 系统性误判套利机会
- 策略失效

通过平稳性检验，系统会自动过滤掉不适合配对交易的币种，提高信号质量。

#### 平稳性分级标准

系统根据 ADF 检验的 p-value 将价差序列分为三个等级：

| 平稳性等级 | p-value 范围 | 含义 | 告警行为 |
|-----------|-------------|------|---------|
| **强平稳** | p < 0.05 | 价差序列高度平稳，均值回归假设成立 | ✅ 正常告警，信号可靠 |
| **弱平稳** | 0.05 ≤ p < 0.10 | 价差序列弱平稳，均值回归假设较弱 | ⚠️ 弱信号告警，谨慎交易 |
| **非平稳** | p ≥ 0.10 | 价差序列非平稳，不适合配对交易 | ❌ 非平稳告警（可选禁用） |

**配置说明**：

- `STATIONARITY_STRONG_THRESHOLD = 0.05`：强平稳阈值，低于此值认为强平稳
- `STATIONARITY_WEAK_THRESHOLD = 0.10`：弱平稳阈值，介于两个阈值之间认为弱平稳
- `ENABLE_WEAK_SIGNAL_FEISHU = True`：是否对弱信号和非平稳信号发送告警

**实际应用**：

- **强平稳信号**：可以正常交易，Z-score 均值回归假设可靠
- **弱平稳信号**：建议谨慎交易，适当降低仓位或观察更长时间
- **非平稳信号**：不建议交易，价差可能持续漂移而不回归

### 异常值处理

采用 Winsorization 方法处理极端收益率:

- **下分位数**: 0.1% (`WINSORIZE_LOWER_PERCENTILE = 0.1`)
- **上分位数**: 99.9% (`WINSORIZE_UPPER_PERCENTILE = 99.9`)
- **启用开关**: `ENABLE_OUTLIER_TREATMENT = True`
- **处理方式**: 将超出范围的值限制在分位数边界内（使用 `np.clip`）
- **数据要求**: 数据点少于 20 个时不进行异常值处理

这种方法可以有效降低极端价格波动对统计分析的影响，提高分析稳健性。

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

或使用 pip（需要先创建 requirements.txt）:
```bash
pip install -r requirements.txt
```

**注意**：项目使用 `pyproject.toml` 管理依赖，推荐使用 `uv` 或 `pip install -e .` 安装。

### 主要依赖

**必需依赖**：
- **ccxt** (>=4.5.14): 加密货币交易所 API 统一接口
- **numpy** (>=2.3.4): 数值计算
- **pandas** (>=2.3.3): 数据分析
- **retry** (>=0.9.2): 自动重试机制

**可选依赖**：
- **matplotlib** (>=3.10.7): 数据可视化
- **seaborn** (>=0.13.2): 统计图表绘制
- **pyinform** (>=0.2.0): 信息论分析
- **redis** (>=7.1.0): Redis 数据库工具（用于数据缓存）
- **hyperliquid-python-sdk** (>=0.8.0): Hyperliquid 交易所 Python SDK

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
1. 连接 Hyperliquid 交易所（使用 CCXT 库）
2. 获取所有 USDC 永续合约交易对（排除 BTC/USDC:USDC）
3. 分析每个币种与 BTC 的相关性（跨周期 + 延迟优化）
4. 计算 Beta 系数和 Z-score 进行验证
5. 检测异常模式并通过飞书推送告警
6. 输出进度报告（25%、50%、75%、100%）
7. 生成分析统计（总数、异常数、跳过数、耗时等）

**注意**：
- 程序会自动跳过数据不足或数据不存在的交易对
- 每个币种分析间隔 2 秒，避免触发 API 限流
- 数据下载请求间隔 1.5 秒，确保安全

## 核心模块说明

### DelayCorrelationAnalyzer

主要分析器类,包含以下核心方法:

#### 初始化参数

```python
analyzer = DelayCorrelationAnalyzer(
    exchange_name="hyperliquid",  # 交易所名称（默认: "kucoin"，但实际使用 "hyperliquid"）
    timeout=30000,                # 请求超时(毫秒)
    default_combinations=[        # K线组合（默认: [("5m", "7d"), ("1m", "1d")]）
        ("5m", "7d"),  # 5分钟K线,7天数据
        ("1m", "1d")   # 1分钟K线,1天数据
    ]
)
```

**注意**：代码中 `exchange_name` 的默认值是 `"kucoin"`，但在主程序中使用的是 `"hyperliquid"`。建议显式指定交易所名称。

#### 关键阈值配置

这些阈值直接对应理论基础中的三个维度分析：

```python
# ========== 跨周期分析阈值 ==========
LONG_TERM_CORR_THRESHOLD = 0.6   # 协整关系判定：长期相关性下限
SHORT_TERM_CORR_THRESHOLD = 0.4  # 相关性破裂判定：短期相关性上限
CORR_DIFF_THRESHOLD = 0.38       # 套利信号触发：最小相关性差值

# ========== 波动性评估阈值 ==========
AVG_BETA_THRESHOLD = 1.0         # 波动充足性判定：保证套利空间
WINSORIZE_LOWER_PERCENTILE = 0.1    # 异常值下限（0.1%分位数）
WINSORIZE_UPPER_PERCENTILE = 99.9   # 异常值上限（99.9%分位数）

# ========== 延迟优化配置 ==========
# 最大延迟搜索范围在 find_optimal_delay() 方法中配置，默认为 3

# ========== Z-score 配置 ==========
ENABLE_ZSCORE_CHECK = True        # 是否启用 Z-score 检查
ZSCORE_THRESHOLD = 2.0            # Z-score 阈值，超过此值才触发告警
ZSCORE_WINDOW = 20                # 滚动窗口大小（建议 20-30）

# ========== 异常值处理配置 ==========
ENABLE_OUTLIER_TREATMENT = True   # 是否启用异常值处理（Winsorization）

# ========== Beta 系数配置 ==========
ENABLE_BETA_CALCULATION = True     # 是否计算 Beta 系数
MIN_POINTS_FOR_BETA_CALC = 10     # Beta 系数计算所需的最小数据点
```

### 数据下载与缓存

```python
# 下载历史数据 (带自动重试，最多重试10次)
df = analyzer.download_ccxt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m"
)

# 获取 BTC 数据 (带缓存，避免重复下载)
btc_df = analyzer._get_btc_data(timeframe="5m", period="7d")

# 获取山寨币数据 (带缓存，自动验证数据有效性)
alt_df = analyzer._get_alt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m",
    coin="ETH"
)
```

**数据验证机制**：
- 自动检查数据是否为空
- 验证数据量是否满足最小要求（默认 50 个数据点）
- 空数据或数据不足时不缓存，避免后续误用

**支持的 Timeframe 格式**：
- 分钟：`1m`, `5m`, `15m`, `30m`
- 小时：`1h`, `4h`, `12h`
- 天：`1d`, `3d`
- 周：`1w`

**Period 格式**：
- 支持天数格式，如 `1d`, `7d`, `30d`
- 自动转换为对应的 K 线数量

### 相关性分析

```python
# 寻找最优延迟（增强版：支持异常值处理和 Beta 系数计算）
tau_star, corrs, max_corr, beta = DelayCorrelationAnalyzer.find_optimal_delay(
    btc_ret=btc_returns,    # BTC 收益率数组
    alt_ret=alt_returns,    # 山寨币收益率数组
    max_lag=3,              # 最大延迟（默认 3）
    enable_outlier_treatment=True,  # 启用异常值处理（默认使用类常量）
    enable_beta_calc=True   # 计算 Beta 系数（默认使用类常量）
)

# 计算 Z-score（用于量化套利信号强度）
zscore = DelayCorrelationAnalyzer._calculate_zscore(
    btc_prices=btc_price_series,  # BTC 价格序列
    alt_prices=alt_price_series,  # 山寨币价格序列
    beta=beta_value,              # Beta 系数
    window=20                     # 滚动窗口大小（默认 20）
)

# 获取交易方向
direction_desc, direction_code = DelayCorrelationAnalyzer._get_trading_direction(
    zscore=zscore_value,
    coin="ETH/USDC:USDC"
)

# 分析单个币种（自动进行 Z-score 验证）
is_anomaly = analyzer.one_coin_analysis("ETH/USDC:USDC")

# 批量分析所有币种（带进度报告）
analyzer.run()
```

**错误处理机制**：
- `_safe_execute()`: 统一错误处理，捕获异常并记录日志
- `_safe_download()`: 安全下载数据，失败时返回 None 而不抛出异常
- 自动重试机制：`download_ccxt_data()` 使用 `@retry` 装饰器，最多重试 10 次

**进度报告**：
- 在 25%、50%、75%、100% 进度时自动输出日志
- 显示已处理币种数、总币种数和百分比

## 输出示例

### 控制台日志

```
2025-12-25 18:00:00 - __main__ - INFO - 启动分析器 | 交易所: hyperliquid | K线组合: [('5m', '7d'), ('1m', '1d')]
2025-12-25 18:00:05 - __main__ - INFO - 发现 150 个 USDC 永续合约交易对
2025-12-25 18:05:30 - __main__ - INFO - 发现异常币种 | 交易所: hyperliquid | 币种: DOGE/USDC:USDC | 差值: 0.42
2025-12-25 18:10:00 - __main__ - INFO - 分析进度: 38/150 (25%)
2025-12-25 18:20:00 - __main__ - INFO - 分析进度: 75/150 (50%)
2025-12-25 18:30:00 - __main__ - INFO - 分析进度: 113/150 (75%)
...
2025-12-25 18:30:00 - __main__ - INFO - 分析完成 | 交易所: hyperliquid | 总数: 150 | 异常: 8 | 跳过: 12 | 耗时: 1800.5s | 平均: 12.00s/币种
2025-12-25 18:30:01 - __main__ - INFO - 平稳性分布统计 | 强平稳: 5 (62.5%) | 弱平稳: 2 (25.0%) | 非平稳: 1 (12.5%)
```

**进度报告机制**：
- 自动在 25%、50%、75%、100% 进度时输出日志
- 显示当前处理的币种数和总币种数
- 便于监控长时间运行的批量分析任务

### 飞书告警通知

告警通知会根据价差平稳性自动分级，提供不同的交易建议：

#### 强平稳信号示例（推荐交易）

```
hyperliquid

DOGE/USDC:USDC 相关系数分析结果
相关系数  时间周期  数据周期  最优延迟  Beta系数
  0.8500      5m      7d       0     1.35
  0.4200      1m      1d       2     1.28

差值: 0.43
⚠️ 中等波动：平均Beta=1.32
✅ 价差平稳性：强平稳 (p=0.012 < 0.05)
📊 中等套利信号：Z-score=2.41（偏离2.4倍标准差）
📌 交易方向：做空DOGE/做多BTC
```

#### 弱平稳信号示例（谨慎交易）

```
hyperliquid

AR/USDC:USDC 相关系数分析结果
相关系数  时间周期  数据周期  最优延迟  Beta系数
  0.7200      5m      7d       0     1.15
  0.3800      1m      1d       1     1.08

差值: 0.34
⚠️ 中等波动：平均Beta=1.12
⚠️ 价差平稳性：弱平稳 (p=0.078)，谨慎交易
📈 弱套利信号：Z-score=1.85（偏离1.9倍标准差）
📌 交易方向：做多AR/做空BTC
```

#### 非平稳信号示例（不建议交易）

```
hyperliquid

SOL/USDC:USDC 相关系数分析结果
相关系数  时间周期  数据周期  最优延迟  Beta系数
  0.6800      5m      7d       0     1.42
  0.3500      1m      1d       1     1.38

差值: 0.33
⚠️ 中等波动：平均Beta=1.40
❌ 价差平稳性：非平稳 (p=0.156)，不适合配对交易
Z-score 计算跳过（价差非平稳）
```

## 自动化交易实现

基于 Z-score 的套利信号，可以实现自动化配对交易系统。本节介绍如何将分析结果转化为可执行的交易策略。

### 交易信号生成

当收到告警时，系统会提供以下关键信息：

1. **Z-score 值**：量化价差偏离程度
2. **交易方向**：根据 Z-score 正负值确定
3. **Beta 系数**：用于计算对冲比例
4. **信号强度**：强/中等/弱

### 交易方向判断逻辑

```python
# 价差公式
spread = alt_prices - beta * btc_prices

# Z-score 判断
if zscore > 0:  # 价差偏高
    # 做空 AR，做多 BTC（价差会缩小）
    signal = "做空AR/做多BTC"
    
elif zscore < 0:  # 价差偏低
    # 做多 AR，做空 BTC（价差会扩大）
    signal = "做多AR/做空BTC"
```

### 仓位计算（对冲比例）

基于 Beta 系数计算对冲仓位：

```python
# 方案1：等价值对冲（推荐，简单易行）
alt_position_value = available_capital / 2  # 一半资金做空 AR
btc_position_value = available_capital / 2  # 一半资金做多 BTC

# 方案2：Beta 加权对冲（更精确，考虑波动性）
alt_position_value = available_capital / (1 + beta)
btc_position_value = available_capital * beta / (1 + beta)
```

**示例**：
- 账户余额：10,000 USDC
- 单笔风险：2%（200 USDC）
- Beta = 1.0
- AR 价格：3 USDC
- BTC 价格：50,000 USDC

**等价值对冲**：
- 做空 AR：100 USDC / 3 = 33.33 个 AR
- 做多 BTC：100 USDC / 50,000 = 0.002 个 BTC

### 交易执行流程

#### 1. 开仓（Entry）

当 Z-score 达到阈值时（|Z| ≥ 2.0）：

```python
# 1. 生成交易信号
signal = generate_trading_signal(coin, zscore, beta)

# 2. 计算仓位大小
position = calculate_position_size(signal, account_balance, risk_per_trade=0.02)

# 3. 同时执行两个订单（尽量同步）
if signal['direction'] == 'short_alt_long_btc':
    # 做空 AR（开空仓）
    alt_order = exchange.create_market_sell_order('AR/USDC:USDC', alt_size)
    # 做多 BTC（开多仓）
    btc_order = exchange.create_market_buy_order('BTC/USDC:USDC', btc_size)
```

#### 2. 监控（Monitoring）

持续监控 Z-score 变化：

```python
while position_is_open:
    current_zscore = get_current_zscore(coin)
    
    # 检查退出条件
    should_exit, reason = check_exit_signal(current_zscore, entry_zscore)
    
    if should_exit:
        close_position(position, reason)
        break
    
    time.sleep(60)  # 每分钟检查一次
```

#### 3. 平仓（Exit）

退出条件：

| 条件 | 说明 | 策略 |
|------|------|------|
| **止盈** | Z-score 回归到 0 附近（\|Z\| < 0.5） | 价差回归，获利了结 |
| **止损** | Z-score 继续扩大（\|Z\| > 4.0） | 价差未回归，及时止损 |
| **反向信号** | Z-score 符号改变 | 市场方向反转 |
| **时间止损** | 持仓超过最大时间（如 24 小时） | 避免长期持仓风险 |

### 风险控制参数

建议的风险控制配置：

```python
RISK_PARAMS = {
    'max_position_size': 0.02,      # 单笔交易最大 2% 资金
    'max_total_exposure': 0.10,    # 总敞口不超过 10%
    'stop_loss_zscore': 4.0,       # 止损 Z-score
    'take_profit_zscore': 0.5,     # 止盈 Z-score
    'max_holding_time': 24,         # 最大持仓时间（小时）
    'min_zscore_for_entry': 2.0,    # 最小 Z-score 才开仓
}
```

### 完整交易流程示例

```python
# 1. 收到告警
alert = {
    'coin': 'AR/USDC:USDC',
    'zscore': 2.41,
    'beta': 1.00
}

# 2. 生成交易信号
signal = generate_trading_signal(
    alert['coin'], 
    alert['zscore'], 
    alert['beta']
)
# signal = {
#     'direction': 'short_alt_long_btc',
#     'alt_action': 'SELL',
#     'btc_action': 'BUY'
# }

# 3. 计算仓位
position = calculate_position_size(
    signal, 
    account_balance=10000, 
    risk_per_trade=0.02
)

# 4. 执行交易
result = execute_pair_trade(position)

# 5. 持续监控，等待平仓信号
while True:
    should_exit, reason = check_exit_signal(position)
    if should_exit:
        close_position(position, reason)
        break
    time.sleep(60)  # 每分钟检查一次
```

### 注意事项

1. **滑点控制**：使用限价单而非市价单，减少滑点损失
2. **手续费考虑**：计算收益时需扣除交易成本
3. **流动性检查**：确保两个交易对都有足够流动性
4. **同步执行**：尽量同时下单，减少价格变动风险
5. **持续监控**：实时跟踪 Z-score，及时调整或平仓
6. **市场中性**：通过做多/做空组合，降低市场整体波动风险

### 策略优势

- **市场中性**：通过配对交易，降低市场整体风险
- **统计优势**：基于均值回归理论，具有统计学基础
- **自动化**：可完全自动化执行，减少人工干预
- **风险可控**：通过 Z-score 阈值和止损机制控制风险

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

根据市场环境和策略需求，可以调整阈值来平衡**准确性**与**机会捕获率**：

```python
# ========== 保守策略：强调协整关系稳定性 ==========
# 适用场景：追求高质量信号，降低假阳性
LONG_TERM_CORR_THRESHOLD = 0.7    # 更强的协整关系要求
SHORT_TERM_CORR_THRESHOLD = 0.3   # 更明显的相关性破裂
CORR_DIFF_THRESHOLD = 0.45        # 更大的套利窗口
AVG_BETA_THRESHOLD = 1.2          # 更高的波动性要求

# ========== 激进策略：扩大捕获范围 ==========
# 适用场景：挖掘更多潜在机会，容忍更高假阳性
LONG_TERM_CORR_THRESHOLD = 0.5    # 放宽协整关系要求
SHORT_TERM_CORR_THRESHOLD = 0.5   # 接受较高的短期相关性
CORR_DIFF_THRESHOLD = 0.30        # 更小的差值即可触发
AVG_BETA_THRESHOLD = 0.8          # 接受较低波动性
```

**调优原则**：
- **协整强度** (LONG_TERM)：值越高，均值回归假设越可靠，但机会越少
- **破裂程度** (SHORT_TERM)：值越低，相关性破裂越明显，信号越强
- **套利窗口** (CORR_DIFF)：值越大，套利空间越大，但触发频率越低
- **波动要求** (BETA)：值越高，收益潜力越大，但风险也越高

### K线组合选择

可以根据交易频率调整 K线组合:

```python
# 高频交易（默认配置）
combinations = [("5m", "7d"), ("1m", "1d")]

# 中长线交易
combinations = [("15m", "14d"), ("1h", "30d")]

# 超短线交易
combinations = [("1m", "6h"), ("5m", "1d")]

# 自定义组合（在初始化时指定）
analyzer = DelayCorrelationAnalyzer(
    exchange_name="hyperliquid",
    default_combinations=[("15m", "14d"), ("1h", "30d")]
)
```

**支持的 Timeframe 格式**：
- 分钟：`1m`, `5m`, `15m`, `30m`
- 小时：`1h`, `4h`, `12h`
- 天：`1d`, `3d`
- 周：`1w`

**Period 格式**：
- 支持天数格式，如 `1d`, `7d`, `30d`
- 系统会自动将 period 和 timeframe 转换为对应的 K 线数量

## 性能优化

### 缓存机制

- **BTC 数据缓存**: 避免重复下载相同周期的 BTC 数据
  - 缓存键：`(timeframe, period)`
  - 同一周期内所有币种共享 BTC 数据
- **山寨币数据缓存**: 缓存已下载的山寨币数据
  - 缓存键：`(symbol, timeframe, period)`
  - 自动验证缓存数据的有效性（非空且数据量足够）
  - 空数据或数据不足时不缓存，避免后续误用
- **缓存策略**: 
  - 仅在数据验证通过后才缓存
  - 缓存的是 DataFrame 的副本，避免意外修改

### 限流控制

- **请求间隔**: 1.5秒（在 `download_ccxt_data()` 中，避免触发 Hyperliquid 限流）
- **币种间隔**: 2秒（在 `run()` 方法中，每个币种分析后等待）
- **启用速率限制**: `enableRateLimit=True`（CCXT 交易所配置）
- **速率限制值**: `rateLimit=1500`（毫秒，即 1.5 秒）
- **请求超时**: `timeout=30000`（30 秒）

### 数据验证

- **最小数据点**: 50 个数据点才进行分析（`MIN_DATA_POINTS_FOR_ANALYSIS`）
- **相关系数计算**: 至少需要 10 个数据点（`MIN_POINTS_FOR_CORR_CALC`）
- **Beta 系数计算**: 至少需要 10 个数据点（`MIN_POINTS_FOR_BETA_CALC`）
- **Z-score 计算**: 至少需要窗口大小（默认 20）个数据点
- **空数据检查**: 自动检测并跳过空数据或数据不足的交易对
- **数据对齐**: 自动对齐 BTC 和山寨币的时间索引，确保数据一致性

### 错误处理

- **自动重试**: `download_ccxt_data()` 使用 `@retry` 装饰器
  - 最多重试 10 次 (`tries=10`)
  - 初始延迟 5 秒 (`delay=5`)
  - 指数退避策略 (`backoff=2`)
  - 所有重试都会记录到日志
- **安全执行**: `_safe_execute()` 统一捕获异常，避免程序崩溃
- **安全下载**: `_safe_download()` 失败时返回 None 而不抛出异常
- **日志记录**: 所有错误都会记录到日志文件，便于排查问题

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

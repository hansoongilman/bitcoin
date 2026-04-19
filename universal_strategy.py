"""
universal_strategy.py
v4.0 Universal - 과적합 제거, 모든 코인에서 범용적으로 작동하는 전략

설계 원칙:
1. 파라미터 최소화 - 모두 업계 표준 기본값 사용 (데이터마이닝 X)
2. 진입: 추세 + 모멘텀 + 거래량 확인 (3개 조건만)
3. 청산: 순수 ATR 트레일링 스탑만 사용 (지표 기반 청산 제거)
4. 어떤 StochRSI/BB/ADX 임계값도 사용하지 않음

파라미터 (전부 업계 표준 디폴트):
- EMA 21일 (추세 필터)
- MACD 12/26/9 (표준 MACD)
- ATR 14일, 배수 2.0 (트레일링 스탑)
- Volume MA 20일
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta, warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# Indicators (standard defaults only)
# ══════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    df['EMA21']    = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    m = ta.trend.MACD(df['Close'], 12, 26, 9)  # STANDARD MACD defaults
    df['MACD']     = m.macd()
    df['MACD_Sig'] = m.macd_signal()
    df['ATR']      = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio']= df['Volume'] / df['Vol_MA20']
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════
# v4 Universal Strategy
# ══════════════════════════════════════════════════════════
def backtest_v4(df, initial=10000.0, trail_mult=2.0):
    """
    진입: Close > EMA21 AND MACD > Signal AND Volume > MA
    청산: 오직 트레일링 스탑 (최고가 - 2*ATR)
    """
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []
    trailing_high = 0.0

    for ts, row in df.iterrows():
        px = float(row['Close'])
        hi = float(row['High'])
        atr = float(row['ATR'])

        if pos > 0:
            if hi > trailing_high:
                trailing_high = hi
            trail_stop = trailing_high - trail_mult * atr

            if px < trail_stop:
                exit_px = max(trail_stop, float(row['Low']))
                cash += pos * exit_px
                trades.append({'profit': (exit_px / ep - 1)})
                pos = 0; ep = 0; trailing_high = 0

        if pos == 0:
            if (px > float(row['EMA21']) and
                float(row['MACD']) > float(row['MACD_Sig']) and
                float(row['Vol_Ratio']) > 1.0):
                ep = px; pos = cash / px; cash = 0
                trailing_high = hi

        equity.append(cash + pos * px)

    if not equity:
        return 0, 0, 0, 0, 0, 0
    final = equity[-1]
    cum_r = final / initial - 1
    peak = pd.Series(equity).cummax()
    mdd = ((pd.Series(equity) - peak) / peak).min()
    dr = pd.Series(equity).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    wins = len([t for t in trades if t['profit'] > 0])
    wr = wins / len(trades) if trades else 0
    bh = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0]) - 1
    return cum_r, mdd, sharpe, wr, len(trades), bh

# v2 baseline for comparison
def backtest_v2(df, initial=10000.0):
    df2 = df.copy()
    df2['SMA30'] = ta.trend.SMAIndicator(df2['Close'], window=30).sma_indicator()
    m2 = ta.trend.MACD(df2['Close'], 12, 21, 7)
    df2['MACD2'] = m2.macd()
    df2['MACD2_Sig'] = m2.macd_signal()
    df2['StochRSI'] = ta.momentum.StochasticOscillator(df2['High'], df2['Low'], df2['Close'], 14, 3).stoch()
    bb = ta.volatility.BollingerBands(df2['Close'], 20)
    df2['BB_Pct'] = bb.bollinger_pband()
    df2['BB_Lower'] = bb.bollinger_lband()
    df2['ATR2'] = ta.volatility.AverageTrueRange(df2['High'], df2['Low'], df2['Close'], 14).average_true_range()
    vma = df2['Volume'].rolling(20).mean()
    df2['VR2'] = df2['Volume'] / vma
    df2.dropna(inplace=True)

    pos=0; cash=initial; ep=0; equity=[]
    for ts, row in df2.iterrows():
        px = float(row['Close'])
        if pos > 0:
            sl = np.clip(1.5*float(row['ATR2'])/ep, 0.03, 0.07)
            if float(row['Low']) <= ep*(1-sl):
                cash += pos*ep*(1-sl); pos=0; ep=0
            elif (px < float(row['SMA30']) or float(row['MACD2']) < float(row['MACD2_Sig']) or
                  float(row['StochRSI']) > 85 or px < float(row['BB_Lower'])):
                cash += pos*px; pos=0; ep=0
        if pos == 0:
            if (px > float(row['SMA30']) and float(row['MACD2']) > float(row['MACD2_Sig']) and
                float(row['StochRSI']) < 80 and float(row['BB_Pct']) > 0.4 and float(row['VR2']) > 1.0):
                ep=px; pos=cash/px; cash=0
        equity.append(cash + pos*px)
    if not equity: return 0, 0
    cum_r = equity[-1]/initial - 1
    peak = pd.Series(equity).cummax()
    mdd = ((pd.Series(equity) - peak) / peak).min()
    return cum_r, mdd

# v3 for comparison
def backtest_v3(df, initial=10000.0):
    df3 = df.copy()
    df3['SMA30'] = ta.trend.SMAIndicator(df3['Close'], window=30).sma_indicator()
    df3['EMA10'] = ta.trend.EMAIndicator(df3['Close'], window=10).ema_indicator()
    m3 = ta.trend.MACD(df3['Close'], 12, 21, 7)
    df3['MACD3'] = m3.macd()
    df3['MACD3_Sig'] = m3.macd_signal()
    df3['StochRSI'] = ta.momentum.StochasticOscillator(df3['High'], df3['Low'], df3['Close'], 14, 3).stoch()
    bb = ta.volatility.BollingerBands(df3['Close'], 20)
    df3['BB_Pct'] = bb.bollinger_pband()
    df3['BB_Lower'] = bb.bollinger_lband()
    df3['ATR3'] = ta.volatility.AverageTrueRange(df3['High'], df3['Low'], df3['Close'], 14).average_true_range()
    vma = df3['Volume'].rolling(20).mean()
    df3['VR3'] = df3['Volume'] / vma
    adx = ta.trend.ADXIndicator(df3['High'], df3['Low'], df3['Close'], 14)
    df3['ADX'] = adx.adx()
    df3['DI_P'] = adx.adx_pos()
    df3['DI_M'] = adx.adx_neg()
    df3.dropna(inplace=True)

    pos=0; cash=initial; ep=0; equity=[]; th=0; ib=False
    for ts, row in df3.iterrows():
        px=float(row['Close']); hi=float(row['High'])
        bull = float(row['ADX'])>25 and float(row['DI_P'])>float(row['DI_M'])
        if pos>0:
            if hi>th: th=hi
            if bull or ib:
                ts_price = th - 2.0*float(row['ATR3'])
                ex=False
                if px<ts_price: ex=True
                elif float(row['StochRSI'])>95: ex=True
                elif px<float(row['EMA10']) and float(row['MACD3'])<float(row['MACD3_Sig']): ex=True
                elif px<float(row['BB_Lower']): ex=True
                if ex: cash+=pos*px; pos=0; ep=0; th=0; ib=False
                elif not bull:
                    ib=False
                    if (px<float(row['SMA30']) or float(row['MACD3'])<float(row['MACD3_Sig']) or
                        float(row['StochRSI'])>85 or px<float(row['BB_Lower'])):
                        cash+=pos*px; pos=0; ep=0; th=0
            else:
                sl=np.clip(1.5*float(row['ATR3'])/ep,0.03,0.07)
                if float(row['Low'])<=ep*(1-sl): cash+=pos*ep*(1-sl); pos=0; ep=0; th=0
                elif (px<float(row['SMA30']) or float(row['MACD3'])<float(row['MACD3_Sig']) or
                      float(row['StochRSI'])>85 or px<float(row['BB_Lower'])):
                    cash+=pos*px; pos=0; ep=0; th=0
        if pos==0:
            sl2=90 if bull else 80; vl2=0.8 if bull else 1.0
            if (px>float(row['SMA30']) and float(row['MACD3'])>float(row['MACD3_Sig']) and
                float(row['StochRSI'])<sl2 and float(row['BB_Pct'])>0.4 and float(row['VR3'])>vl2):
                ep=px; pos=cash/px; cash=0; th=hi; ib=bull
        equity.append(cash+pos*px)
    if not equity: return 0,0
    cum_r=equity[-1]/initial-1
    peak=pd.Series(equity).cummax()
    mdd=((pd.Series(equity)-peak)/peak).min()
    return cum_r, mdd


# ══════════════════════════════════════════════════════════
# Main - Comprehensive Testing
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ALL coins: original + new (never used in design)
    coins = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD',
        'AVAX-USD', 'LINK-USD', 'DOT-USD', 'MATIC-USD', 'NEAR-USD', 'UNI-USD',
    ]
    coin_labels = {
        'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','SOL-USD':'Solana',
        'XRP-USD':'Ripple','ADA-USD':'Cardano','DOGE-USD':'Dogecoin',
        'AVAX-USD':'Avalanche','LINK-USD':'Chainlink','DOT-USD':'Polkadot',
        'MATIC-USD':'Polygon','NEAR-USD':'NEAR','UNI-USD':'Uniswap',
    }

    print("=" * 110)
    print("  v4.0 UNIVERSAL STRATEGY - Overfitting-Free Validation")
    print("  Entry: EMA21 + MACD(12/26/9) + Volume | Exit: Pure Trailing Stop (2*ATR)")
    print("=" * 110)

    print("\nDownloading data...")
    raw = {}
    for tk in coins:
        print(f"  {tk}...")
        df = yf.download(tk, start='2021-06-01', end='2026-04-01', interval='1d', progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        raw[tk] = add_indicators(df)

    # ════════════════════════════════════════════════
    # PART 1: Full Period Comparison (2022~2024)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  [PART 1] In-Sample: 2022~2024 (all strategies)")
    print("=" * 110)
    print(f"  {'Coin':<14} {'v2':>8} {'v3':>8} {'v4':>8} {'B&H':>8} {'v4 MDD':>8} {'v4 Win%':>8} {'v4 Trds':>8}")
    print(f"  {'-' * 95}")

    for tk in coins:
        if tk not in raw: continue
        df = raw[tk]
        sub = df.loc['2022-01-01':'2024-12-31'].copy()
        if len(sub) < 50: continue

        v2_r, v2_m = backtest_v2(sub)
        v3_r, v3_m = backtest_v3(sub)
        v4_r, v4_m, v4_sh, v4_wr, v4_n, bh = backtest_v4(sub)

        tag = '' if tk in ['BTC-USD','ETH-USD','SOL-USD','XRP-USD','ADA-USD','DOGE-USD'] else ' [NEW]'
        print(f"  {coin_labels.get(tk,tk)+tag:<14} {v2_r*100:>+7.1f}% {v3_r*100:>+7.1f}% {v4_r*100:>+7.1f}% "
              f"{bh*100:>+7.1f}% {v4_m*100:>7.1f}% {v4_wr*100:>7.0f}% {v4_n:>7}")

    # ════════════════════════════════════════════════
    # PART 2: True Out-of-Sample (2025)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  [PART 2] TRUE OUT-OF-SAMPLE: 2025 (never used in any strategy design)")
    print("=" * 110)
    print(f"  {'Coin':<14} {'v2':>8} {'v3':>8} {'v4':>8} {'B&H':>8} {'v4>v2':>7} {'v4>v3':>7} {'v4>BH':>7}")
    print(f"  {'-' * 75}")

    v4_vs_v2 = 0; v4_vs_v3 = 0; v4_vs_bh = 0; total_oos = 0
    for tk in coins:
        if tk not in raw: continue
        df = raw[tk]
        sub = df.loc['2025-01-01':'2025-12-31'].copy()
        if len(sub) < 30: continue

        v2_r, _ = backtest_v2(sub)
        v3_r, _ = backtest_v3(sub)
        v4_r, _, _, _, _, bh = backtest_v4(sub)
        total_oos += 1
        if v4_r > v2_r: v4_vs_v2 += 1
        if v4_r > v3_r: v4_vs_v3 += 1
        if v4_r > bh: v4_vs_bh += 1

        print(f"  {coin_labels.get(tk,tk):<14} {v2_r*100:>+7.1f}% {v3_r*100:>+7.1f}% {v4_r*100:>+7.1f}% "
              f"{bh*100:>+7.1f}% {'OK' if v4_r>v2_r else 'NG':>7} {'OK' if v4_r>v3_r else 'NG':>7} {'OK' if v4_r>bh else 'NG':>7}")

    print(f"\n  2025 OOS Results: v4>v2 {v4_vs_v2}/{total_oos} | v4>v3 {v4_vs_v3}/{total_oos} | v4>B&H {v4_vs_bh}/{total_oos}")

    # ════════════════════════════════════════════════
    # PART 3: Walk-Forward by Half Year
    # ════════════════════════════════════════════════
    periods = [
        ('2022 H1', '2022-01-01', '2022-06-30'),
        ('2022 H2', '2022-07-01', '2022-12-31'),
        ('2023 H1', '2023-01-01', '2023-06-30'),
        ('2023 H2', '2023-07-01', '2023-12-31'),
        ('2024 H1', '2024-01-01', '2024-06-30'),
        ('2024 H2', '2024-07-01', '2024-12-31'),
        ('2025 H1', '2025-01-01', '2025-06-30'),
        ('2025 H2', '2025-07-01', '2025-12-31'),
    ]

    print("\n" + "=" * 110)
    print("  [PART 3] Period-by-Period: v4 vs B&H across ALL 12 coins")
    print("=" * 110)

    for pname, s, e in periods:
        wins = 0; total = 0; v4_avg = []; bh_avg = []
        for tk in coins:
            if tk not in raw: continue
            df = raw[tk]
            sub = df.loc[s:e].copy()
            if len(sub) < 20: continue
            v4_r, _, _, _, _, bh = backtest_v4(sub)
            total += 1
            if v4_r > bh: wins += 1
            v4_avg.append(v4_r * 100)
            bh_avg.append(bh * 100)

        avg_v4 = np.mean(v4_avg) if v4_avg else 0
        avg_bh = np.mean(bh_avg) if bh_avg else 0
        print(f"  {pname:<10}  v4 avg: {avg_v4:>+7.1f}%  B&H avg: {avg_bh:>+7.1f}%  "
              f"v4 beats B&H: {wins}/{total} coins  "
              f"{'(bear market defense)' if avg_bh < -10 else '(uptrend capture)' if avg_bh > 10 else '(sideways)'}")

    # ════════════════════════════════════════════════
    # PART 4: Parameter Stability Check
    # ════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  [PART 4] Parameter Stability (v4 uses ONLY trailing ATR mult)")
    print("  Testing across ALL 12 coins, Full period 2022~2025")
    print("=" * 110)

    for mult in [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
        rets = []
        for tk in coins:
            if tk not in raw: continue
            df = raw[tk]
            sub = df.loc['2022-01-01':'2025-12-31'].copy()
            if len(sub) < 50: continue
            r, _, _, _, _, _ = backtest_v4(sub, trail_mult=mult)
            rets.append(r * 100)
        avg = np.mean(rets)
        std = np.std(rets)
        print(f"  Trail={mult:.2f}x  |  avg: {avg:>+7.1f}%  std: {std:>6.1f}%  "
              f"range: [{min(rets):>+7.1f}% ~ {max(rets):>+7.1f}%]  "
              f"{'<-- DEFAULT' if mult == 2.0 else ''}")

    # ════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  STRATEGY COMPARISON SUMMARY")
    print("=" * 110)
    print(f"\n  {'Metric':<35} {'v2':>10} {'v3':>10} {'v4':>10}")
    print(f"  {'-' * 70}")
    print(f"  {'Tunable parameters':<35} {'5':>10} {'8':>10} {'1':>10}")
    print(f"  {'Data-mined thresholds':<35} {'3':>10} {'5':>10} {'0':>10}")
    print(f"  {'Indicator-based exits':<35} {'4':>10} {'5~6':>10} {'0':>10}")
    print(f"  {'Exit method':<35} {'Fixed SL':>10} {'Mixed':>10} {'Trail':>10}")

    print(f"\n  v4 Design Philosophy:")
    print(f"  - Entry:  EMA(21) trend + MACD(12/26/9) momentum + Volume confirmation")
    print(f"  - Exit:   ONLY trailing stop (highest high - 2*ATR)")
    print(f"  - Params: ALL industry standard defaults, ZERO data-mined values")
    print(f"  - Goal:   Consistent across ALL coins, NOT optimized for any single one")

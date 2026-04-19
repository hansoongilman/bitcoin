"""
overfit_validation.py
v3.0 전략 과적합 검증

검증 방법:
1. Walk-Forward Test: 앞쪽 데이터로 설계 -> 뒤쪽 데이터로 검증
2. Out-of-Sample: 2025년 데이터 (전략 설계에 전혀 사용 안 한 기간)
3. Parameter Sensitivity: 핵심 파라미터 변경 시 결과 얼마나 흔들리는지
4. 새로운 코인 테스트: 전략 설계 시 사용 안 한 코인으로 검증
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta, warnings
from itertools import product

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    df['EMA10']   = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    m = ta.trend.MACD(df['Close'], 12, 21, 7)
    df['MACD']    = m.macd()
    df['MACD_Sig']= m.macd_signal()
    df['StochRSI']= ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 14, 3).stoch()
    bb = ta.volatility.BollingerBands(df['Close'], 20)
    df['BB_Pct']  = bb.bollinger_pband()
    df['BB_Lower']= bb.bollinger_lband()
    df['ATR']     = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    vma = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / vma
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14)
    df['ADX']     = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus']= adx.adx_neg()
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════
# v2 Strategy (baseline)
# ══════════════════════════════════════════════════════════
def backtest_v2(df, initial=10000.0):
    pos=0.0; cash=initial; ep=0.0; equity=[]
    for ts, row in df.iterrows():
        px = float(row['Close'])
        if pos > 0:
            sl_pct = np.clip(1.5 * float(row['ATR']) / ep, 0.03, 0.07)
            if float(row['Low']) <= ep * (1 - sl_pct):
                cash += pos * ep * (1 - sl_pct); pos=0; ep=0
            elif (px < float(row['SMA30']) or float(row['MACD']) < float(row['MACD_Sig']) or
                  float(row['StochRSI']) > 85 or px < float(row['BB_Lower'])):
                cash += pos * px; pos=0; ep=0
        if pos == 0:
            if (px > float(row['SMA30']) and float(row['MACD']) > float(row['MACD_Sig']) and
                float(row['StochRSI']) < 80 and float(row['BB_Pct']) > 0.4 and float(row['Vol_Ratio']) > 1.0):
                ep = px; pos = cash / px; cash = 0
        equity.append(cash + pos * px)
    return (equity[-1] / initial - 1) if equity else 0

# ══════════════════════════════════════════════════════════
# v3 Strategy (parameterized for sensitivity test)
# ══════════════════════════════════════════════════════════
def backtest_v3(df, initial=10000.0,
                adx_thresh=25, srsi_entry_bull=90, srsi_exit_bull=95,
                trail_atr_mult=2.0):
    pos=0.0; cash=initial; ep=0.0; equity=[]
    trailing_high=0.0; in_bull=False

    for ts, row in df.iterrows():
        px = float(row['Close']); hi = float(row['High'])
        bull = float(row['ADX']) > adx_thresh and float(row['DI_Plus']) > float(row['DI_Minus'])

        if pos > 0:
            if hi > trailing_high: trailing_high = hi

            if bull or in_bull:
                trail_stop = trailing_high - trail_atr_mult * float(row['ATR'])
                exit_sig = False
                if px < trail_stop: exit_sig = True
                elif float(row['StochRSI']) > srsi_exit_bull: exit_sig = True
                elif px < float(row['EMA10']) and float(row['MACD']) < float(row['MACD_Sig']): exit_sig = True
                elif px < float(row['BB_Lower']): exit_sig = True

                if exit_sig:
                    cash += pos * px; pos=0; ep=0; trailing_high=0; in_bull=False
                elif not bull:
                    in_bull = False
                    if (px < float(row['SMA30']) or float(row['MACD']) < float(row['MACD_Sig']) or
                        float(row['StochRSI']) > 85 or px < float(row['BB_Lower'])):
                        cash += pos * px; pos=0; ep=0; trailing_high=0
            else:
                sl_pct = np.clip(1.5 * float(row['ATR']) / ep, 0.03, 0.07)
                if float(row['Low']) <= ep * (1 - sl_pct):
                    cash += pos * ep * (1 - sl_pct); pos=0; ep=0; trailing_high=0
                elif (px < float(row['SMA30']) or float(row['MACD']) < float(row['MACD_Sig']) or
                      float(row['StochRSI']) > 85 or px < float(row['BB_Lower'])):
                    cash += pos * px; pos=0; ep=0; trailing_high=0

        if pos == 0:
            srsi_lim = srsi_entry_bull if bull else 80
            vol_lim = 0.8 if bull else 1.0
            if (px > float(row['SMA30']) and float(row['MACD']) > float(row['MACD_Sig']) and
                float(row['StochRSI']) < srsi_lim and float(row['BB_Pct']) > 0.4 and float(row['Vol_Ratio']) > vol_lim):
                ep = px; pos = cash / px; cash = 0
                trailing_high = hi; in_bull = bull

        equity.append(cash + pos * px)
    return (equity[-1] / initial - 1) if equity else 0

# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':

    # Original coins + NEW coins never used in design
    original_coins = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']
    new_coins = ['AVAX-USD', 'LINK-USD', 'DOT-USD', 'MATIC-USD']
    all_coins = original_coins + new_coins

    print("=" * 100)
    print("  v3.0 OVERFITTING VALIDATION")
    print("=" * 100)

    # Download all data (2021-06 ~ 2026-04)
    print("\n[1/4] Downloading data...")
    raw = {}
    for tk in all_coins:
        print(f"  {tk}...")
        df = yf.download(tk, start='2021-06-01', end='2026-04-01', interval='1d', progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        raw[tk] = add_indicators(df)

    # ════════════════════════════════════════════════
    # TEST 1: Walk-Forward (In-sample vs Out-of-sample)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  [TEST 1] Walk-Forward Analysis")
    print("  Design period: 2022~2023 / Test period: 2024")
    print("  (v3 was designed using 2022~2024, so 2024 is partially in-sample)")
    print("=" * 100)
    print(f"  {'Coin':<12} {'v2 Train':>10} {'v3 Train':>10} {'v2 Test':>10} {'v3 Test':>10} {'v3 Degrades?':>14}")
    print(f"  {'-' * 80}")

    wf_results = []
    for tk in original_coins:
        if tk not in raw: continue
        df = raw[tk]
        train = df.loc['2022-01-01':'2023-12-31'].copy()
        test  = df.loc['2024-01-01':'2024-12-31'].copy()
        if len(train) < 30 or len(test) < 30: continue

        v2_train = backtest_v2(train) * 100
        v3_train = backtest_v3(train) * 100
        v2_test  = backtest_v2(test)  * 100
        v3_test  = backtest_v3(test)  * 100

        # Overfitting = big gap between train and test performance
        train_edge = v3_train - v2_train
        test_edge  = v3_test  - v2_test
        degrades = 'YES' if test_edge < train_edge * 0.3 else 'minor' if test_edge < train_edge * 0.7 else 'NO'

        wf_results.append({'coin': tk, 'v2_train': v2_train, 'v3_train': v3_train,
                           'v2_test': v2_test, 'v3_test': v3_test,
                           'train_edge': train_edge, 'test_edge': test_edge, 'degrades': degrades})
        print(f"  {tk:<12} {v2_train:>+9.1f}% {v3_train:>+9.1f}% {v2_test:>+9.1f}% {v3_test:>+9.1f}% {degrades:>14}")

    # ════════════════════════════════════════════════
    # TEST 2: True Out-of-Sample (2025 data)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  [TEST 2] TRUE OUT-OF-SAMPLE: 2025 (never used in strategy design)")
    print("=" * 100)
    print(f"  {'Coin':<12} {'v2':>10} {'v3':>10} {'B&H':>10} {'v3 > v2?':>10} {'v3 > B&H?':>10}")
    print(f"  {'-' * 70}")

    oos_v3_wins = 0
    oos_total = 0
    for tk in original_coins:
        if tk not in raw: continue
        df = raw[tk]
        oos = df.loc['2025-01-01':'2025-12-31'].copy()
        if len(oos) < 30: continue

        v2_r = backtest_v2(oos) * 100
        v3_r = backtest_v3(oos) * 100
        bh_r = (float(oos['Close'].iloc[-1]) / float(oos['Close'].iloc[0]) - 1) * 100

        oos_total += 1
        if v3_r > v2_r: oos_v3_wins += 1

        print(f"  {tk:<12} {v2_r:>+9.1f}% {v3_r:>+9.1f}% {bh_r:>+9.1f}% {'OK' if v3_r > v2_r else 'NG':>10} {'OK' if v3_r > bh_r else 'NG':>10}")

    print(f"\n  => v3 wins in {oos_v3_wins}/{oos_total} coins on completely unseen 2025 data")

    # ════════════════════════════════════════════════
    # TEST 3: New Coins (never used in design)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  [TEST 3] NEW COINS (AVAX, LINK, DOT, MATIC - never used in strategy design)")
    print("=" * 100)
    print(f"  {'Coin':<12} {'Period':<20} {'v2':>10} {'v3':>10} {'B&H':>10} {'v3>v2':>8}")
    print(f"  {'-' * 75}")

    new_v3_wins = 0
    new_total = 0
    for tk in new_coins:
        if tk not in raw: continue
        df = raw[tk]
        for period_name, (s, e) in [('2022~2024', ('2022-01-01', '2024-12-31')),
                                     ('2025 (OOS)', ('2025-01-01', '2025-12-31'))]:
            sub = df.loc[s:e].copy()
            if len(sub) < 30: continue

            v2_r = backtest_v2(sub) * 100
            v3_r = backtest_v3(sub) * 100
            bh_r = (float(sub['Close'].iloc[-1]) / float(sub['Close'].iloc[0]) - 1) * 100
            new_total += 1
            if v3_r > v2_r: new_v3_wins += 1

            print(f"  {tk:<12} {period_name:<20} {v2_r:>+9.1f}% {v3_r:>+9.1f}% {bh_r:>+9.1f}% {'OK' if v3_r > v2_r else 'NG':>8}")

    print(f"\n  => v3 wins in {new_v3_wins}/{new_total} new coin/period combos")

    # ════════════════════════════════════════════════
    # TEST 4: Parameter Sensitivity
    # ════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  [TEST 4] PARAMETER SENSITIVITY")
    print("  Varying ADX threshold, StochRSI exit, ATR trailing multiplier")
    print("  If results swing wildly = overfitted to specific parameter values")
    print("=" * 100)

    # Test on BTC + ETH + SOL combined
    test_coins = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    adx_vals = [20, 25, 30, 35]
    srsi_exit_vals = [90, 93, 95, 97]
    trail_vals = [1.5, 2.0, 2.5, 3.0]

    # Baseline (default params)
    base_returns = []
    for tk in test_coins:
        if tk not in raw: continue
        df = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
        if len(df) < 30: continue
        base_returns.append(backtest_v3(df) * 100)
    baseline = np.mean(base_returns)

    print(f"\n  Baseline (ADX=25, SRSI_exit=95, Trail=2.0x): avg return = {baseline:+.1f}%")

    # Vary ADX
    print(f"\n  --- ADX Threshold ---")
    print(f"  {'ADX':>6}  {'Avg Return':>12}  {'vs Baseline':>12}  Stable?")
    for adx_t in adx_vals:
        rets = []
        for tk in test_coins:
            if tk not in raw: continue
            df = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
            if len(df) < 30: continue
            rets.append(backtest_v3(df, adx_thresh=adx_t) * 100)
        avg = np.mean(rets)
        diff = avg - baseline
        stable = 'OK' if abs(diff) < baseline * 0.3 else 'SENSITIVE'
        print(f"  {adx_t:>6}  {avg:>+11.1f}%  {diff:>+11.1f}%  {stable}")

    # Vary StochRSI exit
    print(f"\n  --- StochRSI Bull Exit ---")
    print(f"  {'SRSI':>6}  {'Avg Return':>12}  {'vs Baseline':>12}  Stable?")
    for srsi_e in srsi_exit_vals:
        rets = []
        for tk in test_coins:
            if tk not in raw: continue
            df = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
            if len(df) < 30: continue
            rets.append(backtest_v3(df, srsi_exit_bull=srsi_e) * 100)
        avg = np.mean(rets)
        diff = avg - baseline
        stable = 'OK' if abs(diff) < baseline * 0.3 else 'SENSITIVE'
        print(f"  {srsi_e:>6}  {avg:>+11.1f}%  {diff:>+11.1f}%  {stable}")

    # Vary Trailing ATR mult
    print(f"\n  --- Trailing ATR Multiplier ---")
    print(f"  {'Trail':>6}  {'Avg Return':>12}  {'vs Baseline':>12}  Stable?")
    for tr in trail_vals:
        rets = []
        for tk in test_coins:
            if tk not in raw: continue
            df = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
            if len(df) < 30: continue
            rets.append(backtest_v3(df, trail_atr_mult=tr) * 100)
        avg = np.mean(rets)
        diff = avg - baseline
        stable = 'OK' if abs(diff) < baseline * 0.3 else 'SENSITIVE'
        print(f"  {tr:>6.1f}  {avg:>+11.1f}%  {diff:>+11.1f}%  {stable}")

    # ════════════════════════════════════════════════
    # VERDICT
    # ════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  OVERFITTING VERDICT")
    print("=" * 100)

    wf_degrade = len([r for r in wf_results if r['degrades'] == 'YES'])
    print(f"  Walk-Forward degradation:    {wf_degrade}/{len(wf_results)} coins show significant degradation")
    print(f"  Out-of-Sample (2025):        v3 wins {oos_v3_wins}/{oos_total} coins")
    print(f"  New Coins:                   v3 wins {new_v3_wins}/{new_total} combos")
    print(f"  Parameter Sensitivity:       See tables above")

    print("\n  INTERPRETATION:")
    if oos_v3_wins >= oos_total * 0.5 and new_v3_wins >= new_total * 0.5:
        print("  => v3 strategy shows REASONABLE generalization.")
        print("     It outperforms v2 on unseen data and new coins in most cases.")
        print("     Some overfitting is likely present but the core logic (ADX regime")
        print("     detection + trailing stops) appears to be a genuine alpha source.")
    else:
        print("  => WARNING: v3 may be significantly overfitted.")
        print("     Performance degrades substantially on unseen data.")
        print("     Consider simplifying the strategy or using fewer parameters.")

    print("\n  HONEST RISK FACTORS:")
    print("  1. v3 was designed AFTER seeing 2022-2024 exit patterns -> inherent bias")
    print("  2. ADX=25, StochRSI=95 thresholds were picked from historical analysis")
    print("  3. Crypto markets change character every cycle (macro, regulation, etc)")
    print("  4. Past performance does NOT guarantee future results")
    print("  5. Real trading has slippage, fees, and emotional factors not modeled")

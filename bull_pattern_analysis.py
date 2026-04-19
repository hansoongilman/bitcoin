"""
bull_pattern_analysis.py
상승장에서 전략이 왜 수익을 놓치는지 분석
- 상승 구간에서 청산 원인 분석
- 청산 후 가격이 얼마나 더 올라갔는지 확인
- 상승장 공통 지표 패턴 파악
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings

warnings.filterwarnings('ignore')

def add_indicators(df):
    df = df.copy()
    df['SMA30'] = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    macd_obj = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD'] = macd_obj.macd()
    df['MACD_Sig'] = macd_obj.macd_signal()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K'] = stoch.stoch()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Pct'] = bb.bollinger_pband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    # ADX
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()
    df.dropna(inplace=True)
    return df

def check_entry(row):
    return (row['Close'] > row['SMA30'] and
            row['MACD'] > row['MACD_Sig'] and
            row['StochRSI_K'] < 80 and
            row['BB_Pct'] > 0.4 and
            row['Vol_Ratio'] > 1.0)

def detailed_exit_check(row):
    """Which exit condition triggered?"""
    reasons = []
    if row['Close'] < row['SMA30']:
        reasons.append('SMA30_BREAK')
    if row['MACD'] < row['MACD_Sig']:
        reasons.append('MACD_BEAR')
    if row['StochRSI_K'] > 85:
        reasons.append('STOCHRSI_HOT')
    if row['Close'] < row['BB_Lower']:
        reasons.append('BB_BREAKDOWN')
    return reasons

def backtest_with_analysis(df, initial=10000.0, atr_mult=1.5):
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []

    for ts, row in df.iterrows():
        if pos > 0:
            dynamic_sl_pct = np.clip(atr_mult * float(row['ATR']) / ep, 0.03, 0.07)
            sl_price = ep * (1 - dynamic_sl_pct)

            if float(row['Low']) <= sl_price:
                # After exit, how much more did price go up?
                future_idx = df.index.get_loc(ts)
                future_prices = df['Close'].iloc[future_idx:future_idx+20]
                future_max = float(future_prices.max()) if len(future_prices) > 0 else float(row['Close'])
                missed_pct = (future_max / float(row['Close']) - 1) * 100

                cash += pos * sl_price
                trades.append({
                    'type': 'SL', 'profit': (sl_price / ep) - 1,
                    'exit_date': ts, 'exit_reasons': ['STOP_LOSS'],
                    'adx': float(row['ADX']), 'di_plus': float(row['DI_Plus']),
                    'di_minus': float(row['DI_Minus']),
                    'stochrsi': float(row['StochRSI_K']),
                    'missed_gain_20d': missed_pct
                })
                pos = 0.0; ep = 0.0
            else:
                reasons = detailed_exit_check(row)
                if reasons:
                    future_idx = df.index.get_loc(ts)
                    future_prices = df['Close'].iloc[future_idx:future_idx+20]
                    future_max = float(future_prices.max()) if len(future_prices) > 0 else float(row['Close'])
                    missed_pct = (future_max / float(row['Close']) - 1) * 100

                    cash += pos * float(row['Close'])
                    trades.append({
                        'type': 'EXIT', 'profit': (float(row['Close']) / ep) - 1,
                        'exit_date': ts, 'exit_reasons': reasons,
                        'adx': float(row['ADX']), 'di_plus': float(row['DI_Plus']),
                        'di_minus': float(row['DI_Minus']),
                        'stochrsi': float(row['StochRSI_K']),
                        'missed_gain_20d': missed_pct
                    })
                    pos = 0.0; ep = 0.0

        if pos == 0 and check_entry(row):
            ep = float(row['Close'])
            pos = cash / ep
            cash = 0.0

        equity.append(cash + pos * float(row['Close']))

    df = df.copy()
    df['Equity'] = equity
    return df, trades

if __name__ == '__main__':
    tickers = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'BTC-USD']

    all_trades = []
    for tk in tickers:
        print(f"[{tk}] downloading...")
        df = yf.download(tk, start='2021-06-01', end='2025-01-01', interval='1d', progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df_ind = add_indicators(df)
        _, trades = backtest_with_analysis(df_ind)
        for t in trades:
            t['ticker'] = tk
        all_trades.extend(trades)

    tdf = pd.DataFrame(all_trades)

    # 1. Exit reason frequency
    print("\n" + "=" * 80)
    print("  [1] EXIT REASON FREQUENCY (how often each condition triggers)")
    print("=" * 80)
    reason_counts = {}
    for reasons in tdf['exit_reasons']:
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    total = len(tdf)
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<20}: {cnt:>4} ({cnt/total*100:.1f}%)")

    # 2. Missed gains analysis - trades where price kept going up after exit
    print("\n" + "=" * 80)
    print("  [2] MISSED GAINS: Avg price increase 20d after exit")
    print("=" * 80)
    profitable_exits = tdf[tdf['profit'] > 0]
    losing_exits = tdf[tdf['profit'] <= 0]
    print(f"  Profitable trades that exited too early (20d missed avg): {profitable_exits['missed_gain_20d'].mean():.1f}%")
    print(f"  Losing trades (20d after exit avg):                       {losing_exits['missed_gain_20d'].mean():.1f}%")

    # By exit reason
    print("\n  By exit reason:")
    for reason in ['MACD_BEAR', 'STOCHRSI_HOT', 'SMA30_BREAK', 'BB_BREAKDOWN', 'STOP_LOSS']:
        subset = tdf[tdf['exit_reasons'].apply(lambda x: reason in x)]
        if len(subset) == 0:
            continue
        avg_missed = subset['missed_gain_20d'].mean()
        avg_profit = subset['profit'].mean() * 100
        high_adx = subset[subset['adx'] > 25]
        high_adx_missed = high_adx['missed_gain_20d'].mean() if len(high_adx) > 0 else 0
        print(f"  {reason:<20}: trades={len(subset):>3}  avg_trade_profit={avg_profit:>+6.1f}%  "
              f"missed_20d={avg_missed:>+5.1f}%  "
              f"missed_when_ADX>25={high_adx_missed:>+5.1f}%")

    # 3. ADX analysis during exits
    print("\n" + "=" * 80)
    print("  [3] ADX at exit: strong trend (ADX>25) vs weak trend")
    print("=" * 80)
    strong = tdf[tdf['adx'] > 25]
    weak = tdf[tdf['adx'] <= 25]
    print(f"  Strong trend exits (ADX>25): {len(strong)} trades, avg missed 20d: {strong['missed_gain_20d'].mean():.1f}%")
    print(f"  Weak trend exits   (ADX<=25): {len(weak)} trades, avg missed 20d:  {weak['missed_gain_20d'].mean():.1f}%")

    # 4. StochRSI at exit during strong trends
    print("\n" + "=" * 80)
    print("  [4] StochRSI exits during STRONG TRENDS (ADX>25)")
    print("=" * 80)
    strong_stoch = strong[strong['exit_reasons'].apply(lambda x: 'STOCHRSI_HOT' in x)]
    if len(strong_stoch) > 0:
        print(f"  StochRSI>85 exits during strong trends: {len(strong_stoch)}")
        print(f"  Avg StochRSI at exit: {strong_stoch['stochrsi'].mean():.1f}")
        print(f"  Avg missed gain 20d: {strong_stoch['missed_gain_20d'].mean():.1f}%")
        print(f"  Avg trade profit at exit: {strong_stoch['profit'].mean()*100:.1f}%")

    # 5. MACD bear cross during strong trends
    print("\n" + "=" * 80)
    print("  [5] MACD BEAR exits during STRONG TRENDS (ADX>25)")
    print("=" * 80)
    strong_macd = strong[strong['exit_reasons'].apply(lambda x: 'MACD_BEAR' in x)]
    if len(strong_macd) > 0:
        print(f"  MACD bear exits during strong trends: {len(strong_macd)}")
        print(f"  Avg missed gain 20d: {strong_macd['missed_gain_20d'].mean():.1f}%")
        print(f"  Avg trade profit at exit: {strong_macd['profit'].mean()*100:.1f}%")

    # 6. DI+/DI- analysis
    print("\n" + "=" * 80)
    print("  [6] DI+/DI- at exit (bullish trend = DI+ > DI-)")
    print("=" * 80)
    bullish_exits = tdf[(tdf['di_plus'] > tdf['di_minus']) & (tdf['adx'] > 25)]
    print(f"  Exits during confirmed bullish trend (DI+>DI- & ADX>25): {len(bullish_exits)}")
    if len(bullish_exits) > 0:
        print(f"  Avg missed gain 20d: {bullish_exits['missed_gain_20d'].mean():.1f}%")
        print(f"  Avg trade profit: {bullish_exits['profit'].mean()*100:.1f}%")
        print(f"  => These are the trades where we SHOULD have held longer!")

    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print("  The analysis will show which exit conditions cause premature exits")
    print("  during strong uptrends, and by how much we're missing out.")

"""
multi_coin_backtest.py
여러 알트코인에 대해 v2.0 개선된 전략을 백테스트
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 상수 및 색상 테마
BG_DARK   = '#020818'
BG_PANEL  = '#061025'
BG_CHART  = '#0a1830'
GRID_CLR  = '#0e2244'
TEXT_CLR  = '#93c5fd'
TITLE_CLR = '#dbeafe'

INITIAL_CAPITAL = 10000.0

def add_all_indicators(df):
    df = df.copy()
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    
    macd_obj = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']    = macd_obj.macd()
    df['MACD_Sig']= macd_obj.macd_signal()
    
    stoch_rsi = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K']  = stoch_rsi.stoch()
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Pct']   = bb.bollinger_pband()
    
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    
    df.dropna(inplace=True)
    return df

def check_entry_v2(row):
    return (row['Close'] > row['SMA30'] and
            row['MACD'] > row['MACD_Sig'] and
            row['StochRSI_K'] < 80 and
            row['BB_Pct'] > 0.4 and
            row['Vol_Ratio'] > 1.0)

def check_exit_v2(row):
    return (row['Close'] < row['SMA30'] or
            row['MACD'] < row['MACD_Sig'] or
            row['StochRSI_K'] > 85 or
            row['Close'] < row['BB_Lower'])

def backtest_v2(df, initial=10000.0, atr_mult=1.5):
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []

    for ts, row in df.iterrows():
        if pos > 0:
            dynamic_sl_pct = np.clip(atr_mult * float(row['ATR']) / ep, 0.03, 0.07)
            sl_price = ep * (1 - dynamic_sl_pct)

            if float(row['Low']) <= sl_price:
                cash += pos * sl_price
                trades.append({'type': 'SL', 'profit': (sl_price / ep) - 1})
                pos = 0.0; ep = 0.0
            elif check_exit_v2(row):
                cash += pos * float(row['Close'])
                trades.append({'type': 'EXIT', 'profit': (float(row['Close']) / ep) - 1})
                pos = 0.0; ep = 0.0

        if pos == 0 and check_entry_v2(row):
            ep = float(row['Close'])
            pos = cash / ep
            cash = 0.0

        equity.append(cash + pos * float(row['Close']))

    df = df.copy()
    df['Equity'] = equity
    if not equity:
        return df, [], 0, 0, -99, 0, 0
    cum_r = (df['Equity'].iloc[-1] / initial) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    dr  = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    wr = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    return df, trades, cum_r, mdd, sharpe, wr, bh

if __name__ == '__main__':
    tickers = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']
    print("데이터 수신 중 (2022-01-01 ~ 2025-01-01)...")
    
    results = []
    df_dict = {}
    
    for ticker in tickers:
        print(f"[{ticker}] 다운로드 및 계산 중...")
        df = yf.download(ticker, start='2022-01-01', end='2025-01-01', interval='1d', progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df_ind = add_all_indicators(df)
        if df_ind.empty:
            continue
            
        df_res, trades, cum_r, mdd, sh, wr, bh = backtest_v2(df_ind)
        df_dict[ticker] = df_res
        
        results.append({
            'ticker': ticker,
            'cr': cum_r,
            'mdd': mdd,
            'sh': sh,
            'wr': wr,
            'trades': len(trades),
            'bh': bh
        })

    print(f"\n{'코인':<10} {'전략수익':>10} {'단순보유':>10} {'MDD':>10} {'승률':>8} {'거래수':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['ticker']:<10} {r['cr']*100:>9.1f}% {r['bh']*100:>9.1f}% {r['mdd']*100:>9.1f}% {r['wr']*100:>7.1f}% {r['trades']:>6}")

    # 차트 그리기
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DARK)
    ax.set_facecolor(BG_CHART)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.grid(True, alpha=0.18, color=GRID_CLR)
    
    colors = ['#3b82f6', '#22c55e', '#ef4444', '#fbbf24', '#a78bfa']
    
    for i, r in enumerate(results):
        ticker = r['ticker']
        df_res = df_dict[ticker]
        ax.plot(df_res.index, df_res['Equity'], lw=1.5, label=f"{ticker} ({r['cr']*100:+.1f}%)", color=colors[i%len(colors)])
        
    ax.set_title('주요 알트코인 전략 v2.0 백테스트 (2022~2025)', color=TITLE_CLR, fontsize=16, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    leg = ax.legend(loc='upper left', fontsize=11, facecolor=BG_PANEL, edgecolor=GRID_CLR)
    for t in leg.get_texts():
        t.set_color(TITLE_CLR)
        
    save_path = 'c:/auto_bitcoin/multi_coin_backtest.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=BG_DARK)
    print(f"\n차트 저장 완료: {save_path}")

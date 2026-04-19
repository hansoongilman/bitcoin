"""
v4_short_strategy_backtest.py
실전 환경 반영(거래 수수료/슬리피지) 및 하락장 숏(Short) 매커니즘 추가
"""

import ccxt
import time
import requests
import urllib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

urllib3.disable_warnings()

# SSL 우회 패치 (망 환경 문제 해결)
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

# 상수 및 설정
FEE_RATE = 0.0015  # 1회 거래당 0.15% 수수료(슬리피지 포함) 적용
INITIAL_CAPITAL = 10000.0

BG_DARK   = '#020818'
BG_CHART  = '#0a1830'
TEXT_CLR  = '#93c5fd'
TITLE_CLR = '#dbeafe'
GRID_CLR  = '#0e2244'

def add_indicators(df):
    df = df.copy()
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    df['EMA10']   = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    
    macd_obj      = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']    = macd_obj.macd()
    df['MACD_Sig']= macd_obj.macd_signal()
    
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K'] = stoch.stoch()
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Pct']   = bb.bollinger_pband()
    
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    df['Vol_MA20']  = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX']      = adx.adx()
    df['DI_Plus']  = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()
    
    df.dropna(inplace=True)
    return df

def is_bull_regime(row):
    """ADX > 25 AND DI+ > DI- = confirmed bullish trend"""
    return float(row['ADX']) > 25 and float(row['DI_Plus']) > float(row['DI_Minus'])

def is_bear_regime(row):
    """ADX > 25 AND DI- > DI+ = confirmed bearish trend"""
    return float(row['ADX']) > 25 and float(row['DI_Minus']) > float(row['DI_Plus'])

def backtest_v4(df, initial=10000.0, atr_trail_mult=2.0, use_short=True):
    cash = initial
    pos_type = None  # None, 'LONG', 'SHORT'
    ep = 0.0
    equity = []
    trades = []
    
    trailing_high = 0.0
    trailing_low = 999999999.0
    
    for ts, row in df.iterrows():
        price = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        atr = float(row['ATR'])
        
        # Calculate current equity based on position
        if pos_type == 'LONG':
            current_equity = cash * (price / ep) * (1 - FEE_RATE)
        elif pos_type == 'SHORT':
            # Short margin profit
            profit_pct = (ep - price) / ep
            current_value = cash * (1 + profit_pct) * (1 - FEE_RATE)
            current_equity = current_value
        else:
            current_equity = cash
        
        # ── EXIT LOGIC ──
        if pos_type == 'LONG':
            if high > trailing_high:
                trailing_high = high
            
            # Trailing stop
            trail_stop = trailing_high - atr_trail_mult * atr
            if price < trail_stop or row['StochRSI_K'] > 95 or (price < float(row['EMA10']) and row['MACD'] < row['MACD_Sig']):
                # Sell
                exit_price = max(trail_stop, low) if price < trail_stop else price
                cash = cash * (exit_price / ep) * (1 - FEE_RATE) # deduct exit fee
                trades.append({'type': 'LONG_EXIT', 'profit': (exit_price / ep) - 1 - (FEE_RATE*2)})
                pos_type = None
                
        elif pos_type == 'SHORT':
            if low < trailing_low:
                trailing_low = low
                
            # Trailing stop for short
            trail_stop = trailing_low + atr_trail_mult * atr
            if price > trail_stop or row['StochRSI_K'] < 5 or (price > float(row['EMA10']) and row['MACD'] > row['MACD_Sig']):
                # Cover
                exit_price = min(trail_stop, high) if price > trail_stop else price
                profit_pct = (ep - exit_price) / ep
                cash = cash * (1 + profit_pct) * (1 - FEE_RATE) # deduct exit fee
                trades.append({'type': 'SHORT_EXIT', 'profit': profit_pct - (FEE_RATE*2)})
                pos_type = None

        # ── ENTRY LOGIC ──
        if pos_type is None:
            # Check LONG
            long_trend = price > row['SMA30']
            long_macd = row['MACD'] > row['MACD_Sig']
            long_stoch = row['StochRSI_K'] < 80
            if is_bull_regime(row): long_stoch = row['StochRSI_K'] < 90
            bb_ok = row['BB_Pct'] > 0.4
            vol_ok = row['Vol_Ratio'] > 1.0 if not is_bull_regime(row) else row['Vol_Ratio'] > 0.8
            
            if long_trend and long_macd and long_stoch and bb_ok and vol_ok:
                pos_type = 'LONG'
                ep = price
                cash = cash * (1 - FEE_RATE) # deduct entry fee
                trailing_high = high

            # Check SHORT
            elif use_short:
                short_trend = price < row['SMA30']
                short_macd = row['MACD'] < row['MACD_Sig']
                short_stoch = row['StochRSI_K'] > 20
                if is_bear_regime(row): short_stoch = row['StochRSI_K'] > 10
                bb_short_ok = row['BB_Pct'] < 0.6
                
                if short_trend and short_macd and short_stoch and bb_short_ok and vol_ok:
                    pos_type = 'SHORT'
                    ep = price
                    cash = cash * (1 - FEE_RATE) # deduct entry fee
                    trailing_low = low

        equity.append(current_equity)

    df = df.copy()
    df['Equity'] = equity
    
    if len(equity) == 0:
        return df, [], 0, 0, -99
        
    cum_r = (df['Equity'].iloc[-1] / initial) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    dr = df['Equity'].pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    
    return df, trades, cum_r, mdd, sharpe

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    exchange = ccxt.kraken()
    print("데이터 다운로드 중 (수수료 및 양방향 숏 적용 v4)...")
    
    all_results = []
    
    for tk in tickers:
        print(f"  [{tk}] 다운로드 (Kraken)...")
        symbol = tk.replace('-USD', '') + '/USD'
        if symbol == 'DOGE/USD': symbol = 'DOGE/USD'
        since = exchange.parse8601('2021-06-01T00:00:00Z')
        end_time = exchange.parse8601('2025-01-02T00:00:00Z')
        all_ohlcv = []
        while since < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since)
                if not ohlcv: break
                if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                    ohlcv = [x for x in ohlcv if x[0] > all_ohlcv[-1][0]]
                    if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(1.5)
            except Exception as e:
                break
                
        if not all_ohlcv: continue
            
        df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df[(df.index >= '2021-06-01') & (df.index <= '2025-01-01')]
        
        if len(df) < 50: continue
        
        df_ind = add_indicators(df)
        
        # 1. 롱 전용 전략 (수수료 포함)
        df_v3, trades_v3, cr_v3, mdd_v3, sh_v3 = backtest_v4(df_ind, use_short=False)
        # 2. 롱+숏 스위칭 전략 (수수료 포함)
        df_v4, trades_v4, cr_v4, mdd_v4, sh_v4 = backtest_v4(df_ind, use_short=True)
        
        bh = (df_ind['Close'].iloc[-1] / df_ind['Close'].iloc[0]) - 1
        
        wr3 = len([t for t in trades_v3 if t['profit'] > 0]) / len(trades_v3) if trades_v3 else 0
        wr4 = len([t for t in trades_v4 if t['profit'] > 0]) / len(trades_v4) if trades_v4 else 0
        
        all_results.append({
            'ticker': tk,
            'bh': bh,
            'cr_v3': cr_v3, 'mdd_v3': mdd_v3, 'wr_v3': wr3, 'tr_v3': len(trades_v3),
            'cr_v4': cr_v4, 'mdd_v4': mdd_v4, 'wr_v4': wr4, 'tr_v4': len(trades_v4),
            'df_v3': df_v3, 'df_v4': df_v4
        })

    print(f"\n=========================================================================================")
    print(f"  v3(Long Only) vs v4(Long+Short) vs Buy&Hold (수수료 {FEE_RATE*100:.2f}% 반영완료)")
    print(f"=========================================================================================")
    print(f" {'코인':<10} | {'단순보유':>10} | {'v3(롱)':>10} {'MDD':>7} | {'v4(롱+숏)':>10} {'MDD':>7} {'차이':>8}")
    print(f"-" * 89)
    for r in all_results:
        diff = r['cr_v4'] - r['cr_v3']
        print(f" {r['ticker']:<10} | {r['bh']*100:>9.1f}% | {r['cr_v3']*100:>9.1f}% {r['mdd_v3']*100:>6.1f}% | {r['cr_v4']*100:>9.1f}% {r['mdd_v4']*100:>6.1f}% {diff*100:>+7.1f}%p")

    # 시각화 (v3 vs v4)
    fig, axes = plt.subplots(len(all_results), 1, figsize=(18, 5 * len(all_results)), facecolor=BG_DARK)
    if len(all_results) == 1: axes = [axes]
    
    for i, r in enumerate(all_results):
        ax = axes[i]
        ax.set_facecolor(BG_CHART)
        for sp in ax.spines.values(): sp.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.grid(True, alpha=0.15, color=GRID_CLR)
        
        df_v4 = r['df_v4']
        bh_curve = INITIAL_CAPITAL * (df_v4['Close'] / df_v4['Close'].iloc[0])
        ax.plot(df_v4.index, bh_curve, color='#fbbf24', alpha=0.5, linestyle=':', label=f"B&H ({r['bh']*100:+.1f}%)")
        ax.plot(r['df_v3'].index, r['df_v3']['Equity'], color='#a78bfa', alpha=0.8, lw=1.5, linestyle='--', label=f"v3 Long-Only ({r['cr_v3']*100:+.1f}%)")
        ax.plot(df_v4.index, df_v4['Equity'], color='#06b6d4', lw=2.5, label=f"v4 Long+Short ({r['cr_v4']*100:+.1f}%)")
        
        ax.axhline(INITIAL_CAPITAL, color='#334155', lw=1, linestyle='--')
        ax.set_title(f"{r['ticker']}  |  v4 롱/숏 하이브리드 백테스트", color=TITLE_CLR, fontsize=13, fontweight='bold')
        leg = ax.legend(fontsize=11, facecolor=BG_DARK, edgecolor=GRID_CLR, loc='upper left')
        for t in leg.get_texts(): t.set_color(TITLE_CLR)
            
    plt.tight_layout()
    save_path = 'c:/auto_bitcoin/v4_short_strategy_result.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=BG_DARK)
    print(f"\n차트 저장 완료: {save_path}")

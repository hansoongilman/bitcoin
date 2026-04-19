"""
improved_strategy_v3.py
전략 v3.0 — 상승장 수익률 극대화 개선

핵심 개선사항:
1. ADX 기반 이중 체제: ADX>25 & DI+>DI- -> 강세 모드 활성화
2. 강세 모드에서 청산 조건 대폭 완화:
   - StochRSI 임계값: 85 -> 95
   - MACD 역전 무시 (트레일링 스탑으로 대체)
   - EMA10 기반 트레일링 스탑 사용
3. ATR 트레일링 스탑: 최고점 추적 동적 손절 (2xATR)
4. 빠른 재진입: 쿨다운 없이 즉시 재진입 가능
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import ta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════
# Theme
# ══════════════════════════════════════════════════════════
BG_DARK   = '#020818'
BG_PANEL  = '#061025'
BG_CHART  = '#0a1830'
GRID_CLR  = '#0e2244'
TEXT_CLR  = '#93c5fd'
TITLE_CLR = '#dbeafe'
C_BLUE    = '#3b82f6'
C_CYAN    = '#06b6d4'
C_GREEN   = '#22c55e'
C_RED     = '#ef4444'
C_YELLOW  = '#fbbf24'
C_PURPLE  = '#a78bfa'
C_ORANGE  = '#fb923c'

INITIAL_CAPITAL = 10000.0
COIN_NAMES = {
    'BTC-USD':  'Bitcoin (BTC)',
    'ETH-USD':  'Ethereum (ETH)',
    'SOL-USD':  'Solana (SOL)',
    'XRP-USD':  'Ripple (XRP)',
    'ADA-USD':  'Cardano (ADA)',
    'DOGE-USD': 'Dogecoin (DOGE)',
}

PERIODS = {
    '2022 H1 (Bear)':      ('2022-01-01', '2022-06-30'),
    '2022 H2 (Bottom)':    ('2022-07-01', '2022-12-31'),
    '2023 H1 (Recovery)':  ('2023-01-01', '2023-06-30'),
    '2023 H2 (Sideways)':  ('2023-07-01', '2023-12-31'),
    '2024 H1 (ETF Rally)': ('2024-01-01', '2024-06-30'),
    '2024 H2 (Volatile)':  ('2024-07-01', '2024-12-31'),
    'Full (2022~2024)':    ('2022-01-01', '2024-12-31'),
}

# ══════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    df['EMA10']   = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    df['EMA20']   = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()

    macd_obj      = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']    = macd_obj.macd()
    df['MACD_Sig']= macd_obj.macd_signal()

    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K'] = stoch.stoch()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Lower'] = bb.bollinger_lband()
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

# ══════════════════════════════════════════════════════════
# v2.0 Strategy (original)
# ══════════════════════════════════════════════════════════
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
    dr = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    wr = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    return df, trades, cum_r, mdd, sharpe, wr, bh

# ══════════════════════════════════════════════════════════
# v3.0 Strategy (improved for bull markets)
# ══════════════════════════════════════════════════════════

def is_bull_regime(row):
    """ADX > 25 AND DI+ > DI- = confirmed bullish trend"""
    return float(row['ADX']) > 25 and float(row['DI_Plus']) > float(row['DI_Minus'])

def check_entry_v3(row):
    """
    진입 조건 v3.0:
    - 기본: v2와 동일
    - 추가: 강세 모드에서는 StochRSI 조건 완화 (< 90)
    """
    trend_up    = row['Close'] > row['SMA30']
    macd_bull   = row['MACD'] > row['MACD_Sig']
    bb_ok       = row['BB_Pct'] > 0.4
    vol_confirm = row['Vol_Ratio'] > 1.0

    if is_bull_regime(row):
        # 강세 모드: StochRSI 조건 완화
        stochrsi_ok = row['StochRSI_K'] < 90
        # 강세 모드에서는 거래량 조건도 약간 완화
        vol_confirm = row['Vol_Ratio'] > 0.8
    else:
        stochrsi_ok = row['StochRSI_K'] < 80

    return trend_up and macd_bull and stochrsi_ok and bb_ok and vol_confirm

def check_exit_v3_normal(row):
    """약세/횡보 모드 청산 조건 (v2와 동일)"""
    return (row['Close'] < row['SMA30'] or
            row['MACD'] < row['MACD_Sig'] or
            row['StochRSI_K'] > 85 or
            row['Close'] < row['BB_Lower'])

def check_exit_v3_bull(row, trailing_high, atr_trail_mult=2.0):
    """
    강세 모드 청산 조건 (대폭 완화):
    1. EMA10 이탈 (빠른 추세선만 체크)
    2. StochRSI > 95 (극단적 과매수만)
    3. 트레일링 스탑: 최고가 - 2*ATR 이하
    4. BB 하단 이탈은 유지
    """
    price = float(row['Close'])
    atr = float(row['ATR'])

    # Trailing stop: highest price - 2*ATR
    trailing_stop = trailing_high - atr_trail_mult * atr
    if price < trailing_stop:
        return 'TRAIL_STOP'

    # Extreme overbought only
    if row['StochRSI_K'] > 95:
        return 'EXTREME_OB'

    # EMA10 break (short-term trend break)
    if price < float(row['EMA10']) and row['MACD'] < row['MACD_Sig']:
        return 'EMA10_MACD_BREAK'

    # BB breakdown
    if price < float(row['BB_Lower']):
        return 'BB_BREAK'

    return None

def backtest_v3(df, initial=10000.0, atr_mult=1.5, trail_atr_mult=2.0):
    """
    v3.0 백테스트: ADX 이중 체제 + 트레일링 스탑
    """
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []
    trailing_high = 0.0
    in_bull_entry = False  # Did we enter during a bull regime?

    for ts, row in df.iterrows():
        price = float(row['Close'])
        high = float(row['High'])

        if pos > 0:
            # Update trailing high
            if high > trailing_high:
                trailing_high = high

            # Determine current regime
            bull_now = is_bull_regime(row)

            if bull_now or in_bull_entry:
                # ── 강세 모드: 완화된 청산 ──
                exit_signal = check_exit_v3_bull(row, trailing_high, trail_atr_mult)

                if exit_signal:
                    if exit_signal == 'TRAIL_STOP':
                        # Use the trailing stop price
                        exit_price = trailing_high - trail_atr_mult * float(row['ATR'])
                        exit_price = max(exit_price, float(row['Low']))  # Can't exit below day low
                    else:
                        exit_price = price

                    cash += pos * exit_price
                    trades.append({'type': exit_signal, 'profit': (exit_price / ep) - 1})
                    pos = 0.0; ep = 0.0; trailing_high = 0.0; in_bull_entry = False

                # Also check if regime flipped to bearish
                if pos > 0 and not bull_now and not is_bull_regime(row):
                    # Regime changed: switch to normal exit check
                    in_bull_entry = False
                    if check_exit_v3_normal(row):
                        cash += pos * price
                        trades.append({'type': 'REGIME_EXIT', 'profit': (price / ep) - 1})
                        pos = 0.0; ep = 0.0; trailing_high = 0.0

            else:
                # ── 일반 모드: 기존 손절 + 청산 ──
                dynamic_sl_pct = np.clip(atr_mult * float(row['ATR']) / ep, 0.03, 0.07)
                sl_price = ep * (1 - dynamic_sl_pct)

                if float(row['Low']) <= sl_price:
                    cash += pos * sl_price
                    trades.append({'type': 'SL', 'profit': (sl_price / ep) - 1})
                    pos = 0.0; ep = 0.0; trailing_high = 0.0
                elif check_exit_v3_normal(row):
                    cash += pos * price
                    trades.append({'type': 'EXIT', 'profit': (price / ep) - 1})
                    pos = 0.0; ep = 0.0; trailing_high = 0.0

        # ── Entry ──
        if pos == 0 and check_entry_v3(row):
            ep = price
            pos = cash / ep
            cash = 0.0
            trailing_high = high
            in_bull_entry = is_bull_regime(row)

        equity.append(cash + pos * price)

    df = df.copy()
    df['Equity'] = equity
    if not equity:
        return df, [], 0, 0, -99, 0, 0
    cum_r = (df['Equity'].iloc[-1] / initial) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    dr = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    wr = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    return df, trades, cum_r, mdd, sharpe, wr, bh


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']

    print("Downloading data (2021-06 ~ 2025-01)...")
    ind_data = {}
    for tk in tickers:
        print(f"  [{tk}]...")
        df = yf.download(tk, start='2021-06-01', end='2025-01-01', interval='1d', progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        ind_data[tk] = add_indicators(df)

    # ── Run both strategies ──
    all_results = []
    for tk in tickers:
        if tk not in ind_data:
            continue
        df_full = ind_data[tk]
        for pname, (start, end) in PERIODS.items():
            df_s = df_full.loc[start:end].copy()
            if len(df_s) < 10:
                continue
            df_v2, t2, cr2, mdd2, sh2, wr2, bh = backtest_v2(df_s)
            df_v3, t3, cr3, mdd3, sh3, wr3, _  = backtest_v3(df_s)
            all_results.append({
                'ticker': tk,
                'period': pname,
                'v2_return': cr2, 'v2_mdd': mdd2, 'v2_sharpe': sh2, 'v2_wr': wr2, 'v2_trades': len(t2),
                'v3_return': cr3, 'v3_mdd': mdd3, 'v3_sharpe': sh3, 'v3_wr': wr3, 'v3_trades': len(t3),
                'bh': bh,
                'df_v2': df_v2, 'df_v3': df_v3,
            })

    # ── Console output ──
    period_names = [p for p in PERIODS.keys() if p != 'Full (2022~2024)']

    print("\n" + "=" * 130)
    print("  v2 vs v3 Strategy Comparison (per coin, per period)")
    print("=" * 130)

    for tk in tickers:
        coin_res = [r for r in all_results if r['ticker'] == tk]
        if not coin_res:
            continue
        print(f"\n{'=' * 130}")
        print(f"  [*] {COIN_NAMES.get(tk, tk)}")
        print(f"{'=' * 130}")
        print(f"  {'Period':<25} {'v2':>8} {'v3':>8} {'Diff':>8}  {'B&H':>8} {'v2 MDD':>8} {'v3 MDD':>8} {'v2 Trds':>8} {'v3 Trds':>8}")
        print(f"  {'-' * 115}")
        for r in coin_res:
            diff = r['v3_return'] - r['v2_return']
            marker = ' OK' if diff >= 0 else ' NG'
            print(f"  {r['period']:<25} {r['v2_return']*100:>+7.1f}% {r['v3_return']*100:>+7.1f}% "
                  f"{diff*100:>+7.1f}%{marker} {r['bh']*100:>+7.1f}% "
                  f"{r['v2_mdd']*100:>7.1f}% {r['v3_mdd']*100:>7.1f}% "
                  f"{r['v2_trades']:>7} {r['v3_trades']:>7}")

    # ════════════════════════════════════════════════
    # Chart 1: Per-coin equity curves (Full period)
    # ════════════════════════════════════════════════
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(20, 5 * n), facecolor=BG_DARK)
    if n == 1:
        axes = [axes]

    for idx, tk in enumerate(tickers):
        ax = axes[idx]
        ax.set_facecolor(BG_CHART)
        for sp in ax.spines.values():
            sp.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.grid(True, alpha=0.18, color=GRID_CLR)

        full = [r for r in all_results if r['ticker'] == tk and r['period'] == 'Full (2022~2024)']
        if not full:
            continue
        r = full[0]

        # B&H curve
        bh_curve = INITIAL_CAPITAL * (r['df_v3']['Close'] / r['df_v3']['Close'].iloc[0])
        ax.plot(r['df_v3'].index, bh_curve, color=C_YELLOW, lw=1.2, alpha=0.5, linestyle=':', label=f"B&H ({r['bh']*100:+.1f}%)")
        # v2
        ax.plot(r['df_v2'].index, r['df_v2']['Equity'], color=C_ORANGE, lw=1.3, alpha=0.7, linestyle='--', label=f"v2.0 ({r['v2_return']*100:+.1f}%)")
        # v3
        ax.plot(r['df_v3'].index, r['df_v3']['Equity'], color=C_CYAN, lw=2.2, label=f"v3.0 ({r['v3_return']*100:+.1f}%)", zorder=3)

        ax.axhline(INITIAL_CAPITAL, color='#334155', lw=0.7, linestyle=':')
        ax.fill_between(r['df_v3'].index, r['df_v3']['Equity'], INITIAL_CAPITAL,
                        where=(r['df_v3']['Equity'] >= INITIAL_CAPITAL), alpha=0.08, color=C_GREEN)
        ax.fill_between(r['df_v3'].index, r['df_v3']['Equity'], INITIAL_CAPITAL,
                        where=(r['df_v3']['Equity'] < INITIAL_CAPITAL), alpha=0.08, color=C_RED)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        diff = r['v3_return'] - r['v2_return']
        tc = C_GREEN if diff >= 0 else C_RED
        ax.set_title(f"{COIN_NAMES.get(tk, tk)}  |  v2: {r['v2_return']*100:+.1f}% -> v3: {r['v3_return']*100:+.1f}%  "
                     f"(+{diff*100:.1f}%p)  |  B&H: {r['bh']*100:+.1f}%  |  MDD: {r['v3_mdd']*100:.1f}%",
                     color=tc, fontsize=12, fontweight='bold', pad=10)

        leg = ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=GRID_CLR, loc='upper left')
        for t in leg.get_texts():
            t.set_color(TITLE_CLR)

    plt.tight_layout()
    save1 = 'c:/auto_bitcoin/v3_equity_curves.png'
    plt.savefig(save1, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"\nEquity curves saved: {save1}")

    # ════════════════════════════════════════════════
    # Chart 2: Period bars (v2 vs v3 vs B&H)
    # ════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(n, 1, figsize=(22, 5 * n), facecolor=BG_DARK)
    if n == 1:
        axes2 = [axes2]

    for idx, tk in enumerate(tickers):
        ax = axes2[idx]
        ax.set_facecolor(BG_CHART)
        for sp in ax.spines.values():
            sp.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.grid(True, alpha=0.18, color=GRID_CLR, axis='y')

        coin_res = [r for r in all_results if r['ticker'] == tk and r['period'] != 'Full (2022~2024)']
        if not coin_res:
            continue

        periods_here = [r['period'] for r in coin_res]
        v2_vals = [r['v2_return'] * 100 for r in coin_res]
        v3_vals = [r['v3_return'] * 100 for r in coin_res]
        bh_vals = [r['bh'] * 100 for r in coin_res]

        x = np.arange(len(periods_here))
        w = 0.25
        ax.bar(x - w, v2_vals, w, label='v2.0', color=C_ORANGE, alpha=0.75, edgecolor='#a16207', linewidth=0.5)
        bars_v3 = ax.bar(x, v3_vals, w, label='v3.0', color=C_CYAN, alpha=0.9, edgecolor='#0e7490', linewidth=0.5)
        ax.bar(x + w, bh_vals, w, label='B&H', color=C_YELLOW, alpha=0.5, edgecolor='#a16207', linewidth=0.5)

        for bar in bars_v3:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:+.1f}%', ha='center', va=va,
                    fontsize=8, color=TITLE_CLR, fontweight='bold')

        ax.axhline(0, color='#334155', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods_here, fontsize=9, color=TEXT_CLR)
        ax.set_title(f'{COIN_NAMES.get(tk, tk)} - v2 vs v3 vs B&H', color=TITLE_CLR, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Return (%)', color=TEXT_CLR)

        leg = ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=GRID_CLR, loc='upper left')
        for t in leg.get_texts():
            t.set_color(TITLE_CLR)

    plt.tight_layout()
    save2 = 'c:/auto_bitcoin/v3_period_bars.png'
    plt.savefig(save2, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"Period bars saved: {save2}")

    # ════════════════════════════════════════════════
    # Chart 3: Improvement heatmap (v3 - v2)
    # ════════════════════════════════════════════════
    heatmap_data = []
    for tk in tickers:
        row = []
        for pname in period_names:
            match = [r for r in all_results if r['ticker'] == tk and r['period'] == pname]
            if match:
                row.append((match[0]['v3_return'] - match[0]['v2_return']) * 100)
            else:
                row.append(0)
        heatmap_data.append(row)

    hm = np.array(heatmap_data)
    fig3, ax3 = plt.subplots(figsize=(16, 7), facecolor=BG_DARK)
    ax3.set_facecolor(BG_CHART)

    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(hm.min()), abs(hm.max()), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax3.imshow(hm, cmap='RdYlGn', aspect='auto', norm=norm)

    ax3.set_xticks(range(len(period_names)))
    ax3.set_xticklabels(period_names, fontsize=10, color=TEXT_CLR, rotation=15, ha='right')
    ax3.set_yticks(range(len(tickers)))
    ax3.set_yticklabels([COIN_NAMES.get(t, t) for t in tickers], fontsize=11, color=TEXT_CLR)

    for i in range(len(tickers)):
        for j in range(len(period_names)):
            val = hm[i, j]
            txt_color = 'black' if abs(val) < vmax * 0.6 else 'white'
            ax3.text(j, i, f'{val:+.1f}%p', ha='center', va='center',
                     fontsize=11, fontweight='bold', color=txt_color)

    ax3.set_title('v3.0 Improvement over v2.0 (v3 return - v2 return)',
                  color=TITLE_CLR, fontsize=14, fontweight='bold', pad=15)
    cb = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
    cb.set_label('Improvement (%p)', color=TEXT_CLR, fontsize=10)
    cb.ax.tick_params(colors=TEXT_CLR)

    for sp in ax3.spines.values():
        sp.set_color(GRID_CLR)
    ax3.tick_params(colors=TEXT_CLR)

    save3 = 'c:/auto_bitcoin/v3_improvement_heatmap.png'
    plt.savefig(save3, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"Heatmap saved: {save3}")

    # ════════════════════════════════════════════════
    # Summary table
    # ════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  FULL PERIOD SUMMARY (2022~2024)")
    print("=" * 90)
    print(f"  {'Coin':<18} {'v2':>8} {'v3':>8} {'Diff':>8} {'B&H':>8} {'v2 MDD':>8} {'v3 MDD':>8}")
    print(f"  {'-' * 80}")
    for tk in tickers:
        full = [r for r in all_results if r['ticker'] == tk and r['period'] == 'Full (2022~2024)']
        if not full:
            continue
        r = full[0]
        diff = r['v3_return'] - r['v2_return']
        print(f"  {COIN_NAMES.get(tk, tk):<18} {r['v2_return']*100:>+7.1f}% {r['v3_return']*100:>+7.1f}% "
              f"{diff*100:>+7.1f}% {r['bh']*100:>+7.1f}% {r['v2_mdd']*100:>7.1f}% {r['v3_mdd']*100:>7.1f}%")

    print("\n[DONE] Analysis complete!")

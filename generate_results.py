"""
generate_results.py
README용 결과 차트 생성 스크립트
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ta, warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BG_DARK='#020818'; BG_CHART='#0a1830'; BG_PANEL='#061025'; GRID='#0e2244'
TXT='#93c5fd'; BRIGHT='#dbeafe'
GREEN='#22c55e'; RED='#ef4444'; CYAN='#06b6d4'; YELLOW='#fbbf24'
ORANGE='#fb923c'; PURPLE='#a78bfa'; BLUE='#3b82f6'

INITIAL = 10000.0

def add_indicators(df):
    df = df.copy()
    df['EMA21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    m = ta.trend.MACD(df['Close'], 12, 26, 9)
    df['MACD'] = m.macd(); df['MACD_Sig'] = m.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    df.dropna(inplace=True)
    return df

def backtest_v4_equity(df, initial=10000.0, trail_mult=2.0):
    pos=0; cash=initial; ep=0; equity=[]; trailing_high=0
    for ts, row in df.iterrows():
        px=float(row['Close']); hi=float(row['High']); atr=float(row['ATR'])
        if pos>0:
            if hi>trailing_high: trailing_high=hi
            if px < trailing_high - trail_mult*atr:
                exit_px = max(trailing_high - trail_mult*atr, float(row['Low']))
                cash += pos*exit_px; pos=0; ep=0; trailing_high=0
        if pos==0:
            if px>float(row['EMA21']) and float(row['MACD'])>float(row['MACD_Sig']) and float(row['Vol_Ratio'])>1.0:
                ep=px; pos=cash/px; cash=0; trailing_high=hi
        equity.append(cash+pos*px)
    return equity

def style_ax(ax):
    ax.set_facecolor(BG_CHART)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.grid(True, alpha=0.18, color=GRID)

if __name__ == '__main__':
    coins = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD','ADA-USD','DOGE-USD',
             'AVAX-USD','LINK-USD','DOT-USD','MATIC-USD','NEAR-USD','UNI-USD']
    labels = {'BTC-USD':'BTC','ETH-USD':'ETH','SOL-USD':'SOL','XRP-USD':'XRP',
              'ADA-USD':'ADA','DOGE-USD':'DOGE','AVAX-USD':'AVAX','LINK-USD':'LINK',
              'DOT-USD':'DOT','MATIC-USD':'MATIC','NEAR-USD':'NEAR','UNI-USD':'UNI'}
    colors_map = [CYAN, BLUE, GREEN, YELLOW, PURPLE, ORANGE,
                  '#ff6b6b','#48dbfb','#ff9ff3','#feca57','#54a0ff','#5f27cd']

    print("Downloading data...")
    raw = {}
    for tk in coins:
        df = yf.download(tk, start='2021-06-01', end='2026-04-01', interval='1d', progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        raw[tk] = add_indicators(df)

    # ═══════════════════════════════════════
    # CHART 1: Equity curves (full period, top 6 coins)
    # ═══════════════════════════════════════
    print("Generating equity curves...")
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG_DARK)
    style_ax(ax)

    top6 = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD','DOGE-USD','LINK-USD']
    for i, tk in enumerate(top6):
        if tk not in raw: continue
        sub = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
        if len(sub) < 50: continue
        eq = backtest_v4_equity(sub)
        ret = (eq[-1]/INITIAL - 1)*100
        ax.plot(sub.index, eq, lw=2, label=f"{labels[tk]} ({ret:+.0f}%)", color=colors_map[i])

    ax.axhline(INITIAL, color='#334155', lw=0.8, linestyle=':')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    ax.set_title('v4.0 Strategy Equity Curves (2022-2024, $10,000 start)',
                 color=BRIGHT, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Portfolio Value ($)', color=TXT, fontsize=11)
    leg = ax.legend(fontsize=11, facecolor=BG_PANEL, edgecolor=GRID, loc='upper left', ncol=2)
    for t in leg.get_texts(): t.set_color(BRIGHT)
    plt.tight_layout()
    plt.savefig('c:/auto_bitcoin/results/equity_curves.png', dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════
    # CHART 2: Bar chart - v4 returns vs B&H for all 12 coins
    # ═══════════════════════════════════════
    print("Generating comparison bars...")
    v4_rets = []; bh_rets = []; coin_names = []
    for tk in coins:
        if tk not in raw: continue
        sub = raw[tk].loc['2022-01-01':'2024-12-31'].copy()
        if len(sub) < 50: continue
        eq = backtest_v4_equity(sub)
        v4_rets.append((eq[-1]/INITIAL - 1)*100)
        bh_rets.append((float(sub['Close'].iloc[-1])/float(sub['Close'].iloc[0]) - 1)*100)
        coin_names.append(labels[tk])

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG_DARK)
    style_ax(ax)
    x = np.arange(len(coin_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, v4_rets, w, label='v4 Strategy', color=CYAN, alpha=0.9, edgecolor='#0e7490', linewidth=0.5)
    bars2 = ax.bar(x + w/2, bh_rets, w, label='Buy & Hold', color=YELLOW, alpha=0.6, edgecolor='#a16207', linewidth=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + (3 if h>=0 else -12), f'{h:+.0f}%',
                ha='center', fontsize=8.5, color=BRIGHT, fontweight='bold')
    ax.axhline(0, color='#475569', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(coin_names, fontsize=11, color=TXT)
    ax.set_title('v4 Strategy vs Buy & Hold - 12 Coins (2022-2024)', color=BRIGHT, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Return (%)', color=TXT, fontsize=11)
    leg = ax.legend(fontsize=11, facecolor=BG_PANEL, edgecolor=GRID)
    for t in leg.get_texts(): t.set_color(BRIGHT)
    plt.tight_layout()
    plt.savefig('c:/auto_bitcoin/results/v4_vs_bh.png', dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════
    # CHART 3: 2025 OOS bar chart
    # ═══════════════════════════════════════
    print("Generating 2025 OOS chart...")
    v4_oos = []; bh_oos = []; oos_names = []
    for tk in coins:
        if tk not in raw: continue
        sub = raw[tk].loc['2025-01-01':'2025-12-31'].copy()
        if len(sub) < 30: continue
        eq = backtest_v4_equity(sub)
        v4_oos.append((eq[-1]/INITIAL - 1)*100)
        bh_oos.append((float(sub['Close'].iloc[-1])/float(sub['Close'].iloc[0]) - 1)*100)
        oos_names.append(labels[tk])

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG_DARK)
    style_ax(ax)
    x = np.arange(len(oos_names))
    bars1 = ax.bar(x - w/2, v4_oos, w, label='v4 Strategy', color=CYAN, alpha=0.9, edgecolor='#0e7490', linewidth=0.5)
    bars2 = ax.bar(x + w/2, bh_oos, w, label='Buy & Hold', color=RED, alpha=0.5, edgecolor='#991b1b', linewidth=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + (2 if h>=0 else -8), f'{h:+.0f}%',
                ha='center', fontsize=8.5, color=BRIGHT, fontweight='bold')
    ax.axhline(0, color='#475569', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(oos_names, fontsize=11, color=TXT)
    ax.set_title('2025 Out-of-Sample: v4 Strategy vs Buy & Hold (Unseen Data)', color=BRIGHT, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Return (%)', color=TXT, fontsize=11)
    leg = ax.legend(fontsize=11, facecolor=BG_PANEL, edgecolor=GRID)
    for t in leg.get_texts(): t.set_color(BRIGHT)
    plt.tight_layout()
    plt.savefig('c:/auto_bitcoin/results/oos_2025.png', dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════
    # CHART 4: Parameter stability
    # ═══════════════════════════════════════
    print("Generating parameter stability chart...")
    mults = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    avgs = []; stds = []
    for mult in mults:
        rets = []
        for tk in coins:
            if tk not in raw: continue
            sub = raw[tk].loc['2022-01-01':'2025-12-31'].copy()
            if len(sub) < 50: continue
            eq = backtest_v4_equity(sub, trail_mult=mult)
            rets.append((eq[-1]/INITIAL - 1)*100)
        avgs.append(np.mean(rets))
        stds.append(np.std(rets))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_DARK)
    style_ax(ax)
    ax.bar([str(m)+'x' for m in mults], avgs, color=[CYAN if m==2.0 else BLUE for m in mults],
           alpha=0.85, edgecolor='#0e7490', linewidth=0.5)
    for i, (m, a) in enumerate(zip(mults, avgs)):
        ax.text(i, a+5, f'{a:+.0f}%', ha='center', fontsize=10, color=BRIGHT, fontweight='bold')
    ax.axhline(0, color='#475569', lw=0.8)
    ax.set_title('Parameter Stability: ATR Trailing Multiplier (12 coins avg, 2022-2025)',
                 color=BRIGHT, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Average Return (%)', color=TXT, fontsize=11)
    ax.set_xlabel('ATR Multiplier', color=TXT, fontsize=11)
    # Highlight default
    ax.annotate('DEFAULT', xy=(3, avgs[3]), xytext=(3, avgs[3]+40),
                fontsize=10, color=CYAN, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.5))
    plt.tight_layout()
    plt.savefig('c:/auto_bitcoin/results/param_stability.png', dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════
    # CHART 5: Bear vs Bull market defense
    # ═══════════════════════════════════════
    print("Generating market regime chart...")
    periods = [('2022 H1\n(Bear)', '2022-01-01', '2022-06-30'),
               ('2022 H2\n(Bottom)', '2022-07-01', '2022-12-31'),
               ('2023 H1\n(Recovery)', '2023-01-01', '2023-06-30'),
               ('2023 H2\n(Rally)', '2023-07-01', '2023-12-31'),
               ('2024 H1\n(ETF)', '2024-01-01', '2024-06-30'),
               ('2024 H2\n(Volatile)', '2024-07-01', '2024-12-31'),
               ('2025 H1\n(Bear)', '2025-01-01', '2025-06-30'),
               ('2025 H2\n(Bear)', '2025-07-01', '2025-12-31')]
    p_v4 = []; p_bh = []; p_wins = []; p_names = []
    for pname, s, e in periods:
        v4a = []; bha = []; wins = 0; total = 0
        for tk in coins:
            if tk not in raw: continue
            sub = raw[tk].loc[s:e].copy()
            if len(sub) < 20: continue
            eq = backtest_v4_equity(sub)
            r = (eq[-1]/INITIAL-1)*100
            b = (float(sub['Close'].iloc[-1])/float(sub['Close'].iloc[0])-1)*100
            v4a.append(r); bha.append(b); total += 1
            if r > b: wins += 1
        p_v4.append(np.mean(v4a) if v4a else 0)
        p_bh.append(np.mean(bha) if bha else 0)
        p_wins.append(f'{wins}/{total}')
        p_names.append(pname)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG_DARK)
    style_ax(ax)
    x = np.arange(len(p_names))
    ax.bar(x - w/2, p_v4, w, label='v4 Strategy (avg)', color=CYAN, alpha=0.9, edgecolor='#0e7490')
    ax.bar(x + w/2, p_bh, w, label='Buy & Hold (avg)', color=YELLOW, alpha=0.6, edgecolor='#a16207')
    for i in range(len(p_names)):
        y_pos = max(p_v4[i], p_bh[i]) + 5
        ax.text(i, y_pos, p_wins[i], ha='center', fontsize=9, color=GREEN, fontweight='bold')
    ax.axhline(0, color='#475569', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(p_names, fontsize=9, color=TXT)
    ax.set_title('v4 vs B&H by Market Regime (12 coins avg) - Green = v4 win rate',
                 color=BRIGHT, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Average Return (%)', color=TXT, fontsize=11)
    leg = ax.legend(fontsize=11, facecolor=BG_PANEL, edgecolor=GRID)
    for t in leg.get_texts(): t.set_color(BRIGHT)
    plt.tight_layout()
    plt.savefig('c:/auto_bitcoin/results/market_regimes.png', dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    print("\nAll charts saved to c:/auto_bitcoin/results/")

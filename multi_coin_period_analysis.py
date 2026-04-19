"""
multi_coin_period_analysis.py
각 코인별 기간(연도+분기)별 전략 vs 단순보유 성과 비교 분석
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
# 색상 테마
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
COIN_COLORS = {
    'ETH-USD':  '#627eea',
    'SOL-USD':  '#00ffa3',
    'XRP-USD':  '#00aaff',
    'ADA-USD':  '#0033ad',
    'DOGE-USD': '#c3a634',
}
COIN_NAMES = {
    'ETH-USD':  '이더리움 (ETH)',
    'SOL-USD':  '솔라나 (SOL)',
    'XRP-USD':  '리플 (XRP)',
    'ADA-USD':  '에이다 (ADA)',
    'DOGE-USD': '도지코인 (DOGE)',
}

# ══════════════════════════════════════════════════════════
# 기간 정의 (연도별 + 특수 구간)
# ══════════════════════════════════════════════════════════
PERIODS = {
    '2022 상반기 (하락장)':   ('2022-01-01', '2022-06-30'),
    '2022 하반기 (바닥)':     ('2022-07-01', '2022-12-31'),
    '2023 상반기 (회복장)':   ('2023-01-01', '2023-06-30'),
    '2023 하반기 (횡보)':     ('2023-07-01', '2023-12-31'),
    '2024 상반기 (ETF 랠리)': ('2024-01-01', '2024-06-30'),
    '2024 하반기 (변동성)':   ('2024-07-01', '2024-12-31'),
    '전체 (2022~2024)':       ('2022-01-01', '2024-12-31'),
}

# ══════════════════════════════════════════════════════════
# 지표 계산
# ══════════════════════════════════════════════════════════
def add_all_indicators(df):
    df = df.copy()
    df['SMA30']      = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    macd_obj         = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']       = macd_obj.macd()
    df['MACD_Sig']   = macd_obj.macd_signal()
    stoch_rsi        = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K'] = stoch_rsi.stoch()
    bb               = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Lower']   = bb.bollinger_lband()
    df['BB_Pct']     = bb.bollinger_pband()
    df['ATR']        = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Vol_MA20']   = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio']  = df['Volume'] / df['Vol_MA20']
    df.dropna(inplace=True)
    return df

def check_entry(row):
    return (row['Close'] > row['SMA30'] and
            row['MACD'] > row['MACD_Sig'] and
            row['StochRSI_K'] < 80 and
            row['BB_Pct'] > 0.4 and
            row['Vol_Ratio'] > 1.0)

def check_exit(row):
    return (row['Close'] < row['SMA30'] or
            row['MACD'] < row['MACD_Sig'] or
            row['StochRSI_K'] > 85 or
            row['Close'] < row['BB_Lower'])

def backtest(df, initial=10000.0, atr_mult=1.5):
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
            elif check_exit(row):
                cash += pos * float(row['Close'])
                trades.append({'type': 'EXIT', 'profit': (float(row['Close']) / ep) - 1})
                pos = 0.0; ep = 0.0
        if pos == 0 and check_entry(row):
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
# 메인
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    tickers = ['ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']

    import ccxt
    import time
    import requests
    import urllib3
    urllib3.disable_warnings()
    
    # SSL 우회 패치
    old_request = requests.Session.request
    def new_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return old_request(self, method, url, **kwargs)
    requests.Session.request = new_request

    exchange = ccxt.kraken()
    print("전체 데이터 수신 중 (2021-06-01 ~ 2025-01-01)...")
    raw_data = {}
    ind_data = {}
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
                if not ohlcv:
                    break
                # 중복 데이터 방지
                if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                    ohlcv = [x for x in ohlcv if x[0] > all_ohlcv[-1][0]]
                    if not ohlcv:
                        break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(1.5)
            except Exception as e:
                print(f"Kraken fetch error: {e}")
                break
        
        if not all_ohlcv:
            continue
            
        df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        # 날짜 필터링
        df = df[(df.index >= '2021-06-01') & (df.index <= '2025-01-01')]
        
        raw_data[tk] = df
        ind_data[tk] = add_all_indicators(df)

    # ── 기간별 분석 ──
    all_results = []  # list of dicts
    for tk in tickers:
        if tk not in ind_data:
            continue
        df_full = ind_data[tk]
        for pname, (start, end) in PERIODS.items():
            df_s = df_full.loc[start:end].copy()
            if len(df_s) < 10:
                continue
            df_res, trades, cum_r, mdd, sh, wr, bh = backtest(df_s)
            all_results.append({
                'ticker': tk,
                'coin_name': COIN_NAMES[tk],
                'period': pname,
                'start': start,
                'end': end,
                'strategy_return': cum_r,
                'bh_return': bh,
                'mdd': mdd,
                'sharpe': sh,
                'win_rate': wr,
                'trades': len(trades),
                'df_result': df_res,
            })

    # ── 콘솔 출력 ──
    print("\n" + "=" * 110)
    print("  각 코인별 기간별 전략 vs 단순보유 성과 비교")
    print("=" * 110)

    for tk in tickers:
        coin_res = [r for r in all_results if r['ticker'] == tk]
        if not coin_res:
            continue
        print(f"\n{'=' * 110}")
        print(f"  [*] {COIN_NAMES[tk]}")
        print(f"{'=' * 110}")
        print(f"  {'기간':<25} {'전략 수익':>10} {'단순보유':>10} {'차이':>10} {'MDD':>10} {'샤프':>8} {'승률':>8} {'거래':>6}")
        print(f"  {'-' * 100}")
        for r in coin_res:
            diff = r['strategy_return'] - r['bh_return']
            marker = 'OK' if diff >= 0 else 'NG'
            print(f"  {r['period']:<25} {r['strategy_return']*100:>+9.1f}% {r['bh_return']*100:>+9.1f}% "
                  f"{diff*100:>+9.1f}% {marker} {r['mdd']*100:>9.1f}% {r['sharpe']:>7.2f} "
                  f"{r['win_rate']*100:>6.0f}% {r['trades']:>5}")

    # ══════════════════════════════════════════════════════════
    # 시각화 1: 코인별 기간별 바 차트 (전략 vs B&H)
    # ══════════════════════════════════════════════════════════
    period_names = [p for p in PERIODS.keys() if p != '전체 (2022~2024)']
    n_coins = len(tickers)

    fig, axes = plt.subplots(n_coins, 1, figsize=(20, 5 * n_coins), facecolor=BG_DARK)
    if n_coins == 1:
        axes = [axes]

    for idx, tk in enumerate(tickers):
        ax = axes[idx]
        ax.set_facecolor(BG_CHART)
        for sp in ax.spines.values():
            sp.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.grid(True, alpha=0.18, color=GRID_CLR, axis='y')

        coin_res = [r for r in all_results if r['ticker'] == tk and r['period'] != '전체 (2022~2024)']
        if not coin_res:
            continue

        periods_here = [r['period'] for r in coin_res]
        strat_vals = [r['strategy_return'] * 100 for r in coin_res]
        bh_vals    = [r['bh_return'] * 100 for r in coin_res]

        x = np.arange(len(periods_here))
        w = 0.35
        bars1 = ax.bar(x - w/2, strat_vals, w, label='전략 v2.0', color=C_CYAN, alpha=0.85, edgecolor='#0e7490', linewidth=0.5)
        bars2 = ax.bar(x + w/2, bh_vals,    w, label='단순보유 (B&H)', color=C_YELLOW, alpha=0.7, edgecolor='#a16207', linewidth=0.5)

        # 바 위에 숫자
        for bar in bars1:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:+.1f}%', ha='center', va=va,
                    fontsize=8.5, color=TITLE_CLR, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:+.1f}%', ha='center', va=va,
                    fontsize=8.5, color=C_YELLOW, fontweight='bold')

        ax.axhline(0, color='#334155', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods_here, fontsize=9, color=TEXT_CLR)
        ax.set_title(f'{COIN_NAMES[tk]}  —  기간별 수익률 비교', color=COIN_COLORS.get(tk, TITLE_CLR),
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('수익률 (%)', color=TEXT_CLR, fontsize=10)

        leg = ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=GRID_CLR, loc='upper left')
        for t in leg.get_texts():
            t.set_color(TITLE_CLR)

    plt.tight_layout()
    save1 = 'c:/auto_bitcoin/multi_coin_period_bars.png'
    plt.savefig(save1, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"\n바 차트 저장: {save1}")

    # ══════════════════════════════════════════════════════════
    # 시각화 2: 코인별 자산 곡선 (전체 기간)
    # ══════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(n_coins, 1, figsize=(20, 4.5 * n_coins), facecolor=BG_DARK)
    if n_coins == 1:
        axes2 = [axes2]

    for idx, tk in enumerate(tickers):
        ax = axes2[idx]
        ax.set_facecolor(BG_CHART)
        for sp in ax.spines.values():
            sp.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.grid(True, alpha=0.18, color=GRID_CLR)

        full_res = [r for r in all_results if r['ticker'] == tk and r['period'] == '전체 (2022~2024)']
        if not full_res:
            continue
        r = full_res[0]
        df_res = r['df_result']

        # 전략 자산 곡선
        ax.plot(df_res.index, df_res['Equity'], color=C_CYAN, lw=2.0,
                label=f"전략 v2.0 ({r['strategy_return']*100:+.1f}%)", zorder=3)
        # B&H 곡선
        bh_curve = INITIAL_CAPITAL * (df_res['Close'] / df_res['Close'].iloc[0])
        ax.plot(df_res.index, bh_curve, color=C_YELLOW, lw=1.5, alpha=0.7,
                label=f"단순보유 ({r['bh_return']*100:+.1f}%)", linestyle='--')

        ax.axhline(INITIAL_CAPITAL, color='#334155', lw=0.7, linestyle=':')
        ax.fill_between(df_res.index, df_res['Equity'], INITIAL_CAPITAL,
                        where=(df_res['Equity'] >= INITIAL_CAPITAL), alpha=0.1, color=C_GREEN)
        ax.fill_between(df_res.index, df_res['Equity'], INITIAL_CAPITAL,
                        where=(df_res['Equity'] < INITIAL_CAPITAL), alpha=0.1, color=C_RED)

        # 기간 구분선 (반기별)
        for pname, (s, e) in PERIODS.items():
            if pname == '전체 (2022~2024)':
                continue
            try:
                dt = pd.Timestamp(s)
                if df_res.index[0] <= dt <= df_res.index[-1]:
                    ax.axvline(dt, color='#334155', lw=0.6, linestyle=':', alpha=0.5)
                    ax.text(dt, ax.get_ylim()[1] * 0.98, pname[:8], fontsize=7, color=TEXT_CLR, alpha=0.5,
                            rotation=90, va='top', ha='right')
            except:
                pass

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        diff = r['strategy_return'] - r['bh_return']
        tc = C_GREEN if diff >= 0 else C_RED
        ax.set_title(f"{COIN_NAMES[tk]}  |  전략 {r['strategy_return']*100:+.1f}% vs 보유 {r['bh_return']*100:+.1f}%  |  "
                     f"MDD {r['mdd']*100:.1f}%  샤프 {r['sharpe']:.2f}  승률 {r['win_rate']*100:.0f}%",
                     color=tc, fontsize=12, fontweight='bold', pad=10)

        leg = ax.legend(fontsize=10, facecolor=BG_PANEL, edgecolor=GRID_CLR, loc='upper left')
        for t in leg.get_texts():
            t.set_color(TITLE_CLR)

    plt.tight_layout()
    save2 = 'c:/auto_bitcoin/multi_coin_equity_curves.png'
    plt.savefig(save2, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"자산곡선 저장: {save2}")

    # ══════════════════════════════════════════════════════════
    # 시각화 3: 히트맵 (코인 x 기간, 전략 - B&H 초과수익률)
    # ══════════════════════════════════════════════════════════
    heatmap_data = []
    for tk in tickers:
        row = []
        for pname in period_names:
            match = [r for r in all_results if r['ticker'] == tk and r['period'] == pname]
            if match:
                row.append((match[0]['strategy_return'] - match[0]['bh_return']) * 100)
            else:
                row.append(0)
        heatmap_data.append(row)

    hm = np.array(heatmap_data)
    fig3, ax3 = plt.subplots(figsize=(16, 6), facecolor=BG_DARK)
    ax3.set_facecolor(BG_CHART)

    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(hm.min()), abs(hm.max()), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax3.imshow(hm, cmap='RdYlGn', aspect='auto', norm=norm)

    ax3.set_xticks(range(len(period_names)))
    ax3.set_xticklabels(period_names, fontsize=10, color=TEXT_CLR, rotation=15, ha='right')
    ax3.set_yticks(range(len(tickers)))
    ax3.set_yticklabels([COIN_NAMES[t] for t in tickers], fontsize=11, color=TEXT_CLR)

    for i in range(len(tickers)):
        for j in range(len(period_names)):
            val = hm[i, j]
            txt_color = 'black' if abs(val) < vmax * 0.6 else 'white'
            ax3.text(j, i, f'{val:+.1f}%p', ha='center', va='center',
                     fontsize=11, fontweight='bold', color=txt_color)

    ax3.set_title('전략 v2.0 초과수익률 히트맵  (전략 수익 - 단순보유 수익)',
                  color=TITLE_CLR, fontsize=14, fontweight='bold', pad=15)
    cb = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
    cb.set_label('초과수익률 (%p)', color=TEXT_CLR, fontsize=10)
    cb.ax.tick_params(colors=TEXT_CLR)

    for sp in ax3.spines.values():
        sp.set_color(GRID_CLR)
    ax3.tick_params(colors=TEXT_CLR)

    save3 = 'c:/auto_bitcoin/multi_coin_heatmap.png'
    plt.savefig(save3, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
    print(f"히트맵 저장: {save3}")

    print("\n[OK] 모든 분석 완료!")

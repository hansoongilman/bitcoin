"""
param_tuning_backtest.py

목적:
  - 횡보/혼돈 구간에서 전략이 '가짜 신호(휩쏘)'로 손실을 반복하는 문제를
    파라미터 튜닝(Grid Search)으로 최적화
  - 튜닝 대상: RSI 임계값, SMA 기간, Stop-Loss %, MACD fast/slow/signal
  - 횡보 6개 구간 전체에서 평균 샤프 지수가 가장 높은 조합을 선택해 재검증
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ta
import itertools
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────
# 1. 공통 함수
# ─────────────────────────────────────────────────────

def fetch_full_data():
    print("데이터 수신 중...")
    df = yf.download('BTC-USD', start='2021-01-01', end='2025-01-01', interval='1d')
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def add_indicators(df, sma_w=50, rsi_w=14, macd_fast=12, macd_slow=26, macd_sig=9):
    df = df.copy()
    df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=sma_w).sma_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_w).rsi()
    macd_obj = ta.trend.MACD(df['Close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_sig)
    df['MACD_Line'] = macd_obj.macd()
    df['MACD_Sig']  = macd_obj.macd_signal()
    df.dropna(inplace=True)
    return df

def backtest(df, sl_pct=0.03, rsi_entry=70, rsi_exit=70, initial=10000.0):
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []
    for date, row in df.iterrows():
        if pos > 0:
            sl_p = ep * (1 - sl_pct)
            if row['Low'] <= sl_p:
                cash += pos * sl_p
                trades.append((sl_p / ep) - 1)
                pos = 0.0; ep = 0.0
            elif (row['Close'] < row['SMA'] or row['MACD_Line'] < row['MACD_Sig'] or row['RSI'] > rsi_exit):
                cash += pos * row['Close']
                trades.append((row['Close'] / ep) - 1)
                pos = 0.0; ep = 0.0
        if pos == 0 and (row['Close'] > row['SMA'] and row['MACD_Line'] > row['MACD_Sig'] and row['RSI'] < rsi_entry):
            ep = row['Close']; pos = cash / ep; cash = 0.0
        equity.append(cash + pos * row['Close'])
    df = df.copy()
    df['Equity'] = equity
    if len(equity) == 0:
        return df, 0, 0, -99, 0, 0
    cum_r = (df['Equity'].iloc[-1] / initial) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    dr = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    win_r = len([t for t in trades if t > 0]) / len(trades) if trades else 0
    return df, cum_r, mdd, sharpe, win_r, len(trades)

# ─────────────────────────────────────────────────────
# 2. 횡보 구간 정의
# ─────────────────────────────────────────────────────

SIDEWAYS = {
    '① 5월폭락後 삼각수렴 (21.05~07)': ('2021-05-19', '2021-07-20'),
    '② 엘살바도르 쇼크 (21.09~10)':   ('2021-09-07', '2021-10-05'),
    '③ LUNA붕괴 직전 반등 (22.03~05)': ('2022-03-28', '2022-05-08'),
    '④ 크립토윈터 지루한 횡보 (22.08~11)': ('2022-08-15', '2022-11-07'),
    '⑤ 상승인지 횡보인지 (23.04~06)': ('2023-04-01', '2023-06-15'),
    '⑥ ETF 눌림목 횡보 (24.05~08)':   ('2024-05-20', '2024-08-04'),
}

# ─────────────────────────────────────────────────────
# 3. Grid Search
# ─────────────────────────────────────────────────────

GRID = {
    'sma_w':      [30, 50, 100],
    'rsi_w':      [14],   # RSI 계산 기간은 14 고정
    'rsi_entry':  [60, 65, 70],
    'rsi_exit':   [70, 75, 80],
    'sl_pct':     [0.03, 0.05, 0.07],
    'macd_fast':  [8, 12],
    'macd_slow':  [21, 26],
    'macd_sig':   [7, 9],
}

df_all = fetch_full_data()

param_keys = list(GRID.keys())
param_combos = list(itertools.product(*[GRID[k] for k in param_keys]))
print(f"총 파라미터 조합 수: {len(param_combos)}개, 평균 샤프 최적화 진행 중...")

# Pre-slice all periods from full data
slices_raw = {}
for name, (s, e) in SIDEWAYS.items():
    slices_raw[name] = df_all.loc[s:e].copy()

best_score = -999
best_params = None

for combo in param_combos:
    params = dict(zip(param_keys, combo))
    # skip invalid MACD combos
    if params['macd_fast'] >= params['macd_slow']:
        continue

    sharpes = []
    for name, df_raw in slices_raw.items():
        try:
            df_ind = add_indicators(df_raw,
                                    sma_w=params['sma_w'],
                                    rsi_w=params['rsi_w'],
                                    macd_fast=params['macd_fast'],
                                    macd_slow=params['macd_slow'],
                                    macd_sig=params['macd_sig'])
            if len(df_ind) < 5:
                continue
            _, cr, mdd, sharpe, wr, n = backtest(df_ind,
                                                  sl_pct=params['sl_pct'],
                                                  rsi_entry=params['rsi_entry'],
                                                  rsi_exit=params['rsi_exit'])
            sharpes.append(sharpe)
        except:
            pass

    if not sharpes:
        continue
    avg_sharpe = np.mean(sharpes)
    if avg_sharpe > best_score:
        best_score = avg_sharpe
        best_params = params

print(f"\n최적 파라미터 (평균 샤프 {best_score:.3f}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────────────
# 4. 최적 파라미터로 재백테스트 & 기존 파라미터와 비교
# ─────────────────────────────────────────────────────

DEFAULT = dict(sma_w=50, rsi_w=14, rsi_entry=70, rsi_exit=70, sl_pct=0.03, macd_fast=12, macd_slow=26, macd_sig=9)
TUNED   = best_params

results_default = {}
results_tuned   = {}
summary_rows    = []

for name, df_raw in slices_raw.items():
    # Default
    df_d = add_indicators(df_raw, **{k: DEFAULT[k] for k in ['sma_w','rsi_w','macd_fast','macd_slow','macd_sig']})
    if len(df_d) == 0:
        continue
    df_dr, cr_d, mdd_d, sh_d, wr_d, n_d = backtest(df_d, sl_pct=DEFAULT['sl_pct'],
                                                     rsi_entry=DEFAULT['rsi_entry'], rsi_exit=DEFAULT['rsi_exit'])
    # Tuned
    df_t = add_indicators(df_raw, **{k: TUNED[k] for k in ['sma_w','rsi_w','macd_fast','macd_slow','macd_sig']})
    if len(df_t) == 0:
        continue
    df_tr, cr_t, mdd_t, sh_t, wr_t, n_t = backtest(df_t, sl_pct=TUNED['sl_pct'],
                                                     rsi_entry=TUNED['rsi_entry'], rsi_exit=TUNED['rsi_exit'])
    bh = (df_raw['Close'].iloc[-1] / df_raw['Close'].iloc[0]) - 1

    results_default[name] = df_dr
    results_tuned[name]   = df_tr
    summary_rows.append({
        'name': name,
        'default_r': cr_d, 'default_sh': sh_d, 'default_mdd': mdd_d, 'default_n': n_d,
        'tuned_r':   cr_t, 'tuned_sh':  sh_t,  'tuned_mdd':  mdd_t,  'tuned_n':   n_t,
        'bh': bh,
    })

# ─────────────────────────────────────────────────────
# 5. 출력
# ─────────────────────────────────────────────────────

print("\n" + "="*100)
print("                    기본(DEFAULT) vs 튜닝(TUNED) 비교")
print("="*100)
print(f"{'구간':<38} {'거래':>4} {'기본수익':>9} {'기본샤프':>9}   {'거래':>4} {'튜닝수익':>9} {'튜닝샤프':>9}  {'단순보유':>9}")
print("-"*100)
for r in summary_rows:
    n = r['name'].replace('\n', ' ')
    print(f"{n:<38} {r['default_n']:>4} {r['default_r']*100:>8.1f}% {r['default_sh']:>9.2f}   "
          f"{r['tuned_n']:>4} {r['tuned_r']*100:>8.1f}% {r['tuned_sh']:>9.2f}  {r['bh']*100:>8.1f}%")

# ─────────────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────────────

n_p = len(summary_rows)
fig, axes = plt.subplots(n_p + 1, 1, figsize=(16, 5 * n_p + 5))
param_str = f"SMA{TUNED['sma_w']}  RSI진입<{TUNED['rsi_entry']}  RSI청산>{TUNED['rsi_exit']}  SL={int(TUNED['sl_pct']*100)}%  MACD({TUNED['macd_fast']}/{TUNED['macd_slow']}/{TUNED['macd_sig']})"
fig.suptitle(f'파라미터 튜닝 후 재백테스트 (횡보 구간 집중)\n튜닝 파라미터: {param_str}', fontsize=13, fontweight='bold', y=0.99)

for i, (name, df_tr) in enumerate(results_tuned.items()):
    ax = axes[i]
    df_dr = results_default[name]
    r = summary_rows[i]
    raw_slice = slices_raw[name]
    if len(raw_slice) == 0 or len(df_tr) == 0:
        continue
    # Align B&H to the same date range as the tuned df (post indicator warm-up)
    raw_aligned = raw_slice.loc[df_tr.index[0]:df_tr.index[-1]]
    bh_curve = 10000 * (raw_aligned['Close'] / raw_aligned['Close'].iloc[0])

    ax.plot(df_tr.index, df_tr['Equity'], color='#2563EB', lw=2,   label=f'튜닝 전략 ({r["tuned_r"]*100:+.1f}%)')
    ax.plot(df_dr.index, df_dr['Equity'], color='#7C3AED', lw=1.5, label=f'기본 전략 ({r["default_r"]*100:+.1f}%)', linestyle='--', alpha=0.7)
    ax.plot(df_tr.index, bh_curve,        color='#F59E0B', lw=1.5, label=f'단순보유 ({r["bh"]*100:+.1f}%)',       linestyle=':', alpha=0.7)
    ax.axhline(10000, color='gray', lw=0.8, linestyle=':')

    ret_col = '#10B981' if r['tuned_r'] >= r['default_r'] else '#EF4444'
    diff = r['tuned_r'] - r['default_r']
    ax.set_title(f"{name.replace(chr(10), ' ')}   |   개선: {diff*100:+.1f}%p   |   튜닝 샤프 {r['tuned_sh']:.2f}", fontsize=11, fontweight='bold', color=ret_col)
    ax.set_ylabel('포트폴리오 ($)', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=9.5)
    ax.grid(True, alpha=0.25)

# ---- 하단 비교 테이블 ----
ax_tbl = axes[-1]
ax_tbl.axis('off')

col_labels = ['구간', '기본 거래수', '기본 수익률', '기본 샤프',
              '튜닝 거래수', '튜닝 수익률', '튜닝 샤프', '단순보유', '개선도']
table_data  = []
cell_colors = []
GREEN = '#d1fae5'; RED = '#fee2e2'; WHITE = '#f8fafc'

for r in summary_rows:
    diff = r['tuned_r'] - r['default_r']
    row = [
        r['name'].replace('\n', ' '),
        str(r['default_n']),
        f"{r['default_r']*100:+.1f}%",
        f"{r['default_sh']:.2f}",
        str(r['tuned_n']),
        f"{r['tuned_r']*100:+.1f}%",
        f"{r['tuned_sh']:.2f}",
        f"{r['bh']*100:+.1f}%",
        f"{diff*100:+.1f}%p",
    ]
    rc = [WHITE] * 9
    rc[2] = GREEN if r['default_r'] >= 0 else RED
    rc[5] = GREEN if r['tuned_r'] >= 0 else RED
    rc[8] = GREEN if diff >= 0 else RED
    table_data.append(row)
    cell_colors.append(rc)

tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                   cellColours=cell_colors, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#1e3a5f')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
ax_tbl.set_title('기본 파라미터 vs 튜닝 파라미터 성과 비교', fontsize=12, fontweight='bold', pad=8)

plt.tight_layout()
save_path = 'c:/auto_bitcoin/tuned_backtest.png'
plt.savefig(save_path, dpi=130, bbox_inches='tight')
print(f"\n차트 저장 완료: {save_path}")

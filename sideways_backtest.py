"""
sideways_backtest.py

핵심 목적:
  - 실제 BTC 역사에서 '방향성 없는 횡보/예측불가 구간'을 직접 잘라서
    퀀트 전략(MACD + RSI + SMA50 + 3% 손절)로 백테스트
  - 상승/하락 대세는 직접 판단 가능 → 전략이 '불확실한 중간값' 구간에서
    살아남을 수 있는지 검증하는 것이 목적

대표 BTC 횡보/혼돈 구간 (실제 차트 기반):
  1. 2021-05-19 ~ 2021-07-20 : 5월 대폭락 후 삼각수렴 횡보구간 ($30k~40k)
  2. 2021-09-07 ~ 2021-10-05 : El Salvador 쇼크 후 급락+횡보 ($42k~52k)
  3. 2022-03-28 ~ 2022-05-08 : LUNA 붕괴 직전 가짜 반등+혼조구간
  4. 2022-08-15 ~ 2022-11-07 : 반등인 듯 아닌 듯 지루한 횡보 ($18k~25k)
  5. 2023-04-01 ~ 2023-06-15 : 상승인 듯 횡보인 듯 ($27k~31k)
  6. 2024-05-20 ~ 2024-08-04 : ETF 눌림목+횡보구간 ($58k~72k)
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

# ==========================================================
# 1. 데이터 및 전략 함수 (quant_backtest.py 와 동일 모듈)
# ==========================================================

def fetch_data(symbol='BTC-USD', start=None, end=None):
    df = yf.download(symbol, start=start, end=end, interval='1d')
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def generate_signals(df):
    df = df.copy()
    df['SMA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def check_entry(row):
    return (row['Close'] > row['SMA50']
            and row['MACD_Line'] > row['MACD_Signal']
            and row['RSI'] < 70)

def check_exit(row):
    return (row['Close'] < row['SMA50']
            or row['MACD_Line'] < row['MACD_Signal']
            or row['RSI'] > 70)

def run_backtest(df, initial_capital=10000.0, stop_loss_pct=0.03):
    position = 0.0
    cash = initial_capital
    entry_price = 0.0
    equity_curve = []
    trades = []

    for date, row in df.iterrows():
        if position > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            if row['Low'] <= sl_price:
                cash += position * sl_price
                trades.append({'Date': date, 'Type': 'SL', 'Profit': (sl_price / entry_price) - 1})
                position = 0.0; entry_price = 0.0
            elif check_exit(row):
                cash += position * row['Close']
                trades.append({'Date': date, 'Type': 'EXIT', 'Profit': (row['Close'] / entry_price) - 1})
                position = 0.0; entry_price = 0.0

        if position == 0 and check_entry(row):
            entry_price = row['Close']
            position = cash / entry_price
            cash = 0.0

        equity_curve.append(cash + position * row['Close'])

    df = df.copy()
    df['Equity'] = equity_curve
    return df, trades

def metrics(df, trades):
    cum_return = (df['Equity'].iloc[-1] / df['Equity'].iloc[0]) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    win_rate = len([t for t in trades if t['Profit'] > 0]) / len(trades) if trades else 0
    dr = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else 0
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    return cum_return, mdd, win_rate, sharpe, bh, len(trades)

# ==========================================================
# 2. 횡보/혼돈 구간 정의
# ==========================================================

SIDEWAYS_PERIODS = {
    '① 5월 폭락 후 삼각수렴\n(21.05.19~07.20)': ('2021-05-19', '2021-07-20'),
    '② 엘살바도르 쇼크 횡보\n(21.09.07~10.05)': ('2021-09-07', '2021-10-05'),
    '③ LUNA붕괴 직전 가짜반등\n(22.03.28~05.08)': ('2022-03-28', '2022-05-08'),
    '④ 크립토윈터 지루한횡보\n(22.08.15~11.07)': ('2022-08-15', '2022-11-07'),
    '⑤ 상승인지 횡보인지\n(23.04.01~06.15)': ('2023-04-01', '2023-06-15'),
    '⑥ ETF 눌림목 횡보\n(24.05.20~08.04)': ('2024-05-20', '2024-08-04'),
}

# ==========================================================
# 3. 데이터 한 번에 받아서 warm-up 지표 문제 해결
# ==========================================================

print("데이터 수신 중 (전체)...")
# Fetch enough history for SMA50 warm-up before earliest period
df_all = fetch_data(symbol='BTC-USD', start='2021-01-01', end='2025-01-01')
df_all = generate_signals(df_all)

print("구간별 백테스트 실행 중...\n")

results = {}
summary_rows = []

for period_name, (start, end) in SIDEWAYS_PERIODS.items():
    df_slice = df_all.loc[start:end].copy()
    if len(df_slice) < 10:
        continue

    df_res, trades = run_backtest(df_slice, initial_capital=10000.0, stop_loss_pct=0.03)
    cum_r, mdd, win_rate, sharpe, bh_r, n = metrics(df_res, trades)
    results[period_name] = df_res
    summary_rows.append({
        '구간': period_name,
        '거래수': n,
        '승률': win_rate,
        '전략수익': cum_r,
        '단순보유': bh_r,
        'MDD': mdd,
        '샤프': sharpe,
        '시작': start,
        '종료': end,
    })

# ==========================================================
# 4. 결과 출력 (텍스트)
# ==========================================================
print(f"{'구간':<30} {'거래':>4} {'승률':>7} {'전략수익':>9} {'단순보유':>9} {'MDD':>8} {'샤프':>6}")
print("-" * 80)
for r in summary_rows:
    name_short = r['구간'].replace('\n', ' ')
    print(f"{name_short:<30} {r['거래수']:>4} {r['승률']*100:>6.1f}% {r['전략수익']*100:>8.1f}% {r['단순보유']*100:>8.1f}% {r['MDD']*100:>7.1f}% {r['샤프']:>6.2f}")

# ==========================================================
# 5. 시각화 (2가지 대시보드: 차트 + 성과표)
# ==========================================================

n_periods = len(results)
fig = plt.figure(figsize=(18, 5 * n_periods + 4))
fig.suptitle('BTC 횡보·혼돈 구간 집중 백테스트 결과', fontsize=18, fontweight='bold', y=0.98)

gs = fig.add_gridspec(n_periods + 1, 2, height_ratios=[1] * n_periods + [1.2], hspace=0.6, wspace=0.35)

colors_strategy = '#2563EB'   # blue
colors_bh       = '#F59E0B'   # amber
color_green     = '#10B981'
color_red       = '#EF4444'

for i, (period_name, df_r) in enumerate(results.items()):
    r = summary_rows[i]
    bh_curve = 10000 * (df_r['Close'] / df_r['Close'].iloc[0])

    ax = fig.add_subplot(gs[i, :])

    ax.plot(df_r.index, df_r['Equity'], color=colors_strategy, lw=2, label='전략 (MACD+RSI+SL)')
    ax.plot(df_r.index, bh_curve,       color=colors_bh,       lw=1.5, alpha=0.7, label='단순 보유(B&H)', linestyle='--')
    ax.axhline(10000, color='gray', lw=0.8, linestyle=':')

    # Shade area
    ax.fill_between(df_r.index, df_r['Equity'], 10000,
                    where=(df_r['Equity'] >= 10000), alpha=0.12, color=color_green)
    ax.fill_between(df_r.index, df_r['Equity'], 10000,
                    where=(df_r['Equity'] < 10000), alpha=0.12, color=color_red)

    title_short = period_name.replace('\n', ' ')
    # Color-code the return figure
    ret_color = color_green if r['전략수익'] >= 0 else color_red
    ax.set_title(f"{title_short}   |   전략 {r['전략수익']*100:+.1f}%  vs  단순보유 {r['단순보유']*100:+.1f}%   |   MDD {r['MDD']*100:.1f}%   샤프 {r['샤프']:.2f}",
                 fontsize=12, fontweight='bold', color=ret_color)
    ax.set_ylabel('포트폴리오 ($)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25)

# ---- 하단: 성과 비교 테이블 ----
ax_tbl = fig.add_subplot(gs[n_periods, :])
ax_tbl.axis('off')

col_labels = ['구간', '거래수', '승률(%)', '전략수익(%)', '단순보유(%)', '최대낙폭(MDD%)', '샤프지수']
table_data = []
cell_colors = []

for r in summary_rows:
    name_short = r['구간'].replace('\n', ' ')
    row_data = [
        name_short,
        str(r['거래수']),
        f"{r['승률']*100:.1f}%",
        f"{r['전략수익']*100:+.1f}%",
        f"{r['단순보유']*100:+.1f}%",
        f"{r['MDD']*100:.1f}%",
        f"{r['샤프']:.2f}",
    ]
    # Color per cell
    rc = []
    for j, val in enumerate(row_data):
        if j == 3:  # 전략수익
            rc.append('#d1fae5' if r['전략수익'] >= 0 else '#fee2e2')
        elif j == 4:  # 단순보유
            rc.append('#d1fae5' if r['단순보유'] >= 0 else '#fee2e2')
        elif j == 6:  # 샤프
            rc.append('#d1fae5' if r['샤프'] >= 0.5 else '#fee2e2')
        else:
            rc.append('#f8fafc')
    table_data.append(row_data)
    cell_colors.append(rc)

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=cell_colors,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.0)

# Header style
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#1e3a5f')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

ax_tbl.set_title('구간별 성과 요약 (초기 투자금 $10,000 기준)', fontsize=13, fontweight='bold', pad=12)

plt.savefig('c:/auto_bitcoin/sideways_backtest.png', dpi=130, bbox_inches='tight')
print("\n차트 저장 완료: c:/auto_bitcoin/sideways_backtest.png")

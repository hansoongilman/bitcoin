"""
enhanced_strategy_backtest.py

전략 v2.0 — 보완된 퀀트 전략
============================================================
추가된 지표:
  1. 볼린저 밴드 (BB)    — 변동성 수축/확장 감지
  2. ATR (Average True Range) — 동적 손절가 계산
  3. 거래량 필터 (Volume MA) — 거래량 확인 후 진입
  4. Stochastic RSI       — 일반 RSI보다 민감한 과매수/과매도
  5. OBV (On-Balance Volume) — 가격+거래량 복합 추세 판단

진입 조건 (ALL 충족):
  - Close > SMA30  (상승 추세)
  - MACD(12/21/7) 골든크로스
  - StochRSI < 0.8  (과매수 아님)
  - 볼린저 밴드 %b > 0.4  (밴드 중심 이상 = 상승 모멘텀)
  - 거래량 > Volume MA (평균 이상의 거래량 확인)

청산 조건 (ANY):
  - Close < SMA30 (추세 이탈)
  - MACD 역전
  - StochRSI > 0.85 (과매수 과열)
  - Close < 볼린저 하단 (하락 돌파)

손절:
  - ATR 기반 동적 손절: entry_price - atr_multiplier * ATR
  - 최소 3% / 최대 7% 손절로 클리핑
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
# 1. 상수 및 색상 테마 (찐 파랑)
# ══════════════════════════════════════════════════════════

BG_DARK   = '#020818'   # 찐 파랑 최외곽
BG_PANEL  = '#061025'   # 패널 배경
BG_CHART  = '#0a1830'   # 차트 내부
GRID_CLR  = '#0e2244'
TEXT_CLR  = '#93c5fd'   # 파란 계열 텍스트
TITLE_CLR = '#dbeafe'
C_BLUE    = '#3b82f6'
C_CYAN    = '#06b6d4'
C_GREEN   = '#22c55e'
C_RED     = '#ef4444'
C_YELLOW  = '#fbbf24'
C_PURPLE  = '#a78bfa'
C_ORANGE  = '#fb923c'

INITIAL_CAPITAL = 10000.0

# ══════════════════════════════════════════════════════════
# 2. 횡보/혼돈 구간 정의
# ══════════════════════════════════════════════════════════

SIDEWAYS = {
    '① 5월폭락後 삼각수렴 (21.05~07)': ('2021-05-19', '2021-07-20'),
    '② 엘살바도르 쇼크 (21.09~10)':   ('2021-09-07', '2021-10-05'),
    '③ LUNA붕괴 직전 반등 (22.03~05)': ('2022-03-28', '2022-05-08'),
    '④ 크립토윈터 지루한 횡보 (22.08~11)': ('2022-08-15', '2022-11-07'),
    '⑤ 상승인지 횡보인지 (23.04~06)': ('2023-04-01', '2023-06-15'),
    '⑥ ETF 눌림목 횡보 (24.05~08)':   ('2024-05-20', '2024-08-04'),
}

# ══════════════════════════════════════════════════════════
# 3. 데이터 수신
# ══════════════════════════════════════════════════════════

print("데이터 수신 중 (2021~2025)...")
df_all = yf.download('BTC-USD', start='2021-01-01', end='2025-01-01', interval='1d', progress=False)
df_all.dropna(inplace=True)
if isinstance(df_all.columns, pd.MultiIndex):
    df_all.columns = df_all.columns.droplevel(1)

# ══════════════════════════════════════════════════════════
# 4. 보완된 지표 계산 함수
# ══════════════════════════════════════════════════════════

def add_all_indicators(df):
    """전략 v2.0 — 모든 지표 한 번에 계산"""
    df = df.copy()

    # ── 기본 지표 ──
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    df['SMA200']  = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()  # 장기 추세 필터
    df['EMA20']   = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()   # 단기 추세

    # ── MACD (12/21/7) ──
    macd_obj = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']    = macd_obj.macd()
    df['MACD_Sig']= macd_obj.macd_signal()
    df['MACD_Hist']= macd_obj.macd_diff()  # MACD 히스토그램 추가

    # ── RSI ──
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # ── Stochastic RSI (sensitized RSI) ──
    stoch_rsi = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['StochRSI_K']  = stoch_rsi.stoch()
    df['StochRSI_D']  = stoch_rsi.stoch_signal()

    # ── 볼린저 밴드 ──
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Mid']   = bb.bollinger_mavg()
    df['BB_Pct']   = bb.bollinger_pband()    # %b = 현재 위치 (0=하단, 1=상단)
    df['BB_Width'] = bb.bollinger_wband()    # 밴드 폭 = 변동성 크기

    # ── ATR (Average True Range) — 동적 손절 ──
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

    # ── 거래량 지표 ──
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']     # 1 이상 = 평균 이상 거래량

    # ── OBV (On-Balance Volume) — 가격+거래량 복합 추세 ──
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()       # OBV의 20일 EMA

    # ── ADX (Average Directional Index) — 추세 강도 ──
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['DI_Plus']  = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()

    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════
# 5. 전략 진입/청산 조건
# ══════════════════════════════════════════════════════════

def check_entry_v2(row):
    """전략 v2.0 진입 조건"""
    trend_up     = row['Close'] > row['SMA30']
    macd_bull    = row['MACD'] > row['MACD_Sig']
    stochrsi_ok  = row['StochRSI_K'] < 80          # StochRSI 과매수 아님
    bb_ok        = row['BB_Pct'] > 0.4              # 볼린저 중심 이상
    vol_confirm  = row['Vol_Ratio'] > 1.0           # 평균 이상 거래량
    adx_trend    = row['ADX'] > 20                  # 추세 강도 확인 (20 이상 = 추세 유효)
    obv_bullish  = row['OBV'] > row['OBV_EMA']     # OBV가 EMA 위
    return trend_up and macd_bull and stochrsi_ok and bb_ok and vol_confirm

def check_exit_v2(row):
    """전략 v2.0 청산 조건"""
    trend_break  = row['Close'] < row['SMA30']
    macd_bear    = row['MACD'] < row['MACD_Sig']
    stochrsi_hot = row['StochRSI_K'] > 85
    bb_breakdown = row['Close'] < row['BB_Lower']   # 볼린저 하단 이탈
    return trend_break or macd_bear or stochrsi_hot or bb_breakdown

# ══════════════════════════════════════════════════════════
# 6. 백테스트 엔진 (ATR 동적 손절 적용)
# ══════════════════════════════════════════════════════════

def backtest_v2(df, initial=10000.0, atr_mult=1.5):
    pos = 0.0; cash = initial; ep = 0.0; entry_atr = 0.0
    equity = []; trades = []

    for ts, row in df.iterrows():
        if pos > 0:
            # ATR 기반 동적 손절 (최소 3%, 최대 7% 클리핑)
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

# ══════════════════════════════════════════════════════════
# 7. 기본(v1) vs 보완(v2) 비교 백테스트
# ══════════════════════════════════════════════════════════

def check_entry_v1(row):
    return (row['Close'] > row['SMA30'] and row['MACD'] > row['MACD_Sig'] and row['RSI'] < 60)

def check_exit_v1(row):
    return (row['Close'] < row['SMA30'] or row['MACD'] < row['MACD_Sig'] or row['RSI'] > 70)

def backtest_v1(df, initial=10000.0, sl_pct=0.05):
    pos = 0.0; cash = initial; ep = 0.0
    equity = []; trades = []
    for ts, row in df.iterrows():
        if pos > 0:
            sl_p = ep * (1 - sl_pct)
            if float(row['Low']) <= sl_p:
                cash += pos * sl_p
                trades.append({'type': 'SL', 'profit': (sl_p / ep) - 1})
                pos = 0.0; ep = 0.0
            elif check_exit_v1(row):
                cash += pos * float(row['Close'])
                trades.append({'type': 'EXIT', 'profit': (float(row['Close']) / ep) - 1})
                pos = 0.0; ep = 0.0
        if pos == 0 and check_entry_v1(row):
            ep = float(row['Close']); pos = cash / ep; cash = 0.0
        equity.append(cash + pos * float(row['Close']))
    df = df.copy()
    df['Equity'] = equity
    if not equity: return df, [], 0, 0, -99, 0, 0
    cum_r = (df['Equity'].iloc[-1] / initial) - 1
    df['Peak'] = df['Equity'].cummax()
    mdd = ((df['Equity'] - df['Peak']) / df['Peak']).min()
    dr  = df['Equity'].pct_change()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() != 0 else -99
    wr = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    return df, trades, cum_r, mdd, sharpe, wr, bh

# ══════════════════════════════════════════════════════════
# 8. 실행 & 결과 수집
# ══════════════════════════════════════════════════════════

print("지표 계산 중 (전체 데이터)...")
df_ind = add_all_indicators(df_all)

results_v1 = {}
results_v2 = {}
summary    = []

for name, (start, end) in SIDEWAYS.items():
    df_s = df_ind.loc[start:end].copy()
    if len(df_s) < 10:
        continue

    df_r1, t1, cr1, mdd1, sh1, wr1, bh = backtest_v1(df_s)
    df_r2, t2, cr2, mdd2, sh2, wr2, _  = backtest_v2(df_s)
    results_v1[name] = df_r1
    results_v2[name] = df_r2
    summary.append({'name': name, 'start': start, 'end': end,
                    'cr1': cr1, 'sh1': sh1, 'mdd1': mdd1, 'n1': len(t1), 'wr1': wr1,
                    'cr2': cr2, 'sh2': sh2, 'mdd2': mdd2, 'n2': len(t2), 'wr2': wr2,
                    'bh': bh})

# 출력
print(f"\n{'구간':<38} {'v1 수익':>9} {'v2 수익':>9} {'개선':>8} {'v1 샤프':>8} {'v2 샤프':>8} {'단순보유':>9}")
print("─" * 95)
for r in summary:
    diff = r['cr2'] - r['cr1']
    marker = 'OK' if diff >= 0 else 'NG'
    print(f"{r['name']:<38} {r['cr1']*100:>+8.1f}% {r['cr2']*100:>+8.1f}% {diff*100:>+7.1f}% {marker} {r['sh1']:>8.2f} {r['sh2']:>8.2f} {r['bh']*100:>+8.1f}%")

# ══════════════════════════════════════════════════════════
# 9. 시각화 (찐 파랑 배경 + 인디케이터 서브차트)
# ══════════════════════════════════════════════════════════

n_p = len(summary)
fig = plt.figure(figsize=(22, 6 * n_p + 7), facecolor=BG_DARK)
fig.suptitle('BTC 퀀트 전략 v2.0  —  보완된 전략 (볼린저밴드 + ATR동적SL + StochRSI + 거래량필터 + OBV + ADX)',
             color=TITLE_CLR, fontsize=14, fontweight='bold', y=0.995)

outer_gs = gridspec.GridSpec(n_p + 1, 1, figure=fig, height_ratios=[3] * n_p + [1.8], hspace=0.55)

def set_ax_style(ax):
    ax.set_facecolor(BG_CHART)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.grid(True, alpha=0.18, color=GRID_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)

for i, r in enumerate(summary):
    name = r['name']
    df_1 = results_v1[name]
    df_2 = results_v2[name]
    df_s = df_ind.loc[r['start']:r['end']].copy()

    # 각 구간: 3-row sub-layout (자산곡선 / BB+가격 / 보조지표)
    inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[i],
                                                height_ratios=[2, 1.2, 0.8], hspace=0.0)
    ax_eq  = fig.add_subplot(inner_gs[0])
    ax_bb  = fig.add_subplot(inner_gs[1], sharex=ax_eq)
    ax_osc = fig.add_subplot(inner_gs[2], sharex=ax_eq)

    for ax in [ax_eq, ax_bb, ax_osc]:
        set_ax_style(ax)

    bh_curve = INITIAL_CAPITAL * (df_2['Close'] / df_2['Close'].iloc[0])

    # ─ 자산 곡선 ─
    ax_eq.plot(df_1.index, df_1['Equity'], color=C_ORANGE, lw=1.4, label=f'v1 기본 ({r["cr1"]*100:+.1f}%)', alpha=0.8, linestyle='--')
    ax_eq.plot(df_2.index, df_2['Equity'], color=C_CYAN,   lw=2.0, label=f'v2 보완 ({r["cr2"]*100:+.1f}%)', zorder=3)
    ax_eq.plot(df_2.index, bh_curve,       color=C_YELLOW, lw=1.0, label=f'B&H ({r["bh"]*100:+.1f}%)', alpha=0.6, linestyle=':')
    ax_eq.axhline(INITIAL_CAPITAL, color='#334155', lw=0.7, linestyle=':')
    ax_eq.fill_between(df_2.index, df_2['Equity'], INITIAL_CAPITAL,
                        where=(df_2['Equity'] >= INITIAL_CAPITAL), alpha=0.12, color=C_GREEN)
    ax_eq.fill_between(df_2.index, df_2['Equity'], INITIAL_CAPITAL,
                        where=(df_2['Equity'] <  INITIAL_CAPITAL), alpha=0.12, color=C_RED)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    diff = r['cr2'] - r['cr1']
    imp_color = C_GREEN if diff >= 0 else C_RED
    title = (f"{name}  |  개선: {diff*100:+.1f}%p  |  "
             f"v2 거래 {r['n2']}건  승률 {r['wr2']*100:.0f}%  "
             f"MDD {r['mdd2']*100:.1f}%  샤프 {r['sh2']:.2f}")
    ax_eq.set_title(title, color=imp_color, fontsize=10.5, fontweight='bold', pad=5)
    leg = ax_eq.legend(loc='upper left', fontsize=9, facecolor=BG_PANEL, edgecolor=GRID_CLR)
    for t in leg.get_texts(): t.set_color(TITLE_CLR)

    # ─ 볼린저 밴드 + 가격 ─
    if len(df_s) > 0 and 'BB_Upper' in df_s.columns:
        ax_bb.plot(df_s.index, df_s['Close'],    color=TITLE_CLR, lw=0.9, label='BTC 가격')
        ax_bb.plot(df_s.index, df_s['BB_Upper'], color=C_RED,    lw=0.7, alpha=0.7, linestyle='--', label='BB 상단')
        ax_bb.plot(df_s.index, df_s['BB_Mid'],   color=C_BLUE,   lw=0.7, alpha=0.7, linestyle='-',  label='BB 중심')
        ax_bb.plot(df_s.index, df_s['BB_Lower'], color=C_GREEN,  lw=0.7, alpha=0.7, linestyle='--', label='BB 하단')
        ax_bb.fill_between(df_s.index, df_s['BB_Upper'], df_s['BB_Lower'], alpha=0.06, color=C_BLUE)
        ax_bb.set_ylabel('가격 + BB', fontsize=8)
        leg2 = ax_bb.legend(loc='upper left', fontsize=7.5, facecolor=BG_PANEL, edgecolor=GRID_CLR, ncol=4)
        for t in leg2.get_texts(): t.set_color(TITLE_CLR)

    # ─ 보조 오실레이터 (StochRSI K, ADX) ─
    if len(df_s) > 0 and 'StochRSI_K' in df_s.columns:
        ax_osc.plot(df_s.index, df_s['StochRSI_K'], color=C_PURPLE, lw=0.9, label='Stoch RSI %K')
        ax_osc.axhline(80, color=C_RED,   lw=0.7, linestyle='--', alpha=0.6)
        ax_osc.axhline(20, color=C_GREEN, lw=0.7, linestyle='--', alpha=0.6)
        ax_osc.fill_between(df_s.index, df_s['StochRSI_K'], 80, where=(df_s['StochRSI_K'] > 80), alpha=0.15, color=C_RED)
        ax_osc.fill_between(df_s.index, df_s['StochRSI_K'], 20, where=(df_s['StochRSI_K'] < 20), alpha=0.15, color=C_GREEN)

        ax_adx = ax_osc.twinx()
        ax_adx.set_facecolor(BG_CHART)
        ax_adx.plot(df_s.index, df_s['ADX'], color=C_ORANGE, lw=0.7, alpha=0.6, label='ADX', linestyle=':')
        ax_adx.axhline(25, color=C_ORANGE, lw=0.5, linestyle=':', alpha=0.4)
        ax_adx.set_ylim(0, 80)
        ax_adx.tick_params(colors=TEXT_CLR, labelsize=7)
        ax_adx.set_ylabel('ADX', color=C_ORANGE, fontsize=7)
        for sp in ax_adx.spines.values(): sp.set_color(GRID_CLR)

        ax_osc.set_ylim(0, 100)
        ax_osc.set_ylabel('StochRSI', fontsize=8)
        leg3 = ax_osc.legend(loc='upper left', fontsize=7.5, facecolor=BG_PANEL, edgecolor=GRID_CLR)
        for t in leg3.get_texts(): t.set_color(TITLE_CLR)

    plt.setp(ax_eq.xaxis.get_majorticklabels(),  visible=False)
    plt.setp(ax_bb.xaxis.get_majorticklabels(),  visible=False)
    ax_osc.tick_params(axis='x', labelsize=8, colors=TEXT_CLR)

# ══════════════════════════════════════════════════════════
# 10. 하단 성과 테이블
# ══════════════════════════════════════════════════════════

ax_tbl = fig.add_subplot(outer_gs[-1])
ax_tbl.set_facecolor(BG_PANEL)
ax_tbl.axis('off')
ax_tbl.set_title('전략 v1(기본) vs v2(보완) 성과 종합 비교  $10,000 기준',
                 color=TITLE_CLR, fontsize=12, fontweight='bold', pad=8)

col_labels = ['구간', 'v1 수익', 'v1 샤프', 'v2 수익', 'v2 샤프', 'MDD(v2)', '개선도', 'B&H']
table_data  = []
cell_colors = []

for r in summary:
    diff = r['cr2'] - r['cr1']
    row = [
        r['name'],
        f"{r['cr1']*100:+.1f}%",
        f"{r['sh1']:.2f}",
        f"{r['cr2']*100:+.1f}%",
        f"{r['sh2']:.2f}",
        f"{r['mdd2']*100:.1f}%",
        f"{diff*100:+.1f}%p",
        f"{r['bh']*100:+.1f}%",
    ]
    light_bg = '#0a1830'
    rc = [light_bg] * 8
    rc[1] = '#0d2a14' if r['cr1'] >= 0 else '#2a0d0d'
    rc[3] = '#0d2a14' if r['cr2'] >= 0 else '#2a0d0d'
    rc[6] = '#0d2a14' if diff >= 0      else '#2a0d0d'
    table_data.append(row)
    cell_colors.append(rc)

tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                   cellColours=cell_colors, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#0c2461')
    tbl[(0, j)].set_text_props(color='#93c5fd', fontweight='bold')
for (r_idx, c_idx), cell in tbl.get_celld().items():
    if r_idx > 0:
        cell.get_text().set_color(TITLE_CLR)
    cell.set_edgecolor(GRID_CLR)

save_path = 'c:/auto_bitcoin/enhanced_strategy_backtest.png'
plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_DARK)
print(f"\n차트 저장 완료: {save_path}")

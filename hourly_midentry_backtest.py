"""
hourly_midentry_backtest.py

목적:
  - 현실적인 "중간 진입" 시나리오 시뮬레이션
  - 최적 파라미터 (SMA30, RSI<60진입, SL5%, MACD 12/21/7) 적용
  - 2024-01-01 ~ 현재까지 1시간봉 데이터로 백테스트
  - 상승(ETF 랠리) → 횡보(눌림목) → 하락(조정) → 회복 구간을 모두 포함
  - 실제로 "지금 시장에 들어가면 어떻게 되나?"에 가장 근접한 검증
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import ta
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

INITIAL_CAPITAL = 10000.0

# ══════════════════════════════════════════════════════════
# 1. 데이터 수집
#    - yfinance: 1시간봉은 최대 730일 지원
#    - 2024-01-01 ~ 현재 (약 15개월)
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("  BTC-USD 1시간봉 중간진입 백테스트")
print("  기간: 2024-01-01 ~ 현재 (직전 730일 이내)")
print("=" * 60)

print("\n[1/5] 1시간봉 데이터 수신 중 (yfinance 최대 730일)...")
# yfinance 1h는 최근 729일 이내의 데이터만 지원 (start= 사용 불가, period= 사용)
df = yf.download('BTC-USD', period='729d', interval='1h', progress=True)
df.dropna(inplace=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

total_candles = len(df)
start_dt = df.index[0]
end_dt   = df.index[-1]
print(f"    총 {total_candles:,}개 캔들  ({start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')})")

# ══════════════════════════════════════════════════════════
# 2. 지표 계산 (튜닝 최적 파라미터)
# ══════════════════════════════════════════════════════════

print("\n[2/5] 기술 지표 계산 중 (최적 파라미터 적용)...")

# SMA30 (30시간이동평균)
df['SMA']       = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
# RSI
df['RSI']       = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
# MACD (12/21/7)
macd_obj        = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
df['MACD_Line'] = macd_obj.macd()
df['MACD_Sig']  = macd_obj.macd_signal()
df.dropna(inplace=True)

print(f"    지표 계산 후 유효 캔들: {len(df):,}개")

# ══════════════════════════════════════════════════════════
# 3. 백테스트 엔진
# ══════════════════════════════════════════════════════════

print("\n[3/5] 백테스트 실행 중...")

SL_PCT     = 0.05   # 5% 손절 (튜닝 결과)
RSI_ENTRY  = 60     # RSI 60 미만 진입
RSI_EXIT   = 70     # RSI 70 초과 청산

cash      = INITIAL_CAPITAL
position  = 0.0
entry_px  = 0.0
equity    = []
trades    = []
in_trade_at = []

for ts, row in df.iterrows():
    close = float(row['Close'])
    low   = float(row['Low'])
    sma   = float(row['SMA'])
    rsi   = float(row['RSI'])
    macd  = float(row['MACD_Line'])
    msig  = float(row['MACD_Sig'])

    # ── 보유 중 ──
    if position > 0:
        sl_price = entry_px * (1 - SL_PCT)

        # 손절선 도달
        if low <= sl_price:
            cash += position * sl_price
            trades.append({'ts': ts, 'type': 'SL', 'price': sl_price,
                           'profit': (sl_price / entry_px) - 1})
            position = 0.0; entry_px = 0.0

        # 일반 청산 (추세 이탈 or MACD 역전 or RSI 과매수)
        elif close < sma or macd < msig or rsi > RSI_EXIT:
            cash += position * close
            trades.append({'ts': ts, 'type': 'EXIT', 'price': close,
                           'profit': (close / entry_px) - 1})
            position = 0.0; entry_px = 0.0

    # ── 미보유 중 - 진입 조건 ──
    if position == 0:
        if close > sma and macd > msig and rsi < RSI_ENTRY:
            entry_px  = close
            position  = cash / entry_px
            cash      = 0.0
            in_trade_at.append(ts)

    equity.append(cash + position * close)

df['Equity']  = equity
df['BH']      = INITIAL_CAPITAL * (df['Close'] / df['Close'].iloc[0])

# ══════════════════════════════════════════════════════════
# 4. 성과 지표 계산
# ══════════════════════════════════════════════════════════

print("\n[4/5] 성과 지표 계산 중...")

final_equity  = df['Equity'].iloc[-1]
cum_return    = (final_equity / INITIAL_CAPITAL) - 1
bh_return     = (df['BH'].iloc[-1]  / INITIAL_CAPITAL) - 1

df['Peak']    = df['Equity'].cummax()
df['DD']      = (df['Equity'] - df['Peak']) / df['Peak']
mdd           = df['DD'].min()

dr            = df['Equity'].pct_change()
sharpe        = (dr.mean() / dr.std()) * np.sqrt(365 * 24) if dr.std() != 0 else 0  # annualized for hourly

winning  = [t for t in trades if t['profit'] > 0]
losing   = [t for t in trades if t['profit'] <= 0]
win_rate = len(winning) / len(trades) if trades else 0
avg_win  = np.mean([t['profit'] for t in winning]) * 100 if winning else 0
avg_loss = np.mean([t['profit'] for t in losing])  * 100 if losing  else 0

sl_count   = sum(1 for t in trades if t['type'] == 'SL')
exit_count = sum(1 for t in trades if t['type'] == 'EXIT')

days_total  = (end_dt - start_dt).days
months_est  = days_total / 30

print(f"""
{'='*60}
  최적 파라미터 Mid-Entry 백테스트 최종 결과
  기간: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days_total}일, 약 {months_est:.0f}개월)
  데이터 해상도: 1시간봉  |  총 캔들 수: {total_candles:,}개
{'='*60}
초기 투자금           : ${INITIAL_CAPITAL:>10,.2f}
최종 자산 (전략)      : ${final_equity:>10,.2f}
최종 자산 (단순보유)  : ${INITIAL_CAPITAL*(1+bh_return):>10,.2f}
─────────────────────────────────────────────────────
누적 수익률 (전략)    : {cum_return*100:>+8.2f}%
누적 수익률 (단순보유): {bh_return*100:>+8.2f}%
최대 낙폭 (MDD)       : {mdd*100:>+8.2f}%
샤프 지수 (연환산)    : {sharpe:>8.2f}
─────────────────────────────────────────────────────
총 거래 횟수          : {len(trades):>4}건
  - 손절(SL) 발동    : {sl_count:>4}건
  - 일반 청산        : {exit_count:>4}건
승률                  : {win_rate*100:>6.1f}%
평균 이익 (건당)      : {avg_win:>+7.2f}%
평균 손실 (건당)      : {avg_loss:>+7.2f}%
{'='*60}
""")

# ══════════════════════════════════════════════════════════
# 5. 시각화
# ══════════════════════════════════════════════════════════

print("[5/5] 차트 생성 중...")

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('#0f172a')

gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 0.7], hspace=0.08)

ax1 = fig.add_subplot(gs[0])  # 자산 곡선
ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 낙폭
ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI
ax4 = fig.add_subplot(gs[3])  # 성과 요약 표

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#1e293b')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.grid(True, alpha=0.15, color='#475569')

# ── 패널1: 자산 곡선 ──
ax1.plot(df.index, df['Equity'], color='#60a5fa', lw=1.5, label=f'전략 자산 ({cum_return*100:+.1f}%)', zorder=3)
ax1.plot(df.index, df['BH'],     color='#f59e0b', lw=1.0, label=f'단순 보유 ({bh_return*100:+.1f}%)', alpha=0.7, linestyle='--', zorder=2)
ax1.axhline(INITIAL_CAPITAL, color='#475569', lw=0.8, linestyle=':')
ax1.fill_between(df.index, df['Equity'], INITIAL_CAPITAL,
                 where=(df['Equity'] >= INITIAL_CAPITAL), alpha=0.2, color='#22c55e', zorder=1)
ax1.fill_between(df.index, df['Equity'], INITIAL_CAPITAL,
                 where=(df['Equity'] < INITIAL_CAPITAL),  alpha=0.2, color='#ef4444', zorder=1)

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.set_ylabel('포트폴리오 ($)', color='#94a3b8', fontsize=11)
legend = ax1.legend(loc='upper left', fontsize=10, facecolor='#1e293b', edgecolor='#334155')
for text in legend.get_texts():
    text.set_color('#e2e8f0')
ax1.set_title(f'BTC-USD 1시간봉 중간진입 백테스트  |  {start_dt.strftime("%Y-%m-%d")} ~ {end_dt.strftime("%Y-%m-%d")}  |  초기 ${INITIAL_CAPITAL:,.0f}',
              color='#e2e8f0', fontsize=13, fontweight='bold', pad=10)

# ── 패널2: 낙폭 (Drawdown) ──
ax2.fill_between(df.index, df['DD'] * 100, 0, color='#ef4444', alpha=0.6)
ax2.axhline(-5, color='#f59e0b', lw=0.8, linestyle='--', alpha=0.6, label='SL 기준선 (-5%)')
ax2.set_ylabel('낙폭 (%)', color='#94a3b8', fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.legend(loc='lower left', fontsize=8, facecolor='#1e293b', edgecolor='#334155')
for text in ax2.get_legend().get_texts():
    text.set_color('#e2e8f0')

# ── 패널3: RSI ──
ax3.plot(df.index, df['RSI'], color='#a78bfa', lw=0.8)
ax3.axhline(RSI_ENTRY, color='#22c55e', lw=0.8, linestyle='--', alpha=0.7, label=f'진입 RSI<{RSI_ENTRY}')
ax3.axhline(RSI_EXIT,  color='#ef4444', lw=0.8, linestyle='--', alpha=0.7, label=f'청산 RSI>{RSI_EXIT}')
ax3.fill_between(df.index, df['RSI'], RSI_ENTRY, where=(df['RSI'] < RSI_ENTRY), alpha=0.2, color='#22c55e')
ax3.fill_between(df.index, df['RSI'], RSI_EXIT,  where=(df['RSI'] > RSI_EXIT),  alpha=0.2, color='#ef4444')
ax3.set_ylim(0, 100)
ax3.set_ylabel('RSI', color='#94a3b8', fontsize=9)
ax3.legend(loc='upper left', fontsize=8, facecolor='#1e293b', edgecolor='#334155')
for text in ax3.get_legend().get_texts():
    text.set_color('#e2e8f0')

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

# ── 패널4: 성과 요약 표 ──
ax4.set_facecolor('#0f172a')
ax4.axis('off')

table_data = [
    ['초기 투자금', f'${INITIAL_CAPITAL:,.0f}',
     '최종 자산 (전략)', f'${final_equity:,.2f}',
     '최종 자산 (단순보유)', f'${INITIAL_CAPITAL*(1+bh_return):,.2f}'],
    ['전략 수익률', f'{cum_return*100:+.2f}%',
     '단순보유 수익률', f'{bh_return*100:+.2f}%',
     '최대 낙폭 MDD', f'{mdd*100:.2f}%'],
    ['샤프 지수', f'{sharpe:.3f}',
     '총 거래 횟수', f'{len(trades)}건 (손절 {sl_count}건)',
     '승률 / 평균이익/손실', f'{win_rate*100:.1f}% / {avg_win:+.2f}% / {avg_loss:+.2f}%'],
]

for r_idx, row_data in enumerate(table_data):
    for c_idx in range(0, 6, 2):
        label = row_data[c_idx]
        value = row_data[c_idx+1]
        x_l = 0.01 + (c_idx // 2) * 0.34
        x_v = x_l + 0.16
        y = 0.75 - r_idx * 0.28
        ax4.text(x_l, y, label, color='#94a3b8', fontsize=10, transform=ax4.transAxes)
        # Color the value
        val_color = '#e2e8f0'
        if '+' in value and '%' in value:
            val_color = '#22c55e'
        elif value.startswith('-') and '%' in value:
            val_color = '#ef4444'
        ax4.text(x_v, y, value, color=val_color, fontsize=11, fontweight='bold', transform=ax4.transAxes)

save_path = 'c:/auto_bitcoin/hourly_midentry_backtest.png'
plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='#0f172a')
print(f"차트 저장 완료: {save_path}")

"""
live_monitor.py — 실시간 BTC 신호 모니터 (터미널)
==========================================================
실행: python live_monitor.py
  - 1분마다 yfinance에서 최신 BTC 가격 + 지표 갱신
  - 매수 조건 달성 시 콘솔에 BUY SIGNAL 크게 출력
  - 매도/손절 조건 달성 시 SELL SIGNAL 출력
  - Ctrl+C 로 종료
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── 설정 ─────────────────────────────────────────────────
CHECK_INTERVAL_SEC = 60   # 1분마다 체크
INITIAL_CAPITAL    = 10000.0
SL_PCT             = 0.05   # 5% 손절

# 포지션 상태
position = 0.0
cash     = INITIAL_CAPITAL
entry_px = 0.0

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_data(symbol='BTC-USD', period='5d', interval='1h'):
    """최근 5일치 1시간봉 데이터 가져와서 지표 계산"""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if len(df) < 40:
        return None

    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    macd_obj      = ta.trend.MACD(df['Close'], window_fast=12, window_slow=21, window_sign=7)
    df['MACD']    = macd_obj.macd()
    df['MACD_Sig']= macd_obj.macd_signal()
    stoch         = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['StochRSI']= stoch.stoch()
    bb            = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_Pct']  = bb.bollinger_pband()
    df['BB_Lower']= bb.bollinger_lband()
    df['ATR']     = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Vol_MA20']= df['Volume'].rolling(20).mean()
    df['Vol_Ratio']= df['Volume'] / df['Vol_MA20']
    df.dropna(inplace=True)
    return df

def check_entry(row):
    return (float(row['Close']) > float(row['SMA30']) and
            float(row['MACD'])  > float(row['MACD_Sig']) and
            float(row['StochRSI']) < 80 and
            float(row['BB_Pct'])   > 0.4 and
            float(row['Vol_Ratio']) > 1.0)

def check_exit(row, ep):
    sl = ep * (1 - SL_PCT)
    if float(row['Close']) <= sl:
        return 'SL'
    if (float(row['Close']) < float(row['SMA30']) or
        float(row['MACD'])  < float(row['MACD_Sig']) or
        float(row['StochRSI']) > 85 or
        float(row['Close'])    < float(row['BB_Lower'])):
        return 'EXIT'
    return None

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def print_dashboard(row, signal, equity, pnl_pct, cycle):
    clear()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    price = float(row['Close'])
    sma   = float(row['SMA30'])
    macd  = float(row['MACD'])
    msig  = float(row['MACD_Sig'])
    srsi  = float(row['StochRSI'])
    bb_p  = float(row['BB_Pct'])
    vr    = float(row['Vol_Ratio'])
    atr   = float(row['ATR'])

    trend_ok = price > sma
    macd_ok  = macd  > msig
    srsi_ok  = srsi  < 80
    bb_ok    = bb_p  > 0.4
    vol_ok   = vr    > 1.0
    conditions_met = sum([trend_ok, macd_ok, srsi_ok, bb_ok, vol_ok])

    print(color("═" * 65, '34'))
    print(color(f"  BTC 실시간 신호 모니터  |  {now}  |  #{cycle}", '96'))
    print(color("═" * 65, '34'))
    print(f"\n  {'현재가':<14}: {color(f'${price:,.2f}', '97')}")
    print(f"  {'자산 (전략)':<14}: {color(f'${equity:,.2f}  ({pnl_pct:+.2f}%)', '92' if pnl_pct >= 0 else '91')}")
    if entry_px > 0:
        pos_pnl = (price / entry_px - 1) * 100
        print(f"  {'현재 포지션':<14}: {color(f'보유중  진입가 ${entry_px:,.2f}  ({pos_pnl:+.2f}%)', '93')}")
    else:
        print(f"  {'현재 포지션':<14}: {color('현금 보유', '94')}")

    print(f"\n  {'─ 지표 상태 ─':}")
    def chk(flag): return color('  OK', '92') if flag else color('  --', '90')
    print(f"  SMA30 추세    : {chk(trend_ok)}  ({price:,.0f} {'>' if trend_ok else '<'} {sma:,.0f})")
    print(f"  MACD 골든크로스: {chk(macd_ok)}")
    print(f"  StochRSI < 80 : {chk(srsi_ok)}  (현재 {srsi:.1f})")
    print(f"  BB %b > 0.4   : {chk(bb_ok)}   ({bb_p:.2f})")
    print(f"  거래량 확인    : {chk(vol_ok)}  (비율 {vr:.2f}x)")
    print(f"  ATR            :        ${atr:,.2f}")
    print(f"\n  조건 충족: {conditions_met}/5")

    print(color("\n" + "═" * 65, '34'))
    if signal == 'BUY':
        print(color("""
  ██████╗ ██╗   ██╗██╗   ██╗    ███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗
  ██╔══██╗██║   ██║╚██╗ ██╔╝    ██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║
  ██████╔╝██║   ██║ ╚████╔╝     ███████╗██║██║  ███╗██╔██╗ ██║███████║██║
  ██╔══██╗██║   ██║  ╚██╔╝      ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║
  ██████╔╝╚██████╔╝   ██║       ███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗
  ╚═════╝  ╚═════╝    ╚═╝       ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
""", '92;1'))
    elif signal == 'SL':
        print(color("\n  !! STOP-LOSS 손절 발동 !!", '91;1'))
    elif signal == 'EXIT':
        print(color("\n  >> SELL SIGNAL  일반 청산", '93;1'))
    else:
        print(color("  대기 중 (HOLD)  —  다음 체크까지 60초...", '90'))

    print(color("═" * 65, '34'))
    print(f"  다음 갱신: {CHECK_INTERVAL_SEC}초 후  |  Ctrl+C 로 종료")

# ── 메인 루프 ───────────────────────────────────────────
print("실시간 모니터 시작 중... (초기 데이터 수신)")
cycle = 0

while True:
    cycle += 1
    df = get_latest_data()
    if df is None or len(df) == 0:
        print("데이터 수신 실패. 60초 후 재시도...")
        time.sleep(CHECK_INTERVAL_SEC)
        continue

    row = df.iloc[-1]
    signal = None
    price  = float(row['Close'])

    # ── 포지션 있으면 청산 조건 먼저 체크 ──
    if position > 0:
        exit_type = check_exit(row, entry_px)
        if exit_type:
            signal = exit_type
            cash_new = position * price
            globals()['cash']     = cash_new
            globals()['position'] = 0.0
            globals()['entry_px'] = 0.0

    # ── 포지션 없으면 진입 조건 체크 ──
    if position == 0 and check_entry(row):
        signal = 'BUY'
        globals()['position'] = cash / price
        globals()['entry_px'] = price
        globals()['cash']     = 0.0

    equity  = cash + position * price
    pnl_pct = (equity / INITIAL_CAPITAL - 1) * 100

    try:
        print_dashboard(row, signal, equity, pnl_pct, cycle)
    except UnicodeEncodeError:
        # Fallback for terminals without ANSI
        print(f"[{datetime.now().strftime('%H:%M:%S')}] BTC: ${price:,.2f}  Equity: ${equity:,.2f} ({pnl_pct:+.2f}%)  Signal: {signal or 'HOLD'}")

    time.sleep(CHECK_INTERVAL_SEC)

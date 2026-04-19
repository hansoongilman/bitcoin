"""
live_dashboard_v3.py - v3.0 Multi-Coin Real-Time Trading Dashboard
Run: python live_dashboard_v3.py -> http://127.0.0.1:5000

Features:
- 6 coins live monitoring (BTC, ETH, SOL, XRP, ADA, DOGE)
- v3.0 strategy with ADX dual-regime (Bull/Normal mode)
- Trailing stop visualization
- Premium dark UI with real-time updates via SSE
"""

from flask import Flask, jsonify, render_template_string, Response
import yfinance as yf
import pandas as pd
import numpy as np
import ta, json, time, threading, warnings
from datetime import datetime

warnings.filterwarnings('ignore')
app = Flask(__name__)

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════
TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']
COIN_INFO = {
    'BTC-USD':  {'name': 'Bitcoin',  'symbol': 'BTC', 'color': '#f7931a'},
    'ETH-USD':  {'name': 'Ethereum', 'symbol': 'ETH', 'color': '#627eea'},
    'SOL-USD':  {'name': 'Solana',   'symbol': 'SOL', 'color': '#00ffa3'},
    'XRP-USD':  {'name': 'Ripple',   'symbol': 'XRP', 'color': '#00aaff'},
    'ADA-USD':  {'name': 'Cardano',  'symbol': 'ADA', 'color': '#0033ad'},
    'DOGE-USD': {'name': 'Dogecoin', 'symbol': 'DOGE','color': '#c3a634'},
}
INITIAL = 10000.0
CHECK_INTERVAL = 60
lock = threading.Lock()

# Per-coin state
def make_coin_state():
    return {
        'price': 0.0, 'prev_price': 0.0,
        'signal': 'HOLD', 'regime': 'NORMAL',
        'sma30': 0.0, 'ema10': 0.0,
        'macd': 0.0, 'macd_sig': 0.0,
        'stochrsi': 50.0, 'bb_pct': 0.5, 'bb_lower': 0.0,
        'atr': 0.0, 'vol_ratio': 1.0,
        'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0,
        'equity': INITIAL, 'pnl_pct': 0.0,
        'position': 0.0, 'cash': INITIAL, 'entry_px': 0.0,
        'trailing_high': 0.0, 'trailing_stop': 0.0,
        'in_bull_entry': False,
        'last_update': '',
        'conditions': [False] * 5,
        'condition_labels': ['', '', '', '', ''],
        'price_history': [],  # last 50 prices for sparkline
        'trade_log': [],  # last 10 trades
    }

states = {tk: make_coin_state() for tk in TICKERS}

# ══════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════
def calc_indicators(df):
    df = df.copy()
    df['SMA30']   = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    df['EMA10']   = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
    m = ta.trend.MACD(df['Close'], 12, 21, 7)
    df['MACD']    = m.macd()
    df['MACD_Sig']= m.macd_signal()
    df['StochRSI']= ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 14, 3).stoch()
    bb = ta.volatility.BollingerBands(df['Close'], 20)
    df['BB_Pct']  = bb.bollinger_pband()
    df['BB_Lower']= bb.bollinger_lband()
    df['ATR']     = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    vma = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / vma
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14)
    df['ADX']     = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus']= adx.adx_neg()
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════
# v3.0 Strategy Logic
# ══════════════════════════════════════════════════════════
def is_bull(row):
    return float(row['ADX']) > 25 and float(row['DI_Plus']) > float(row['DI_Minus'])

def entry_ok(row, bull):
    trend = float(row['Close']) > float(row['SMA30'])
    macd  = float(row['MACD']) > float(row['MACD_Sig'])
    bb    = float(row['BB_Pct']) > 0.4
    if bull:
        srsi = float(row['StochRSI']) < 90
        vol  = float(row['Vol_Ratio']) > 0.8
    else:
        srsi = float(row['StochRSI']) < 80
        vol  = float(row['Vol_Ratio']) > 1.0
    return trend and macd and srsi and bb and vol

def exit_check_bull(row, trailing_high):
    px = float(row['Close'])
    atr = float(row['ATR'])
    ts = trailing_high - 2.0 * atr
    if px < ts:
        return 'TRAIL_STOP'
    if float(row['StochRSI']) > 95:
        return 'EXTREME_OB'
    if px < float(row['EMA10']) and float(row['MACD']) < float(row['MACD_Sig']):
        return 'EMA10_BREAK'
    if px < float(row['BB_Lower']):
        return 'BB_BREAK'
    return None

def exit_check_normal(row, ep):
    px = float(row['Close'])
    atr = float(row['ATR'])
    sl_pct = np.clip(1.5 * atr / ep, 0.03, 0.07)
    if px <= ep * (1 - sl_pct):
        return 'STOP_LOSS'
    if (px < float(row['SMA30']) or
        float(row['MACD']) < float(row['MACD_Sig']) or
        float(row['StochRSI']) > 85 or
        px < float(row['BB_Lower'])):
        return 'EXIT'
    return None

# ══════════════════════════════════════════════════════════
# Live Monitor Thread
# ══════════════════════════════════════════════════════════
def live_loop():
    while True:
        for tk in TICKERS:
            try:
                df = yf.download(tk, period='5d', interval='1h', progress=False)
                df.dropna(inplace=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if len(df) < 40:
                    continue
                df = calc_indicators(df)
                row = df.iloc[-1]

                with lock:
                    s = states[tk]
                    px = float(row['Close'])
                    hi = float(row['High'])
                    bull = is_bull(row)

                    s['prev_price'] = s['price']
                    s['price'] = px
                    s['sma30'] = float(row['SMA30'])
                    s['ema10'] = float(row['EMA10'])
                    s['macd'] = float(row['MACD'])
                    s['macd_sig'] = float(row['MACD_Sig'])
                    s['stochrsi'] = float(row['StochRSI'])
                    s['bb_pct'] = float(row['BB_Pct'])
                    s['bb_lower'] = float(row['BB_Lower'])
                    s['atr'] = float(row['ATR'])
                    s['vol_ratio'] = float(row['Vol_Ratio'])
                    s['adx'] = float(row['ADX'])
                    s['di_plus'] = float(row['DI_Plus'])
                    s['di_minus'] = float(row['DI_Minus'])
                    s['regime'] = 'BULL' if bull else 'NORMAL'
                    s['last_update'] = datetime.now().strftime('%H:%M:%S')

                    # Conditions
                    c0 = px > s['sma30']
                    c1 = s['macd'] > s['macd_sig']
                    c2 = s['stochrsi'] < (90 if bull else 80)
                    c3 = s['bb_pct'] > 0.4
                    c4 = s['vol_ratio'] > (0.8 if bull else 1.0)
                    s['conditions'] = [c0, c1, c2, c3, c4]
                    s['condition_labels'] = [
                        f"{'OK' if c0 else '--'} ({px:,.0f} {'>' if c0 else '<'} {s['sma30']:,.0f})",
                        f"{'OK' if c1 else '--'}",
                        f"{'OK' if c2 else '--'} ({s['stochrsi']:.1f})",
                        f"{'OK' if c3 else '--'} ({s['bb_pct']:.2f})",
                        f"{'OK' if c4 else '--'} ({s['vol_ratio']:.2f}x)",
                    ]

                    # Price history for sparkline
                    s['price_history'].append(px)
                    if len(s['price_history']) > 50:
                        s['price_history'] = s['price_history'][-50:]

                    # Position management
                    pos = s['position']
                    ep = s['entry_px']
                    cash = s['cash']

                    if pos > 0:
                        if hi > s['trailing_high']:
                            s['trailing_high'] = hi
                        s['trailing_stop'] = s['trailing_high'] - 2.0 * s['atr']

                        if bull or s['in_bull_entry']:
                            sig = exit_check_bull(row, s['trailing_high'])
                            if sig:
                                s['cash'] = pos * px
                                s['position'] = 0.0
                                s['entry_px'] = 0.0
                                s['signal'] = sig
                                s['trailing_high'] = 0.0
                                s['trailing_stop'] = 0.0
                                s['in_bull_entry'] = False
                                pnl = (px / ep - 1) * 100
                                s['trade_log'].append({
                                    'time': s['last_update'],
                                    'type': sig,
                                    'pnl': round(pnl, 2)
                                })
                                if len(s['trade_log']) > 10:
                                    s['trade_log'] = s['trade_log'][-10:]
                            elif not bull:
                                s['in_bull_entry'] = False
                                sig2 = exit_check_normal(row, ep)
                                if sig2:
                                    s['cash'] = pos * px
                                    s['position'] = 0.0
                                    s['entry_px'] = 0.0
                                    s['signal'] = sig2
                                    s['trailing_high'] = 0.0
                                    s['trailing_stop'] = 0.0
                                    pnl = (px / ep - 1) * 100
                                    s['trade_log'].append({
                                        'time': s['last_update'],
                                        'type': sig2,
                                        'pnl': round(pnl, 2)
                                    })
                                    if len(s['trade_log']) > 10:
                                        s['trade_log'] = s['trade_log'][-10:]
                                else:
                                    s['signal'] = 'HOLD'
                            else:
                                s['signal'] = 'HOLD'
                        else:
                            sig = exit_check_normal(row, ep)
                            if sig:
                                s['cash'] = pos * px
                                s['position'] = 0.0
                                s['entry_px'] = 0.0
                                s['signal'] = sig
                                s['trailing_high'] = 0.0
                                s['trailing_stop'] = 0.0
                                pnl = (px / ep - 1) * 100
                                s['trade_log'].append({
                                    'time': s['last_update'],
                                    'type': sig,
                                    'pnl': round(pnl, 2)
                                })
                                if len(s['trade_log']) > 10:
                                    s['trade_log'] = s['trade_log'][-10:]
                            else:
                                s['signal'] = 'HOLD'

                    elif entry_ok(row, bull):
                        s['signal'] = 'BUY'
                        s['position'] = cash / px
                        s['entry_px'] = px
                        s['cash'] = 0.0
                        s['trailing_high'] = hi
                        s['trailing_stop'] = hi - 2.0 * s['atr']
                        s['in_bull_entry'] = bull
                        s['trade_log'].append({
                            'time': s['last_update'],
                            'type': 'BUY',
                            'pnl': 0
                        })
                        if len(s['trade_log']) > 10:
                            s['trade_log'] = s['trade_log'][-10:]
                    else:
                        s['signal'] = 'HOLD'

                    eq = s['cash'] + s['position'] * px
                    s['equity'] = eq
                    s['pnl_pct'] = (eq / INITIAL - 1) * 100

            except Exception as e:
                print(f"[{tk}] Error: {e}")

        time.sleep(CHECK_INTERVAL)

# ══════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════
@app.route('/api/state')
def api_state():
    with lock:
        out = {}
        for tk in TICKERS:
            s = states[tk]
            out[tk] = {k: v for k, v in s.items()}
    return jsonify(out)

@app.route('/api/stream')
def api_stream():
    def gen():
        while True:
            with lock:
                out = {}
                for tk in TICKERS:
                    s = states[tk]
                    out[tk] = {k: v for k, v in s.items()}
            yield f"data: {json.dumps(out)}\n\n"
            time.sleep(2)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/')
def index():
    return render_template_string(PAGE)

# ══════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto Quant v3.0 Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-dark:#020818;--bg-panel:#061025;--bg-card:#0a1830;
  --border:#0e2244;--text:#93c5fd;--text-dim:#475569;--text-bright:#dbeafe;
  --green:#22c55e;--red:#ef4444;--cyan:#06b6d4;--yellow:#fbbf24;
  --purple:#a78bfa;--orange:#fb923c;--blue:#3b82f6;
}
html,body{height:100%;background:var(--bg-dark);color:var(--text);
  font-family:'Inter',sans-serif;overflow-x:hidden}

/* Header */
.header{background:linear-gradient(135deg,#0a1535 0%,#061025 50%,#0c1a3e 100%);
  border-bottom:1px solid var(--border);padding:12px 24px;
  display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:1.15rem;font-weight:700;color:var(--text-bright);
  display:flex;align-items:center;gap:10px}
.header h1 span{background:linear-gradient(135deg,var(--cyan),var(--blue));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.3rem}
.live-badge{display:flex;align-items:center;gap:6px;padding:4px 12px;
  border-radius:20px;font-size:.72rem;font-weight:600;
  background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:blink 1.5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* Coin Tabs */
.coin-tabs{display:flex;gap:4px;padding:8px 16px;background:var(--bg-panel);
  border-bottom:1px solid var(--border);overflow-x:auto}
.coin-tab{padding:8px 16px;border-radius:8px;cursor:pointer;font-size:.82rem;
  font-weight:600;border:1px solid transparent;color:var(--text-dim);
  transition:all .2s;white-space:nowrap;display:flex;align-items:center;gap:8px}
.coin-tab:hover{background:rgba(59,130,246,.08);color:var(--text)}
.coin-tab.active{background:rgba(59,130,246,.15);color:var(--text-bright);
  border-color:var(--blue)}
.coin-tab .dot{width:8px;height:8px;border-radius:50%}
.coin-tab .mini-price{font-family:'JetBrains Mono',monospace;font-size:.72rem;
  color:var(--text-dim)}
.coin-tab .mini-sig{font-size:.6rem;padding:1px 5px;border-radius:3px;font-weight:700}
.mini-sig.BUY{background:rgba(34,197,94,.2);color:var(--green)}
.mini-sig.SELL,.mini-sig.EXIT,.mini-sig.STOP_LOSS,.mini-sig.TRAIL_STOP,.mini-sig.EXTREME_OB,.mini-sig.EMA10_BREAK,.mini-sig.BB_BREAK,.mini-sig.REGIME_EXIT{
  background:rgba(239,68,68,.2);color:var(--red)}
.mini-sig.HOLD{background:rgba(71,85,105,.2);color:var(--text-dim)}

/* Main Grid */
.main{display:grid;grid-template-columns:320px 1fr;gap:0;height:calc(100vh - 96px)}

/* Left Panel */
.left{padding:12px;display:flex;flex-direction:column;gap:10px;overflow-y:auto;
  border-right:1px solid var(--border);background:var(--bg-panel)}

.card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px}
.card-title{font-size:.7rem;font-weight:600;color:var(--blue);text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}

/* Price display */
.price-display{text-align:center;padding:8px 0}
.price-main{font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;
  transition:color .3s}
.price-main.up{color:var(--green)}.price-main.down{color:var(--red)}.price-main.flat{color:#fff}
.price-change{font-size:.8rem;margin-top:4px;font-family:'JetBrains Mono',monospace}

/* Regime badge */
.regime{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;
  border-radius:6px;font-size:.72rem;font-weight:700;margin:6px auto;text-align:center}
.regime.BULL{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.regime.NORMAL{background:rgba(71,85,105,.15);color:var(--text-dim);border:1px solid rgba(71,85,105,.3)}

/* Signal */
.signal-box{padding:10px;border-radius:8px;text-align:center;font-size:1rem;font-weight:700;
  margin:6px 0;transition:all .3s}
.signal-box.BUY{background:rgba(34,197,94,.12);color:var(--green);border:2px solid var(--green);
  animation:glow-g .8s infinite alternate}
.signal-box.SELL,.signal-box.EXIT,.signal-box.STOP_LOSS,.signal-box.TRAIL_STOP,
.signal-box.EXTREME_OB,.signal-box.EMA10_BREAK,.signal-box.BB_BREAK,.signal-box.REGIME_EXIT{
  background:rgba(239,68,68,.12);color:var(--red);border:2px solid var(--red);
  animation:glow-r .8s infinite alternate}
.signal-box.HOLD{background:var(--bg-card);color:var(--text-dim);border:1px solid var(--border)}
@keyframes glow-g{from{box-shadow:0 0 4px rgba(34,197,94,.3)}to{box-shadow:0 0 16px rgba(34,197,94,.5)}}
@keyframes glow-r{from{box-shadow:0 0 4px rgba(239,68,68,.3)}to{box-shadow:0 0 16px rgba(239,68,68,.5)}}

/* Conditions */
.cond-row{display:flex;justify-content:space-between;align-items:center;
  padding:5px 8px;margin:2px 0;border-radius:5px;font-size:.78rem;
  background:rgba(10,24,48,.5);border:1px solid var(--border)}
.cond-row .label{color:var(--text)}
.cond-row .status{font-family:'JetBrains Mono',monospace;font-weight:600}
.cond-row .status.ok{color:var(--green)}
.cond-row .status.ng{color:var(--text-dim)}

/* Metrics grid */
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.metric{background:var(--bg-card);border:1px solid var(--border);border-radius:6px;
  padding:8px;text-align:center}
.metric .lbl{font-size:.65rem;color:var(--text-dim);margin-bottom:3px;text-transform:uppercase}
.metric .val{font-size:.9rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.pos{color:var(--green)}.neg{color:var(--red)}.neu{color:var(--cyan)}

/* Right Panel */
.right{display:flex;flex-direction:column;gap:0;overflow:hidden}
.right-inner{flex:1;padding:12px;display:flex;flex-direction:column;gap:10px;overflow-y:auto}

/* Indicator Gauges */
.gauge-row{display:flex;align-items:center;gap:8px;padding:6px 0;
  border-bottom:1px solid rgba(14,34,68,.5)}
.gauge-label{width:100px;font-size:.78rem;color:var(--text);flex-shrink:0}
.gauge-track{flex:1;background:var(--bg-card);border-radius:4px;height:18px;
  overflow:hidden;border:1px solid var(--border);position:relative}
.gauge-fill{height:100%;border-radius:4px;transition:width .6s ease}
.gauge-thresh{position:absolute;top:0;bottom:0;width:2px;z-index:2}
.gauge-val{width:60px;text-align:right;font-size:.78rem;font-family:'JetBrains Mono',monospace;
  color:var(--text-bright);flex-shrink:0}

/* Trailing Stop Card */
.trail-card{background:linear-gradient(135deg,rgba(6,182,212,.06),rgba(59,130,246,.06));
  border:1px solid rgba(6,182,212,.2);border-radius:10px;padding:14px}
.trail-row{display:flex;justify-content:space-between;padding:4px 0;font-size:.82rem}
.trail-row .k{color:var(--text-dim)}
.trail-row .v{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-bright)}

/* Trade Log */
.trade-item{display:flex;justify-content:space-between;align-items:center;
  padding:5px 8px;margin:2px 0;border-radius:5px;font-size:.78rem;
  background:var(--bg-card);border:1px solid var(--border)}
.trade-item .type{font-weight:700;font-family:'JetBrains Mono',monospace}
.trade-item .type.buy{color:var(--green)}
.trade-item .type.sell{color:var(--red)}
.trade-item .pnl{font-family:'JetBrains Mono',monospace;font-weight:600}

/* Responsive */
@media(max-width:900px){
  .main{grid-template-columns:1fr}
  .left{border-right:none;border-bottom:1px solid var(--border)}
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1><span>QUANT</span> v3.0 Multi-Coin Dashboard</h1>
  <div class="live-badge"><div class="live-dot"></div>LIVE</div>
</div>

<!-- Coin Tabs -->
<div class="coin-tabs" id="coinTabs"></div>

<!-- Main -->
<div class="main">
  <!-- Left Panel -->
  <div class="left">
    <div class="card">
      <div class="card-title" id="coinTitle">BITCOIN</div>
      <div class="price-display">
        <div class="price-main flat" id="priceMain">Loading...</div>
        <div class="price-change" id="priceChange"></div>
      </div>
      <div style="text-align:center">
        <span class="regime NORMAL" id="regime">NORMAL MODE</span>
      </div>
      <div class="signal-box HOLD" id="signalBox">HOLD</div>
      <div style="font-size:.68rem;color:var(--text-dim);text-align:center;margin-top:4px" id="updateTime">-</div>
    </div>

    <div class="card">
      <div class="card-title">Entry Conditions <span id="condCount" style="float:right;color:var(--cyan)">0/5</span></div>
      <div class="cond-row"><span class="label">SMA30 Trend</span><span class="status ng" id="c0">--</span></div>
      <div class="cond-row"><span class="label">MACD Golden Cross</span><span class="status ng" id="c1">--</span></div>
      <div class="cond-row"><span class="label" id="c2lbl">StochRSI < 80</span><span class="status ng" id="c2">--</span></div>
      <div class="cond-row"><span class="label">BB %b > 0.4</span><span class="status ng" id="c3">--</span></div>
      <div class="cond-row"><span class="label" id="c4lbl">Volume > 1.0x</span><span class="status ng" id="c4">--</span></div>
    </div>

    <div class="card">
      <div class="card-title">Portfolio</div>
      <div class="metrics">
        <div class="metric"><div class="lbl">Equity</div><div class="val neu" id="equity">$10,000</div></div>
        <div class="metric"><div class="lbl">P&L</div><div class="val neu" id="pnl">+0.00%</div></div>
        <div class="metric"><div class="lbl">Position</div><div class="val neu" id="posStatus">Cash</div></div>
        <div class="metric"><div class="lbl">Entry Price</div><div class="val neu" id="entryPx">-</div></div>
      </div>
    </div>
  </div>

  <!-- Right Panel -->
  <div class="right">
    <div class="right-inner">
      <!-- Indicator Gauges -->
      <div class="card">
        <div class="card-title">Indicator Gauges</div>
        <div class="gauge-row">
          <span class="gauge-label">ADX Strength</span>
          <div class="gauge-track">
            <div class="gauge-fill" id="gf-adx" style="background:var(--orange);width:0%"></div>
            <div class="gauge-thresh" style="left:25%;background:var(--orange);opacity:.5"></div>
          </div>
          <span class="gauge-val" id="gv-adx">-</span>
        </div>
        <div class="gauge-row">
          <span class="gauge-label">StochRSI %K</span>
          <div class="gauge-track">
            <div class="gauge-fill" id="gf-sr" style="background:var(--purple);width:50%"></div>
            <div class="gauge-thresh" style="left:85%;background:var(--red);opacity:.5"></div>
            <div class="gauge-thresh" style="left:20%;background:var(--green);opacity:.5"></div>
          </div>
          <span class="gauge-val" id="gv-sr">-</span>
        </div>
        <div class="gauge-row">
          <span class="gauge-label">BB %b</span>
          <div class="gauge-track">
            <div class="gauge-fill" id="gf-bb" style="background:var(--blue);width:50%"></div>
            <div class="gauge-thresh" style="left:40%;background:var(--yellow);opacity:.5"></div>
          </div>
          <span class="gauge-val" id="gv-bb">-</span>
        </div>
        <div class="gauge-row">
          <span class="gauge-label">Volume Ratio</span>
          <div class="gauge-track">
            <div class="gauge-fill" id="gf-vl" style="background:var(--green);width:33%"></div>
            <div class="gauge-thresh" style="left:33%;background:var(--yellow);opacity:.5"></div>
          </div>
          <span class="gauge-val" id="gv-vl">-</span>
        </div>
        <div class="gauge-row">
          <span class="gauge-label">MACD Gap</span>
          <div class="gauge-track">
            <div class="gauge-fill" id="gf-mc" style="background:var(--green);width:50%"></div>
          </div>
          <span class="gauge-val" id="gv-mc">-</span>
        </div>
      </div>

      <!-- ADX Regime Detail -->
      <div class="card">
        <div class="card-title">ADX Regime Analysis</div>
        <div class="metrics" style="grid-template-columns:1fr 1fr 1fr">
          <div class="metric"><div class="lbl">ADX</div><div class="val" id="adxVal" style="color:var(--orange)">-</div></div>
          <div class="metric"><div class="lbl">DI+</div><div class="val" id="diPlus" style="color:var(--green)">-</div></div>
          <div class="metric"><div class="lbl">DI-</div><div class="val" id="diMinus" style="color:var(--red)">-</div></div>
        </div>
        <div style="margin-top:8px;font-size:.75rem;padding:6px 10px;border-radius:6px;text-align:center" id="regimeExplain">-</div>
      </div>

      <!-- Trailing Stop -->
      <div class="trail-card">
        <div class="card-title" style="color:var(--cyan);border-color:rgba(6,182,212,.2)">Trailing Stop Status</div>
        <div class="trail-row"><span class="k">Highest Price</span><span class="v" id="trailHigh">-</span></div>
        <div class="trail-row"><span class="k">Trailing Stop</span><span class="v" id="trailStop">-</span></div>
        <div class="trail-row"><span class="k">ATR (14)</span><span class="v" id="atrVal">-</span></div>
        <div class="trail-row"><span class="k">Distance to Stop</span><span class="v" id="trailDist">-</span></div>
      </div>

      <!-- Trade Log -->
      <div class="card">
        <div class="card-title">Recent Trades</div>
        <div id="tradeLog" style="max-height:200px;overflow-y:auto">
          <div style="color:var(--text-dim);font-size:.78rem;text-align:center;padding:12px">No trades yet</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const COINS = %s;
let activeTicker = 'BTC-USD';
let allData = {};

// Build coin tabs
function buildTabs() {
  const container = document.getElementById('coinTabs');
  container.innerHTML = '';
  COINS.forEach(c => {
    const tab = document.createElement('div');
    tab.className = 'coin-tab' + (c.ticker === activeTicker ? ' active' : '');
    tab.dataset.ticker = c.ticker;
    tab.innerHTML = `
      <span class="dot" style="background:${c.color}"></span>
      <span>${c.symbol}</span>
      <span class="mini-price" id="mp-${c.ticker}">--</span>
      <span class="mini-sig HOLD" id="ms-${c.ticker}"></span>
    `;
    tab.onclick = () => selectCoin(c.ticker);
    container.appendChild(tab);
  });
}

function selectCoin(ticker) {
  activeTicker = ticker;
  document.querySelectorAll('.coin-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.ticker === ticker);
  });
  if (allData[ticker]) updateUI(ticker, allData[ticker]);
}

// Signal display text
const SIG_TEXT = {
  BUY: 'BUY SIGNAL',
  HOLD: 'HOLD',
  EXIT: 'SELL (Exit)',
  STOP_LOSS: 'STOP LOSS',
  TRAIL_STOP: 'TRAILING STOP',
  EXTREME_OB: 'EXTREME OVERBOUGHT',
  EMA10_BREAK: 'EMA10 BREAK',
  BB_BREAK: 'BB BREAKDOWN',
  REGIME_EXIT: 'REGIME EXIT',
  SELL: 'SELL'
};

function getSignalClass(sig) {
  if (sig === 'BUY') return 'BUY';
  if (sig === 'HOLD') return 'HOLD';
  return 'EXIT';
}

function updateUI(ticker, s) {
  const info = COINS.find(c => c.ticker === ticker);
  if (!info) return;

  // Title
  document.getElementById('coinTitle').textContent = info.name.toUpperCase() + ' (' + info.symbol + ')';

  // Price
  const px = s.price || 0;
  const prev = s.prev_price || 0;
  const priceEl = document.getElementById('priceMain');
  priceEl.textContent = px ? '$' + px.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '--';
  priceEl.className = 'price-main ' + (px > prev ? 'up' : px < prev ? 'down' : 'flat');

  // Regime
  const regEl = document.getElementById('regime');
  regEl.textContent = s.regime === 'BULL' ? 'BULL MODE (Relaxed Exits)' : 'NORMAL MODE';
  regEl.className = 'regime ' + (s.regime || 'NORMAL');

  // Signal
  const sig = s.signal || 'HOLD';
  const sigEl = document.getElementById('signalBox');
  sigEl.textContent = SIG_TEXT[sig] || sig;
  sigEl.className = 'signal-box ' + getSignalClass(sig);

  document.getElementById('updateTime').textContent = s.last_update ? 'Updated: ' + s.last_update : '';

  // Conditions
  const conds = s.conditions || [false, false, false, false, false];
  const labels = s.condition_labels || ['--', '--', '--', '--', '--'];
  const okCount = conds.filter(Boolean).length;
  document.getElementById('condCount').textContent = okCount + '/5';

  // Update condition labels for regime
  const bull = s.regime === 'BULL';
  document.getElementById('c2lbl').textContent = bull ? 'StochRSI < 90 (Bull)' : 'StochRSI < 80';
  document.getElementById('c4lbl').textContent = bull ? 'Volume > 0.8x (Bull)' : 'Volume > 1.0x';

  for (let i = 0; i < 5; i++) {
    const el = document.getElementById('c' + i);
    el.textContent = labels[i];
    el.className = 'status ' + (conds[i] ? 'ok' : 'ng');
  }

  // Portfolio
  const eq = s.equity || INITIAL;
  const pnl = s.pnl_pct || 0;
  document.getElementById('equity').textContent = '$' + eq.toLocaleString(undefined, {minimumFractionDigits: 2});
  const pnlEl = document.getElementById('pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
  pnlEl.className = 'val ' + (pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : 'neu');
  document.getElementById('posStatus').textContent = s.position > 0 ? info.symbol + ' Holding' : 'Cash';
  document.getElementById('entryPx').textContent = s.entry_px > 0 ? '$' + s.entry_px.toLocaleString(undefined, {maximumFractionDigits: 2}) : '-';

  // Gauges
  const adx = s.adx || 0;
  const sr = s.stochrsi || 50;
  const bb = s.bb_pct || 0.5;
  const vl = Math.min(s.vol_ratio || 1, 3);
  const mdiff = (s.macd || 0) - (s.macd_sig || 0);

  setGauge('adx', adx / 60, adx.toFixed(1), adx > 25 ? 'var(--orange)' : 'var(--text-dim)');
  setGauge('sr', sr / 100, sr.toFixed(1), sr > 85 ? 'var(--red)' : sr < 20 ? 'var(--green)' : 'var(--purple)');
  setGauge('bb', Math.max(0, Math.min(bb, 1.5)) / 1.5, bb.toFixed(3), bb > 0.4 ? 'var(--green)' : 'var(--yellow)');
  setGauge('vl', vl / 3, (s.vol_ratio || 1).toFixed(2) + 'x', vl > 1 ? 'var(--green)' : 'var(--yellow)');
  setGauge('mc', 0.5 + Math.max(-0.5, Math.min(0.5, mdiff / 1000)), mdiff.toFixed(1), mdiff >= 0 ? 'var(--green)' : 'var(--red)');

  // ADX Detail
  document.getElementById('adxVal').textContent = adx.toFixed(1);
  document.getElementById('diPlus').textContent = (s.di_plus || 0).toFixed(1);
  document.getElementById('diMinus').textContent = (s.di_minus || 0).toFixed(1);

  const regExplainEl = document.getElementById('regimeExplain');
  if (adx > 25 && (s.di_plus || 0) > (s.di_minus || 0)) {
    regExplainEl.textContent = 'Strong Bullish Trend -> Relaxed exits, Trailing Stop active';
    regExplainEl.style.background = 'rgba(34,197,94,.1)';
    regExplainEl.style.color = 'var(--green)';
  } else if (adx > 25) {
    regExplainEl.textContent = 'Strong Bearish Trend -> Tight exits, Normal stops';
    regExplainEl.style.background = 'rgba(239,68,68,.1)';
    regExplainEl.style.color = 'var(--red)';
  } else {
    regExplainEl.textContent = 'Weak/Sideways -> Normal strategy mode';
    regExplainEl.style.background = 'rgba(71,85,105,.1)';
    regExplainEl.style.color = 'var(--text-dim)';
  }

  // Trailing Stop
  document.getElementById('trailHigh').textContent = s.trailing_high > 0 ? '$' + s.trailing_high.toLocaleString(undefined, {maximumFractionDigits: 2}) : '-';
  document.getElementById('trailStop').textContent = s.trailing_stop > 0 ? '$' + s.trailing_stop.toLocaleString(undefined, {maximumFractionDigits: 2}) : '-';
  document.getElementById('atrVal').textContent = s.atr > 0 ? '$' + s.atr.toLocaleString(undefined, {maximumFractionDigits: 2}) : '-';

  if (s.trailing_stop > 0 && px > 0) {
    const dist = ((px - s.trailing_stop) / px * 100).toFixed(2);
    const distEl = document.getElementById('trailDist');
    distEl.textContent = dist + '%';
    distEl.style.color = dist < 2 ? 'var(--red)' : dist < 5 ? 'var(--yellow)' : 'var(--green)';
  } else {
    document.getElementById('trailDist').textContent = '-';
  }

  // Trade Log
  const logEl = document.getElementById('tradeLog');
  const trades = s.trade_log || [];
  if (trades.length === 0) {
    logEl.innerHTML = '<div style="color:var(--text-dim);font-size:.78rem;text-align:center;padding:12px">No trades yet</div>';
  } else {
    logEl.innerHTML = trades.slice().reverse().map(t => {
      const isBuy = t.type === 'BUY';
      const pnlColor = t.pnl > 0 ? 'var(--green)' : t.pnl < 0 ? 'var(--red)' : 'var(--text-dim)';
      return `<div class="trade-item">
        <span class="type ${isBuy ? 'buy' : 'sell'}">${t.type}</span>
        <span style="color:var(--text-dim)">${t.time}</span>
        <span class="pnl" style="color:${pnlColor}">${t.pnl >= 0 ? '+' : ''}${t.pnl}%</span>
      </div>`;
    }).join('');
  }
}

function setGauge(id, pct, val, color) {
  const fill = document.getElementById('gf-' + id);
  const valEl = document.getElementById('gv-' + id);
  if (fill) {
    fill.style.width = Math.max(0, Math.min(100, pct * 100)) + '%';
    fill.style.background = color;
  }
  if (valEl) valEl.textContent = val;
}

// SSE
const INITIAL = 10000;
const es = new EventSource('/api/stream');
es.onmessage = e => {
  try {
    const data = JSON.parse(e.data);
    allData = data;

    // Update mini prices on all tabs
    COINS.forEach(c => {
      const s = data[c.ticker];
      if (!s) return;
      const mpEl = document.getElementById('mp-' + c.ticker);
      if (mpEl) mpEl.textContent = s.price ? '$' + s.price.toLocaleString(undefined, {maximumFractionDigits: 0}) : '--';
      const msEl = document.getElementById('ms-' + c.ticker);
      if (msEl) {
        const sig = s.signal || 'HOLD';
        msEl.textContent = sig === 'HOLD' ? '' : sig.substring(0, 3);
        msEl.className = 'mini-sig ' + getSignalClass(sig);
      }
    });

    // Update main UI for active coin
    if (data[activeTicker]) updateUI(activeTicker, data[activeTicker]);
  } catch(err) {}
};
es.onerror = () => {
  setInterval(() => fetch('/api/state').then(r => r.json()).then(data => {
    allData = data;
    if (data[activeTicker]) updateUI(activeTicker, data[activeTicker]);
  }), 10000);
};

// Init
buildTabs();
fetch('/api/state').then(r => r.json()).then(data => {
  allData = data;
  COINS.forEach(c => {
    const s = data[c.ticker];
    if (!s) return;
    const mpEl = document.getElementById('mp-' + c.ticker);
    if (mpEl) mpEl.textContent = s.price ? '$' + s.price.toLocaleString(undefined, {maximumFractionDigits: 0}) : '--';
  });
  if (data[activeTicker]) updateUI(activeTicker, data[activeTicker]);
});
</script>
</body>
</html>
""" % json.dumps([{
    'ticker': tk,
    'name': COIN_INFO[tk]['name'],
    'symbol': COIN_INFO[tk]['symbol'],
    'color': COIN_INFO[tk]['color']
} for tk in TICKERS])

if __name__ == '__main__':
    print("=" * 55)
    print("  QUANT v3.0 Multi-Coin Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    threading.Thread(target=live_loop, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

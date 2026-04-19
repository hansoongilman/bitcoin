"""
web_dashboard.py — BTC 퀀트 전략 v2.0 웹 대시보드 (수정판)
실행: python web_dashboard.py → http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template_string, Response
import yfinance as yf
import pandas as pd
import numpy as np
import ta, json, time, threading, warnings
from datetime import datetime

warnings.filterwarnings('ignore')
app = Flask(__name__)

# ── 공유 상태 ──────────────────────────────────────────────
state = {
    'price':0.0,'signal':'HOLD','sma30':0.0,'macd':0.0,
    'macd_sig':0.0,'stochrsi':50.0,'bb_pct':0.5,'atr':0.0,
    'vol_ratio':1.0,'equity':10000.0,'pnl_pct':0.0,
    'position':0.0,'cash':10000.0,'entry_px':0.0,
    'last_update':'로딩 중...','conditions':[False]*5,
    'replay_ready':False,'replay_data':[],
}
lock = threading.Lock()
INITIAL = 10000.0

# ── 지표 계산 ──────────────────────────────────────────────
def calc_indicators(df):
    df = df.copy()
    df['SMA30']    = ta.trend.SMAIndicator(df['Close'], window=30).sma_indicator()
    m              = ta.trend.MACD(df['Close'], 12, 21, 7)
    df['MACD']     = m.macd()
    df['MACD_Sig'] = m.macd_signal()
    df['StochRSI'] = ta.momentum.StochasticOscillator(df['High'],df['Low'],df['Close']).stoch()
    bb             = ta.volatility.BollingerBands(df['Close'], 20)
    df['BB_Pct']   = bb.bollinger_pband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['ATR']      = ta.volatility.AverageTrueRange(df['High'],df['Low'],df['Close'],14).average_true_range()
    vma            = df['Volume'].rolling(20).mean()
    df['Vol_Ratio']= df['Volume'] / vma
    df.dropna(inplace=True)
    return df

def entry_ok(r):
    return (float(r['Close'])>float(r['SMA30']) and float(r['MACD'])>float(r['MACD_Sig'])
            and float(r['StochRSI'])<80 and float(r['BB_Pct'])>0.4 and float(r['Vol_Ratio'])>1.0)

def exit_signal(r, ep):
    p = float(r['Close'])
    if p <= ep*0.95: return 'SL'
    if p<float(r['SMA30']) or float(r['MACD'])<float(r['MACD_Sig']) or float(r['StochRSI'])>85: return 'EXIT'
    return None

# ── 실시간 체크 스레드 (1분) ──────────────────────────────
def live_loop():
    while True:
        try:
            df = yf.download('BTC-USD', period='5d', interval='1h', progress=False)
            df.dropna(inplace=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            if len(df) < 40: time.sleep(60); continue
            df = calc_indicators(df)
            row = df.iloc[-1]
            with lock:
                px = float(row['Close'])
                state.update({
                    'price':px, 'sma30':float(row['SMA30']),
                    'macd':float(row['MACD']), 'macd_sig':float(row['MACD_Sig']),
                    'stochrsi':float(row['StochRSI']), 'bb_pct':float(row['BB_Pct']),
                    'atr':float(row['ATR']), 'vol_ratio':float(row['Vol_Ratio']),
                    'last_update':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
                state['conditions'] = [
                    px>state['sma30'], state['macd']>state['macd_sig'],
                    state['stochrsi']<80, state['bb_pct']>0.4, state['vol_ratio']>1.0
                ]
                pos,ep,cash = state['position'],state['entry_px'],state['cash']
                if pos > 0:
                    sig = exit_signal(row, ep)
                    if sig:
                        state['signal'] = 'SL' if sig=='SL' else 'SELL'
                        state['cash'] = pos*px; state['position']=0.0; state['entry_px']=0.0
                    else: state['signal']='HOLD'
                elif entry_ok(row):
                    state['signal']='BUY'
                    state['position']=cash/px; state['entry_px']=px; state['cash']=0.0
                else: state['signal']='HOLD'
                eq = state['cash']+state['position']*px
                state['equity']=eq; state['pnl_pct']=(eq/INITIAL-1)*100
        except Exception as e: print(f"live err: {e}")
        time.sleep(60)

# ── 리플레이 스레드 ───────────────────────────────────────
def replay_loop():
    try:
        print("리플레이 데이터 로딩중 (729d 1h)...")
        df = yf.download('BTC-USD', period='729d', interval='1h', progress=False)
        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = calc_indicators(df)
        cash_r=INITIAL; pos_r=0.0; ep_r=0.0
        rows = []
        init_px = float(df['Close'].iloc[0])
        for ts, row in df.iterrows():
            px = float(row['Close'])
            if pos_r>0:
                s = exit_signal(row, ep_r)
                if s: cash_r+=pos_r*px; pos_r=0.0; ep_r=0.0
            if pos_r==0 and entry_ok(row):
                pos_r=cash_r/px; ep_r=px; cash_r=0.0
            eq = cash_r+pos_r*px
            # 수익률 % (둘 다 0에서 시작)
            strat_pct = round((eq/INITIAL-1)*100, 3)
            bh_pct    = round((px/init_px-1)*100, 3)
            dt = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts,'strftime') else str(ts)
            rows.append([dt, strat_pct, bh_pct, round(px,2)])
        with lock:
            state['replay_data']  = rows
            state['replay_ready'] = True
        print(f"리플레이 완료: {len(rows)}건")
    except Exception as e: print(f"replay err: {e}")

# ── API ──────────────────────────────────────────────────
@app.route('/api/state')
def api_state():
    with lock:
        s = {k:v for k,v in state.items() if k not in ('replay_data',)}
    return jsonify(s)

@app.route('/api/replay')
def api_replay():
    with lock:
        data  = state['replay_data']
        ready = state['replay_ready']
    return jsonify({'ready': ready, 'data': data})

@app.route('/api/stream')
def api_stream():
    def gen():
        while True:
            with lock:
                s = {k:v for k,v in state.items() if k not in ('replay_data',)}
            yield f"data: {json.dumps(s)}\n\n"
            time.sleep(1)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/')
def index():
    return render_template_string(PAGE)

# ── HTML ─────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>BTC 퀀트 v2.0</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#020818;color:#93c5fd;font-family:'Noto Sans KR',sans-serif}
.hdr{background:linear-gradient(135deg,#0c1a3e,#061025);border-bottom:2px solid #1e40af;
  padding:14px 24px;display:flex;align-items:center;justify-content:space-between}
.hdr h1{font-size:1.3rem;font-weight:700;color:#dbeafe}
.badge{padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:700;
  background:#064e3b;color:#34d399;border:1px solid #065f46;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.wrap{display:grid;grid-template-columns:300px 1fr;gap:12px;padding:12px;
  height:calc(100vh - 60px)}
.left{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
.card{background:#061025;border:1px solid #0e2244;border-radius:10px;padding:16px}
.card h2{font-size:.78rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;
  margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #0e2244}
.price{font-size:2.2rem;font-weight:700;font-family:'JetBrains Mono',monospace;text-align:center;padding:10px 0}
.price.up{color:#22c55e}.price.down{color:#ef4444}.price.neu{color:#fff}
.sig{padding:12px;border-radius:8px;text-align:center;font-size:1.1rem;font-weight:700;
  margin:8px 0;transition:.3s}
.BUY{background:#065f46;color:#34d399;border:2px solid #22c55e;animation:glow-g 1s infinite alternate}
.SELL,.SL{background:#7f1d1d;color:#fca5a5;border:2px solid #ef4444;animation:glow-r 1s infinite alternate}
.HOLD{background:#0a1830;color:#475569;border:1px solid #1e3a5f}
@keyframes glow-g{from{box-shadow:0 0 4px #22c55e}to{box-shadow:0 0 20px #22c55e}}
@keyframes glow-r{from{box-shadow:0 0 4px #ef4444}to{box-shadow:0 0 20px #ef4444}}
.ind{display:flex;justify-content:space-between;align-items:center;
  padding:6px 10px;border-radius:6px;margin:3px 0;font-size:.82rem;background:#0a1830;border:1px solid #0e2244}
.ok{color:#22c55e;font-weight:700}.ng{color:#475569}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.m{background:#0a1830;border:1px solid #0e2244;border-radius:6px;padding:10px;text-align:center}
.m .lbl{font-size:.7rem;color:#475569;margin-bottom:3px}
.m .val{font-size:1rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.pos{color:#22c55e}.neg{color:#ef4444}.neu2{color:#60a5fa}
/* right */
.right{display:flex;flex-direction:column;gap:10px;overflow:hidden;min-height:0}
.tabs{display:flex;gap:6px}
.tab{padding:5px 14px;border-radius:6px;cursor:pointer;font-size:.8rem;border:1px solid #0e2244;
  background:#0a1830;color:#475569}
.tab.on{background:#1e40af;color:#fff;border-color:#3b82f6}
.panels{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:10px;min-height:0}
.panel{background:#061025;border:1px solid #0e2244;border-radius:10px;padding:14px;
  display:flex;flex-direction:column;min-height:0}
.panel h3{font-size:.82rem;color:#60a5fa;margin-bottom:8px;flex-shrink:0}
.cbox{flex:1;position:relative;min-height:0}
/* gauge */
.grow{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #0e2244}
.glbl{width:100px;font-size:.8rem;color:#93c5fd;flex-shrink:0}
.gtrk{flex:1;background:#0a1830;border-radius:4px;height:16px;overflow:hidden;border:1px solid #1e3a5f}
.gfil{height:100%;border-radius:4px;transition:width .6s}
.gnum{width:55px;text-align:right;font-size:.8rem;font-family:monospace;color:#dbeafe;flex-shrink:0}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}
</style>
</head>
<body>
<div class="hdr">
  <h1>BTC 퀀트 전략 v2.0 &nbsp; 실시간 대시보드</h1>
  <span class="badge">● LIVE</span>
</div>
<div class="wrap">
  <!-- LEFT -->
  <div class="left">
    <div class="card">
      <h2>실시간 시세</h2>
      <div class="price neu" id="px">로딩...</div>
      <div style="font-size:.75rem;color:#475569;text-align:center" id="upd">-</div>
      <div class="sig HOLD" id="sig">대기 중 (HOLD)</div>
    </div>
    <div class="card">
      <h2>진입 조건 (5/5 충족 시 BUY)</h2>
      <div class="ind"><span>SMA30 추세</span><span class="ng" id="c0">--</span></div>
      <div class="ind"><span>MACD 골든크로스</span><span class="ng" id="c1">--</span></div>
      <div class="ind"><span>StochRSI &lt; 80</span><span class="ng" id="c2">--</span></div>
      <div class="ind"><span>BB %b &gt; 0.4</span><span class="ng" id="c3">--</span></div>
      <div class="ind"><span>거래량 확인</span><span class="ng" id="c4">--</span></div>
    </div>
    <div class="card">
      <h2>수익 현황</h2>
      <div class="metrics">
        <div class="m"><div class="lbl">전략 자산</div><div class="val neu2" id="eq">$10,000</div></div>
        <div class="m"><div class="lbl">수익률</div><div class="val neu2" id="pnl">+0.00%</div></div>
        <div class="m"><div class="lbl">포지션</div><div class="val neu2" id="pos">현금</div></div>
        <div class="m"><div class="lbl">진입가</div><div class="val neu2" id="ent">-</div></div>
      </div>
    </div>
  </div>
  <!-- RIGHT -->
  <div class="right">
    <div class="tabs">
      <div class="tab on" onclick="showTab('replay',this)">백테스트 리플레이 (729일)</div>
      <div class="tab"    onclick="showTab('live',this)">실시간 지표 현황</div>
    </div>
    <!-- 리플레이 패널 -->
    <div class="panels" id="pane-replay">
      <div class="panel" style="flex:2">
        <h3 id="replay-hdr">수익률 곡선 (%) — 전략 vs 단순보유(B&H)&nbsp;<span id="rstat" style="color:#60a5fa;font-family:monospace;font-size:.78rem"></span></h3>
        <div class="cbox"><canvas id="cEq"></canvas></div>
      </div>
      <div class="panel" style="flex:1">
        <h3>BTC 실제 가격 ($)</h3>
        <div class="cbox"><canvas id="cPx"></canvas></div>
      </div>
    </div>
    <!-- 지표 패널 -->
    <div class="panels" id="pane-live" style="display:none">
      <div class="panel" style="flex:1">
        <h3>지표 게이지 &nbsp;<span id="lv-tick" style="color:#22c55e;font-family:monospace;font-size:.75rem"></span></h3>
        <div class="grow"><span class="glbl">StochRSI %K</span><div class="gtrk"><div class="gfil" id="gf-sr" style="background:#a78bfa;width:50%"></div></div><span class="gnum" id="gn-sr">-</span></div>
        <div class="grow"><span class="glbl">BB %b</span><div class="gtrk"><div class="gfil" id="gf-bb" style="background:#3b82f6;width:50%"></div></div><span class="gnum" id="gn-bb">-</span></div>
        <div class="grow"><span class="glbl">거래량 비율</span><div class="gtrk"><div class="gfil" id="gf-vl" style="background:#22c55e;width:33%"></div></div><span class="gnum" id="gn-vl">-</span></div>
        <div class="grow"><span class="glbl">MACD 차이</span><div class="gtrk"><div class="gfil" id="gf-mc" style="background:#34d399;width:50%"></div></div><span class="gnum" id="gn-mc">-</span></div>
        <div class="grow"><span class="glbl">조건 달성률</span><div class="gtrk"><div class="gfil" id="gf-cd" style="background:#22c55e;width:0%"></div></div><span class="gnum" id="gn-cd">0/5</span></div>
        <div class="grid3" style="margin-top:12px">
          <div class="m"><div class="lbl">StochRSI</div><div class="val" id="lv-sr" style="color:#a78bfa">-</div></div>
          <div class="m"><div class="lbl">BB %b</div><div class="val" id="lv-bb" style="color:#60a5fa">-</div></div>
          <div class="m"><div class="lbl">ATR ($)</div><div class="val" id="lv-at" style="color:#fbbf24">-</div></div>
          <div class="m"><div class="lbl">MACD</div><div class="val" id="lv-mc" style="color:#34d399">-</div></div>
          <div class="m"><div class="lbl">MACD Sig</div><div class="val" id="lv-ms" style="color:#f87171">-</div></div>
          <div class="m"><div class="lbl">거래량x</div><div class="val" id="lv-vl" style="color:#fb923c">-</div></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
// ─── Chart 초기화 ─────────────────────────────────────
const mkScale = (pct) => ({
  type:'linear',
  ticks:{ color:'#93c5fd', font:{size:10},
    callback: pct ? v=>(v>=0?'+':'')+v.toFixed(1)+'%' : v=>'$'+v.toLocaleString() },
  grid:{ color:'#0e2244' }
});
const mkXScale = () => ({
  type:'category',
  ticks:{ color:'#475569', font:{size:9}, maxTicksLimit:12, maxRotation:0 },
  grid:{ color:'#0e2244' }
});

const COMMON = {
  animation:false, responsive:true, maintainAspectRatio:false,
  plugins:{ legend:{ labels:{ color:'#93c5fd', font:{size:11} } } }
};

const cEq = new Chart(document.getElementById('cEq').getContext('2d'), {
  type:'line',
  data:{ labels:[], datasets:[
    { label:'전략 수익률(%)', data:[], borderColor:'#3b82f6', borderWidth:2.5,
      pointRadius:0, fill:false, tension:0.1 },
    { label:'B&H 수익률(%)', data:[], borderColor:'#fbbf24', borderWidth:1.8,
      borderDash:[5,3], pointRadius:0, fill:false, tension:0.1 }
  ]},
  options:{ ...COMMON, scales:{ x:mkXScale(), y:mkScale(true) } }
});

const cPx = new Chart(document.getElementById('cPx').getContext('2d'), {
  type:'line',
  data:{ labels:[], datasets:[
    { label:'BTC 가격($)', data:[], borderColor:'#a78bfa', borderWidth:1.5,
      pointRadius:0, fill:false, tension:0.1 }
  ]},
  options:{ ...COMMON, scales:{ x:mkXScale(), y:mkScale(false) } }
});

// ─── 리플레이 ──────────────────────────────────────────
let rIdx = 0, rData = [];

async function loadReplay() {
  const res = await fetch('/api/replay');
  const j   = await res.json();
  if (!j.ready || !j.data.length) { setTimeout(loadReplay, 2000); return; }
  rData = j.data;   // [date, strat_pct, bh_pct, price]
  rIdx  = 0;
  stepReplay();
}

function stepReplay() {
  if (rIdx >= rData.length) return;
  const STEP = 30, end = Math.min(rIdx + STEP, rData.length);
  for (let i = rIdx; i < end; i++) {
    const [dt, sp, bp, px] = rData[i];
    cEq.data.labels.push(dt.slice(5,13));
    cEq.data.datasets[0].data.push(sp);
    cEq.data.datasets[1].data.push(bp);
    cPx.data.labels.push(dt.slice(5,13));
    cPx.data.datasets[0].data.push(px);
  }
  // 최근 600개만 유지
  if (cEq.data.labels.length > 600) {
    cEq.data.labels.splice(0, STEP);
    cEq.data.datasets.forEach(d => d.data.splice(0, STEP));
    cPx.data.labels.splice(0, STEP);
    cPx.data.datasets[0].data.splice(0, STEP);
  }
  cEq.update('none');
  cPx.update('none');
  rIdx = end;
  const sp = rData[end-1][1], bp = rData[end-1][2];
  document.getElementById('rstat').textContent =
    `${rData[0][0].slice(0,10)} ~ ${rData[end-1][0].slice(0,10)}  |  전략 ${sp>=0?'+':''}${sp.toFixed(1)}%  |  B&H ${bp>=0?'+':''}${bp.toFixed(1)}%`;
  setTimeout(stepReplay, 40);
}

// ─── SSE 실시간 갱신 ───────────────────────────────────
let prevPx = 0;

function applyState(s) {
  const px = +s.price || 0;
  const pxEl = document.getElementById('px');
  pxEl.textContent = px ? '$'+px.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '--';
  pxEl.className = 'price ' + (px > prevPx ? 'up' : px < prevPx ? 'down' : 'neu');
  prevPx = px;

  document.getElementById('upd').textContent = s.last_update || '';

  const sig = s.signal || 'HOLD';
  const sigEl = document.getElementById('sig');
  sigEl.className = 'sig ' + sig;
  const ST = {BUY:'🚀 BUY SIGNAL — 매수 조건 달성!', SELL:'⚡ SELL SIGNAL — 청산', SL:'🛑 STOP-LOSS 발동!', HOLD:'대기 중 (HOLD)'};
  sigEl.textContent = ST[sig] || 'HOLD';

  const ci = s.conditions || [false,false,false,false,false];
  const vt = [
    s.price>s.sma30 ? `OK (${(+s.price).toFixed(0)}>${(+s.sma30).toFixed(0)})` : `-- (<SMA)`,
    s.macd>s.macd_sig ? 'OK' : '--',
    s.stochrsi<80 ? `OK (${(+s.stochrsi).toFixed(1)})` : `-- (${(+s.stochrsi).toFixed(1)})`,
    s.bb_pct>0.4 ? `OK (${(+s.bb_pct).toFixed(2)})` : `-- (${(+s.bb_pct).toFixed(2)})`,
    s.vol_ratio>1 ? `OK (${(+s.vol_ratio).toFixed(2)}x)` : `-- (${(+s.vol_ratio).toFixed(2)}x)`,
  ];
  ['c0','c1','c2','c3','c4'].forEach((id,i) => {
    const el = document.getElementById(id);
    el.textContent = vt[i]; el.className = ci[i]?'ok':'ng';
  });

  const eq = +s.equity||10000, pnl = +s.pnl_pct||0;
  document.getElementById('eq').textContent = '$'+eq.toLocaleString(undefined,{minimumFractionDigits:2});
  const pnlEl = document.getElementById('pnl');
  pnlEl.textContent = (pnl>=0?'+':'')+pnl.toFixed(2)+'%';
  pnlEl.className = 'val '+(pnl>0?'pos':pnl<0?'neg':'neu2');
  document.getElementById('pos').textContent = s.position>0?'BTC 보유':'현금';
  document.getElementById('ent').textContent = s.entry_px>0?'$'+(+s.entry_px).toLocaleString():'-';

  // 게이지
  const sr = +(s.stochrsi||50), bb = +(s.bb_pct||0.5), vl = Math.min(+(s.vol_ratio||1),3);
  const mdiff = (+(s.macd||0))-(+(s.macd_sig||0));
  const ok = ci.filter(Boolean).length;
  function gset(sid,nid,pct,val,col) {
    const f=document.getElementById('gf-'+sid), n=document.getElementById('gn-'+nid);
    if(f){ f.style.width=Math.max(0,Math.min(100,pct*100))+'%'; if(col)f.style.background=col; }
    if(n) n.textContent = val;
  }
  gset('sr','sr', sr/100, sr.toFixed(1), sr>80?'#ef4444':sr<20?'#22c55e':'#a78bfa');
  gset('bb','bb', bb,      bb.toFixed(2), bb>0.4?'#22c55e':'#f59e0b');
  gset('vl','vl', vl/3,    (+(s.vol_ratio||1)).toFixed(2)+'x', vl/3>0.33?'#22c55e':'#f59e0b');
  gset('mc','mc', Math.min(Math.abs(mdiff)/500,1), mdiff.toFixed(1), mdiff>=0?'#22c55e':'#ef4444');
  gset('cd','cd', ok/5,    ok+'/5', undefined);
  document.getElementById('lv-sr').textContent = sr.toFixed(1);
  document.getElementById('lv-bb').textContent = bb.toFixed(3);
  document.getElementById('lv-at').textContent = '$'+(+(s.atr||0)).toFixed(0);
  document.getElementById('lv-mc').textContent = (+(s.macd||0)).toFixed(1);
  document.getElementById('lv-ms').textContent = (+(s.macd_sig||0)).toFixed(1);
  document.getElementById('lv-vl').textContent = (+(s.vol_ratio||1)).toFixed(2)+'x';
  const tick = document.getElementById('lv-tick');
  if(tick) tick.textContent = '● '+new Date().toLocaleTimeString('ko-KR');
}

// SSE
const es = new EventSource('/api/stream');
es.onmessage = e => { try { applyState(JSON.parse(e.data)); } catch{} };
es.onerror = () => { setInterval(()=>fetch('/api/state').then(r=>r.json()).then(applyState), 30000); };
fetch('/api/state').then(r=>r.json()).then(applyState);

// 탭 전환
function showTab(name, el) {
  document.getElementById('pane-replay').style.display = name==='replay'?'flex':'none';
  document.getElementById('pane-live').style.display   = name==='live'  ?'flex':'none';
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
}

// 시작
loadReplay();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("="*50)
    print("  http://127.0.0.1:5000")
    print("="*50)
    threading.Thread(target=live_loop,   daemon=True).start()
    threading.Thread(target=replay_loop, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

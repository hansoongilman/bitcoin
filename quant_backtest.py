import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import warnings

warnings.filterwarnings('ignore')

def fetch_data(symbol='BTC-USD', period='5y', interval='1d'):
    """Fetch historical data from yfinance"""
    print(f"Fetching {period} of {interval} data for {symbol}...")
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    # yfinance sometimes returns MultiIndex columns if single ticker is used depending on version, check & flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def generate_signals(df):
    """Generate trading signals using MACD, RSI, and SMA for market filtering."""
    print("Generating technical indicators...")
    # SMA 50 for Market Avoidance
    df['SMA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    df.dropna(inplace=True)
    return df

def check_entry_condition(row):
    """Modular function for Buy condition"""
    # Buy when Trend is UP (Close > SMA50), MACD crosses signal, RSI is reasonable
    trend_up = row['Close'] > row['SMA50']
    macd_bullish = row['MACD_Line'] > row['MACD_Signal']
    rsi_ok = row['RSI'] < 70 # Arbitrary avoid overbought 
    return trend_up and macd_bullish and rsi_ok

def check_exit_condition(row):
    """Modular function for normal Sell condition"""
    # Sell on trend breakdown or MACD bearish cross
    trend_down = row['Close'] < row['SMA50']
    macd_bearish = row['MACD_Line'] < row['MACD_Signal']
    rsi_overbought = row['RSI'] > 70
    return trend_down or macd_bearish or rsi_overbought

def run_backtest(df, initial_capital=10000.0, stop_loss_pct=0.03):
    """Row-by-row iteration backtest engine with Stop-Loss."""
    print("Running backtest engine...")
    position = 0.0          # Amount of BTC held
    cash = initial_capital
    entry_price = 0.0
    
    # To track performance
    equity_curve = []
    trades = []             # List of dicts with Trade Profit info
    
    for date, row in df.iterrows():
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # 1. State check: In position
        if position > 0:
            # Check Stop-Loss FIRST during the day
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            
            if low_price <= stop_loss_price:
                # Stop-Loss triggered
                # Assuming execution at exactly stop_loss_price (slippage ignored for simplicity)
                sell_price = stop_loss_price
                cash += position * sell_price
                
                profit_pct = (sell_price / entry_price) - 1
                trades.append({'Date': date, 'Type': 'SL', 'Profit': profit_pct})
                
                position = 0.0
                entry_price = 0.0
                
            elif check_exit_condition(row):
                # Normal Exit Strategy based on Close price
                sell_price = close_price
                cash += position * sell_price
                
                profit_pct = (sell_price / entry_price) - 1
                trades.append({'Date': date, 'Type': 'EXIT', 'Profit': profit_pct})
                
                position = 0.0
                entry_price = 0.0
        
        # 2. State check: Out of position
        if position == 0:
            if check_entry_condition(row):
                # Execute Buy at Close
                entry_price = close_price
                # Buy as much as possible with available cash
                position = cash / entry_price
                cash = 0.0
                
        # Calculate daily equity
        current_value = cash + (position * close_price if position > 0 else 0)
        equity_curve.append(current_value)
        
    df['Equity'] = equity_curve
    df['Buy_Hold'] = initial_capital * (df['Close'] / df['Close'].iloc[0])
    
    return df, trades

def calculate_metrics(df, trades):
    """Calculates key performance metrics"""
    print("Calculating performance metrics...")
    
    # Cumulative Return
    initial_equity = df['Equity'].iloc[0]
    final_equity = df['Equity'].iloc[-1]
    cum_return = (final_equity / initial_equity) - 1
    
    # Maximum Drawdown (MDD)
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
    mdd = df['Drawdown'].min()
    
    # Win Rate
    winning_trades = [t for t in trades if t['Profit'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0
    
    # Sharpe Ratio (using Daily Returns)
    df['Daily_Return'] = df['Equity'].pct_change()
    mean_daily_return = df['Daily_Return'].mean()
    std_daily_return = df['Daily_Return'].std()
    
    # Assuming risk-free rate = 0 for crypto, annualized by sqrt(365)
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return != 0 else 0
    
    return {
        'CumReturn': cum_return,
        'MDD': mdd,
        'WinRate': win_rate,
        'SharpeRatio': sharpe_ratio,
        'TotalTrades': len(trades)
    }

def analyze_worst_periods(df):
    """Analyzes and reports the worst-performing periods (e.g., by month/year)"""
    print("\n[Worst Periods Analysis]")
    
    # Group by Year-Month
    df['YearMonth'] = df.index.to_period('M')
    monthly_returns = df.groupby('YearMonth')['Daily_Return'].apply(lambda x: (1 + x).prod() - 1)*100
    
    worst_months = monthly_returns.nsmallest(5)
    print("Top 5 Worst Performing Months (Strategy):")
    for period, ret in worst_months.items():
        print(f"  - {period}: {ret:.2f}%")
        
    return worst_months

def plot_performance_multi(results, save_path="c:/auto_bitcoin/backtest_multi_result.png"):
    """Visualizes the equity curve vs buy & hold for multiple periods."""
    print("Plotting multi-period performance chart...")
    num_periods = len(results)
    
    # Create subplots based on number of periods
    fig, axes = plt.subplots(num_periods, 1, figsize=(14, 5 * num_periods))
    if num_periods == 1:
        axes = [axes]
        
    import matplotlib.font_manager as fm
    plt.rc('font', family='Malgun Gothic') # Windows default Korean font
    plt.rcParams['axes.unicode_minus'] = False
    
    for i, (period_name, data) in enumerate(results.items()):
        df = data['df']
        ax = axes[i]
        
        # Normalize Buy & Hold to start at same initial capital
        bh_capital = 10000.0 * (df['Close'] / df['Close'].iloc[0])
        
        ax.plot(df.index, df['Equity'], label='전략 자산 흐름 (Strategy)', color='blue', linewidth=1.5)
        ax.plot(df.index, bh_capital, label='단순 보유 (Buy & Hold)', color='orange', alpha=0.7, linewidth=1.5)
        
        ax.set_title(f'[{period_name}] 성과 비교', fontsize=15, fontweight='bold')
        ax.set_ylabel('포트폴리오 가치 ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")

def main():
    # 1. Fetch entire data (6 years to ensure we have enough history for early 2020)
    df_full = fetch_data(symbol='BTC-USD', period='6y', interval='1d')
    if df_full.empty:
        print("Error: No data fetched.")
        return
        
    # 2. Signals on full data to avoid warm-up issues
    df_full = generate_signals(df_full)
    
    # 3. Define Periods
    periods = {
        '1. 전체 구간 (2020~2025)': ('2020-01-01', '2025-12-31'),
        '2. 대상승장 (2020~2021)': ('2020-01-01', '2021-12-31'),
        '3. 대하락장 (2022)': ('2022-01-01', '2022-12-31'),
        '4. 횡보 및 회복장 (2023)': ('2023-01-01', '2023-12-31'),
        '5. ETF 승인 상승장 (2024~현재)': ('2024-01-01', '2025-12-31')
    }
    
    results = {}
    summary_data = []
    
    for name, (start_date, end_date) in periods.items():
        # Slice Data
        try:
            df_period = df_full.loc[start_date:end_date].copy()
        except:
            continue
            
        if len(df_period) == 0:
            continue
            
        # Run Backtest
        df_res, trades = run_backtest(df_period, initial_capital=10000.0, stop_loss_pct=0.03)
        metrics = calculate_metrics(df_res, trades)
        
        results[name] = {'df': df_res, 'metrics': metrics}
        
        # Buy & Hold Return for comparison
        bh_return = (df_res['Close'].iloc[-1] / df_res['Close'].iloc[0]) - 1
        
        summary_data.append({
            '기간 구분': name,
            '매매 횟수': metrics['TotalTrades'],
            '승률 (%)': f"{metrics['WinRate'] * 100:.2f}%",
            '전략 수익률 (%)': f"{metrics['CumReturn'] * 100:.2f}%",
            '단순 보유 수익률 (%)': f"{bh_return * 100:.2f}%",
            '최대 낙폭 (%)': f"{metrics['MDD'] * 100:.2f}%",
            '샤프 지수': f"{metrics['SharpeRatio']:.2f}"
        })
        
    # 4. Print Markdown Table
    print("\n\n### [구간별 백테스트 성과 요약]\n")
    print("| 기간 구분 | 매매 횟수 | 승률 | 전략 수익률 | 단순 보유 수익률 | 최대 낙폭(MDD) | 샤프 지수 |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for row in summary_data:
        print(f"| {row['기간 구분']} | {row['매매 횟수']} | {row['승률 (%)']} | {row['전략 수익률 (%)']} | {row['단순 보유 수익률 (%)']} | {row['최대 낙폭 (%)']} | {row['샤프 지수']} |")
    print("\n")

    # 5. Plot Multi-Period
    plot_performance_multi(results, save_path="c:/auto_bitcoin/backtest_multi_result.png")

if __name__ == "__main__":
    main()

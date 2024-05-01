from flask import Flask, render_template, request
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
import os
import pandas_datareader.data as pdr
import plotly.express as px


# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
template_dir = os.path.abspath('/Users/appleowner/etf_analysis/templates')
app = Flask(__name__, template_folder=template_dir)

def fetch_data(symbol, start, end):
    try:
        # Fetch the historical market data
        data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if data.empty:
            logging.warning(f"No data available for {symbol} starting from {start.strftime('%Y-%m-%d')}. Trying more recent start date.")
            data = yf.download(symbol)  # Try fetching without specifying date range
            start = data.index.min()
        
        # Fetch metadata
        etf_info = yf.Ticker(symbol)
        name = etf_info.info.get('shortName', 'N/A')

        return data, start, name
    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {str(e)}", exc_info=True)
        return pd.DataFrame(), dt.datetime.now(), 'N/A', 'N/A'
    
def calculate_metrics(data, index_data):
    try:
        data = data.sort_index()

        # Calculate daily returns
        daily_returns = data['Adj Close'].pct_change()

        # Calculate compounded annual returns
        annual_returns = (1 + daily_returns).resample('Y').prod() - 1

        # Calculate total compounded return over the entire period
        total_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1

        # Moving averages
        ma50 = data['Adj Close'].rolling(window=50).mean()
        ma200 = data['Adj Close'].rolling(window=200).mean()

        # Best performing quarter
        quarterly_returns = (1 + daily_returns).resample('Q').prod() - 1
        best_quarter = f"Q{quarterly_returns.idxmax().quarter}"

        # Beta calculation
        aligned_data = daily_returns.dropna().align(index_data['Adj Close'].pct_change().dropna(), join='inner')
        covariance_matrix = np.cov(aligned_data[0], aligned_data[1])
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

        # Sharpe Ratio calculation
        risk_free_rate = 0.02  # Assume an annual risk-free rate of 2%
        sharpe_ratio = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)

        return annual_returns, total_return, ma50, ma200, best_quarter, beta, sharpe_ratio
    except Exception as e:
        logging.error(f"Error in calculating metrics: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, None
def plot_data(data_dict, index_data):
    plots = {}
    try:
        # Cumulative Returns Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for symbol, data in data_dict.items():
            ax.plot(data['Adj Close'].pct_change().cumsum() * 100, label=f'{symbol} Returns %')
        ax.plot(index_data['Adj Close'].pct_change().cumsum() * 100, label='SPY Returns %', color='black', linestyle='--')
        ax.set_title('Cumulative Returns Comparison')
        ax.legend()
        ax.grid(True)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plots['cumulative_returns'] = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)

        # Moving Averages for each ETF
        for symbol, data in data_dict.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['Adj Close'], label=f'{symbol} Price', color='gray')
            ax.plot(data['Adj Close'].rolling(window=50).mean(), label='50-Day MA', color='blue')
            ax.plot(data['Adj Close'].rolling(window=200).mean(), label='200-Day MA', color='red')
            ax.set_title(f'{symbol} Moving Averages')
            ax.legend()
            ax.grid(True)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plots[f'{symbol}_ma'] = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close(fig)
    except Exception as e:
        logging.error(f"Failed to generate plots: {str(e)}", exc_info=True)
    return plots
import plotly.graph_objects as go

# plot moving averages with plotly

def plot_ma_with_plotly(data_dict):
    ma_plots = {}
    for symbol, data in data_dict.items():
        fig = go.Figure()

        # Add the Adjusted Close price trace
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Adj Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='lightgray')
        ))

        # Add the 50-Day Moving Average trace
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Adj Close'].rolling(window=50).mean(),
            mode='lines',
            name=f'{symbol} 50-Day MA',
            line=dict(color='blue')
        ))

        # Add the 200-Day Moving Average trace
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Adj Close'].rolling(window=200).mean(),
            mode='lines',
            name=f'{symbol} 200-Day MA',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=f'{symbol} Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

        # Convert the figure to HTML to embed in Flask
        ma_plots[symbol] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return ma_plots


def plot_data_with_plotly(data_dict, index_data, rfr_annual=0.05):
    plots = {}
    start_date = list(data_dict.values())[0].index.min()
    end_date = list(data_dict.values())[0].index.max()
    rfr_daily = (1 + rfr_annual) ** (1/252) - 1
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    rfr_cumulative = np.cumsum(np.repeat(rfr_daily, len(dates))) * 100

    # Combined plot for "ALL"
    fig_all = go.Figure()

    for symbol, data in data_dict.items():
        fig = go.Figure()

        # Individual ETF Returns
        fig.add_trace(go.Scatter(
            x=data['Adj Close'].pct_change().cumsum().index,
            y=data['Adj Close'].pct_change().cumsum() * 100,
            mode='lines',
            name=f'{symbol} Returns'
        ))

        # Adding data to the "ALL" plot
        fig_all.add_trace(go.Scatter(
            x=data['Adj Close'].pct_change().cumsum().index,
            y=data['Adj Close'].pct_change().cumsum() * 100,
            mode='lines',
            name=f'{symbol} Returns'
        ))

        # Configuration for individual plots
        fig.update_layout(
            title=f'{symbol} Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns (%)',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

        plots[symbol] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Adding SPY and RFR to "ALL" plot
    fig_all.add_trace(go.Scatter(
        x=index_data['Adj Close'].pct_change().cumsum().index,
        y=index_data['Adj Close'].pct_change().cumsum() * 100,
        mode='lines',
        name='SPY Returns',
        line=dict(color='white', dash='dash')
    ))

    fig_all.add_trace(go.Scatter(
        x=dates,
        y=rfr_cumulative,
        mode='lines',
        name='Risk-Free Rate (5% p.a.)',
        line=dict(color='green', dash='dot')
    ))

    # Configuration for "ALL" plot
    fig_all.update_layout(
        title='All ETFs Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (%)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    plots['ALL'] = fig_all.to_html(full_html=False, include_plotlyjs='cdn')

    return plots


@app.route('/', methods=['GET', 'POST'])
def index():
    symbols = []
    if request.method == 'POST':
        symbols = request.form.get('symbols', '').split(',')
        start_date = dt.datetime.now() - dt.timedelta(days=3650)
        end_date = dt.datetime.now()

        data_dict = {}
        metrics = {}
        index_data, _, _ = fetch_data('SPY', start_date, end_date)

        for symbol in symbols:
            data, _, name = fetch_data(symbol, start_date, end_date)
            if not data.empty:
                annual_returns, total_return, ma50, ma200, best_quarter, beta, sharpe_ratio = calculate_metrics(data, index_data)
                metrics[symbol] = {
                    'name': name,
                    'annual_returns': annual_returns.to_dict(),
                    'total_return': total_return,
                    'ma50': ma50.iloc[-1] if not ma50.empty else None,
                    'ma200': ma200.iloc[-1] if not ma200.empty else None,
                    'best_quarter': best_quarter,
                    'beta': beta,
                    'sharpe_ratio': sharpe_ratio
                }
                data_dict[symbol] = data

        plot_html = plot_data_with_plotly(data_dict, index_data)
        ma_plots = plot_ma_with_plotly(data_dict)

        return render_template('index.html', symbols=symbols, metrics=metrics, plots=plot_html, ma_plots=ma_plots)

    # Always return render_template to ensure the page loads even if no POST request is made
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

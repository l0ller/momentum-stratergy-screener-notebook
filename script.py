# --- Section 1: Setup and Data Loading ---
# Install necessary libraries
# pip install ta yfinance pandas plotly

import pandas as pd
import os
import yfinance as yf
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the path to the CSV file containing the symbols
csv_path = "/content/MW-NIFTY-500-06-Sep-2025.csv"

# Directory to save historical data
data_dir = 'nifty500_historical'
os.makedirs(data_dir, exist_ok=True)

# Load symbols from the CSV
try:
    df_symbols = pd.read_csv(csv_path)
    # Extract symbols from the 'Symbol' column, handle missing values and whitespace
    symbols_list = df_symbols['SYMBOL \n'].dropna().astype(str).str.strip().tolist()
    # Add ".NS" to each symbol for compatibility with yfinance and filter out non-stock tickers
    filtered_symbol_list = [symbol + ".NS" for symbol in symbols_list if symbol != 'NIFTY 500']
    print(f"Total number of symbols extracted and filtered: {len(filtered_symbol_list)}")
except FileNotFoundError:
    print(f"Error: Symbol file not found at {csv_path}")
    filtered_symbol_list = [] # Initialize as empty list if file not found
except KeyError:
    print(f"Error: 'SYMBOL \n' column not found in {csv_path}")
    filtered_symbol_list = []


# --- Section 2: Download Historical Data ---
print("\n--- Downloading Historical Data ---")
for symbol in filtered_symbol_list:
    try:
        # print(f"Downloading {symbol}...") # Commented out for cleaner output
        df_hist = yf.download(symbol, period="150d", interval="1d", progress=False, auto_adjust=False)

        if df_hist.empty:
            print(f"No data found for {symbol}. Skipping.")
            continue

        # Flatten multi-index columns if they exist
        if isinstance(df_hist.columns, pd.MultiIndex):
            df_hist.columns = df_hist.columns.get_level_values(0)

        # Keep only necessary columns if present
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in df_hist.columns if any(c in col for c in cols)]

        if not available_cols:
            print(f"No relevant columns found for {symbol}. Skipping.")
            continue

        df_hist = df_hist[available_cols].dropna()

        # Reset index to bring 'Date' as a column
        if not isinstance(df_hist.index, pd.DatetimeIndex):
            df_hist.index = pd.to_datetime(df_hist.index)
        df_hist = df_hist.reset_index()

        # Handle potential duplicate Date columns
        date_cols = [col for col in df_hist.columns if 'Date' in col or 'date' in col]
        if len(date_cols) > 1:
             # Keep the first and drop others
             df_hist = df_hist.drop(columns=date_cols[1:])

        # Drop rows where all values match the ticker name (metadata row), if necessary
        df_hist = df_hist[~(df_hist.astype(str) == symbol).all(axis=1)]

        # Ensure 'Date' column is datetime and unique
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.loc[:, ~df_hist.columns.duplicated()]

        # Select columns to save, making sure they exist
        cols_to_save = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        cols_to_save_present = [col for col in df_hist.columns if col in cols_to_save] # Check if column name exactly matches

        if not cols_to_save_present:
             print(f"No essential columns (Date, Open, High, Low, Close, Volume) found for {symbol}. Skipping save.")
             continue

        df_to_save = df_hist[cols_to_save_present]

        # Save to CSV
        filename = f'{data_dir}/{symbol.replace(".NS", "")}.csv'
        df_to_save.to_csv(filename, index=False, header=True)

        # print(f"Saved {symbol} with {len(df_to_save)} rows.") # Commented out for cleaner output

    except Exception as e:
        print(f"Error processing {symbol} during download: {e}")

# --- Section 3: Stock Screening and Indicator Calculation ---

# Custom OBV calculation function
def calculate_obv(df):
    obv = [0] * len(df)
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=df.index)

# Parameters for screening and indicators
adx_period = 14
dma_20_period = 20
dma_50_period = 50
atr_period = 14
obv_ma_period = 20
# Minimum data points needed for the longest indicator
min_data_points = max(adx_period, dma_50_period, atr_period, obv_ma_period)

# Risk/Reward Parameters
ATR_MULTIPLIER = 1.5
MIN_RISK_REWARD = 1.5
TARGET_ATR_MULTIPLIER = 3.0

# Initialize a list to store data for screened symbols
screened_data = []

print("\n--- Screening Stocks and Calculating Indicators ---")

# Get list of all saved CSV files
try:
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
except FileNotFoundError:
    print(f"Error: Data directory '{data_dir}' not found.")
    csv_files = [] # Initialize as empty list if directory not found


for file in csv_files:
    try:
        symbol = file.replace('.csv', '')
        file_path = os.path.join(data_dir, file)

        # Load the data with 'Date' as index and parse dates
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        # Ensure necessary columns exist and have correct data types
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {symbol}: Missing required columns for screening. Found columns: {df.columns.tolist()}")
            continue

        # Convert relevant columns to numeric, coercing errors
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values in relevant columns before calculations
        df_cleaned = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

        if df_cleaned.empty or len(df_cleaned) < min_data_points:
            # print(f"Not enough data for {symbol} after cleaning for indicator calculation.")
            continue

        # Calculate ADX, +DI, and -DI
        adx_indicator = ADXIndicator(high=df_cleaned['High'], low=df_cleaned['Low'], close=df_cleaned['Close'], window=adx_period)
        df['ADX14'] = adx_indicator.adx()
        df['PLUS_DI'] = adx_indicator.adx_pos()
        df['MINUS_DI'] = adx_indicator.adx_neg()

        # Calculate ATR
        atr_indicator = AverageTrueRange(high=df_cleaned['High'], low=df_cleaned['Low'], close=df_cleaned['Close'], window=atr_period)
        df['ATR'] = atr_indicator.average_true_range()

        # Calculate 20DMA and 50DMA
        df['DMA20'] = SMAIndicator(close=df_cleaned['Close'], window=dma_20_period).sma_indicator()
        df['DMA50'] = SMAIndicator(close=df_cleaned['Close'], window=dma_50_period).sma_indicator()

        # Calculate OBV using the custom function and then its MA
        df['OBV'] = calculate_obv(df_cleaned)
        df['OBV_MA20'] = SMAIndicator(close=df['OBV'], window=obv_ma_period).sma_indicator()

        # Drop rows with NaN values created by indicator calculation
        df = df.dropna(subset=['ADX14', 'PLUS_DI', 'MINUS_DI', 'DMA20', 'DMA50', 'Close', 'ATR', 'OBV', 'OBV_MA20'])

        # Get the latest data point after calculating indicators and dropping NaNs
        if not df.empty:
            latest_data = df.iloc[-1]

            # Check if the screening criteria are met
            if (latest_data['ADX14'] > 25 and
                latest_data['DMA20'] > latest_data['DMA50'] and
                (latest_data['Close'] < latest_data['DMA20'] ) and
                latest_data['OBV'] > latest_data['OBV_MA20'] and
                latest_data['PLUS_DI'] > latest_data['MINUS_DI']):

                # Calculate percentage difference between 20DMA and LTP
                percentage_diff = ((latest_data['Close'] - latest_data['DMA20']) / latest_data['DMA20']) * 100

                # Calculate Risk to Reward Ratio
                atr = latest_data['ATR']
                ltp = latest_data['Close']
                dma20 = latest_data['DMA20']

                # Assuming entry price is close to 20DMA and stop loss is based on ATR below entry
                entry_price = dma20
                stop_price = entry_price - (atr * ATR_MULTIPLIER)
                risk_per_share = entry_price - stop_price

                # Assuming a target price based on ATR above the entry price
                target_price = entry_price + (atr * TARGET_ATR_MULTIPLIER)
                expected_reward_per_share = target_price - entry_price

                risk_reward_ratio = expected_reward_per_share / risk_per_share if risk_per_share != 0 else 0

                # Check if Risk to Reward meets the minimum requirement
                if risk_reward_ratio >= MIN_RISK_REWARD:
                    screened_data.append({
                        'Symbol': symbol,
                        'ADX14': latest_data['ADX14'],
                        'PLUS_DI': latest_data['PLUS_DI'],
                        'MINUS_DI': latest_data['MINUS_DI'],
                        'DMA20': latest_data['DMA20'],
                        'DMA50': latest_data['DMA50'],
                        'OBV': latest_data['OBV'],
                        'OBV_MA20': latest_data['OBV_MA20'], # Add OBV MA to results
                        'LTP': latest_data['Close'],
                        '20DMA_LTP_Diff_pct': percentage_diff,
                        'ATR': latest_data['ATR'],
                        'Risk_Reward_Ratio': risk_reward_ratio
                    })

    except Exception as e:
        print(f"Error processing {symbol} during screening and calculation: {e}")

# Create a DataFrame from the screened data
screened_df = pd.DataFrame(screened_data)

# Sort the DataFrame by percentage difference
if not screened_df.empty:
    screened_df = screened_df.sort_values(by='20DMA_LTP_Diff_pct', ascending=False)

    # Save the screened data to a new CSV file
    output_file = 'nifty500_screener_results.csv'
    screened_df.to_csv(output_file, index=False)

    print(f"\nScreening results saved to {output_file}")
    print(f"Number of stocks meeting the criteria: {len(screened_df)}")
else:
    print("\nNo stocks met the screening criteria.")

# --- Section 4: Capital Allocation Prioritization ---
print("\n--- Calculating Capital Allocation Priority ---")

# Load the screened results
try:
    screened_df_for_priority = pd.read_csv('nifty500_screener_results.csv')
except FileNotFoundError:
    print("Error: 'nifty500_screener_results.csv' not found. Please run the screening process first.")
    screened_df_for_priority = pd.DataFrame() # Create empty DataFrame to avoid errors

if not screened_df_for_priority.empty:
    # Calculate ATR as a percentage of LTP for prioritization
    screened_df_for_priority['ATR_pct_of_LTP'] = (screened_df_for_priority['ATR'] / screened_df_for_priority['LTP']) * 100

    # Example: Create a composite score (higher is better for allocation)
    # You can adjust the weights based on the importance you assign to each factor
    # Normalize ADX (higher is better)
    normalized_adx = screened_df_for_priority['ADX14'] / screened_df_for_priority['ADX14'].max() if screened_df_for_priority['ADX14'].max() != 0 else 0

    # Normalize ATR (lower is better)
    # Use ATR_pct_of_LTP for normalization if you want to prioritize based on relative volatility
    normalized_atr = 1 - (screened_df_for_priority['ATR_pct_of_LTP'] / screened_df_for_priority['ATR_pct_of_LTP'].max()) if screened_df_for_priority['ATR_pct_of_LTP'].max() != 0 else 0

    # Normalize 20DMA_LTP_Diff_pct (closer to 0 is better, so we use the inverse of the absolute value)
    # To avoid division by zero or very small numbers, add a small constant
    normalized_diff = 1 / (abs(screened_df_for_priority['20DMA_LTP_Diff_pct']) + 0.01)
    normalized_diff = normalized_diff / normalized_diff.max() if normalized_diff.max() != 0 else 0 # Normalize the inverse


    screened_df_for_priority['Priority_Score'] = (
        normalized_adx +
        normalized_atr +
        normalized_diff
    )

    # Sort by the composite score in descending order
    capital_allocation_priority_df = screened_df_for_priority.sort_values(by='Priority_Score', ascending=False)

    # Display the sorted results, including the new column
    print("\nStocks sorted by Capital Allocation Priority:")
    display(capital_allocation_priority_df[['Symbol', 'ADX14', 'ATR', 'ATR_pct_of_LTP', '20DMA_LTP_Diff_pct', 'Priority_Score']].head(20)) # Display top 20 as a sample

    # Optionally save the prioritized list to a new CSV
    capital_allocation_priority_df.to_csv('nifty500_capital_priority.csv', index=False)
    print("\nCapital allocation priority results saved to 'nifty500_capital_priority.csv'")

else:
    print("\nNo screened stocks available for capital allocation prioritization.")


# --- Section 5: Plotting Screened Stocks ---

# Custom OBV calculation function (returns OBV SMA 9 for plotting)
def calculate_obv_for_plotting(df):
    obv = [0] * len(df)
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    obv_series = pd.Series(obv, index=df.index)
    # Calculate 9-period SMA of OBV
    obv_sma_9 = SMAIndicator(close=obv_series, window=9).sma_indicator()
    return obv_sma_9


def plot_stock_with_dma_and_obv(symbol, data_dir='nifty500_historical', dma_periods=[20, 50]):
    """
    Loads historical data for a given stock symbol from a CSV file,
    calculates DMAs and OBV, and plots the price along with the DMAs
    and OBV in separate subplots.

    Args:
        symbol (str): The stock symbol (without .NS).
        data_dir (str): The directory where the historical CSV files are saved.
        dma_periods (list): A list of periods for calculating Simple Moving Averages.
    """
    file_path = os.path.join(data_dir, f'{symbol}.csv')

    if not os.path.exists(file_path):
        print(f"Error: Data file for {symbol} not found at {file_path}")
        return

    try:
        # Load the data
        df = pd.read_csv(file_path, parse_dates=['Date'])

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)

        # Ensure necessary columns exist and have correct data types
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {symbol}: Missing required columns for plotting. Found columns: {df.columns.tolist()}")
            return

        # Convert relevant columns to numeric, coercing errors
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values in relevant columns before calculations
        df_cleaned = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

        if df_cleaned.empty:
            print(f"No data for {symbol} after cleaning for plotting.")
            return

        # Calculate DMAs
        for period in dma_periods:
            df[f'DMA{period}'] = SMAIndicator(close=df_cleaned['Close'], window=period).sma_indicator()

        # Calculate OBV (now returns OBV SMA 9)
        df['OBV_SMA9'] = calculate_obv_for_plotting(df_cleaned)

        # Drop rows with NaN values created by DMA calculation and OBV calculation
        df = df.dropna(subset=[f'DMA{period}' for period in dma_periods] + ['OBV_SMA9'])

        if df.empty:
             print(f"Not enough data for {symbol} to calculate indicators for plotting.")
             return

        # Create subplots: 2 rows, 1 column
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f'{symbol} Price with DMAs', 'On-Balance Volume (OBV) SMA 9'),
                            vertical_spacing=0.1)

        # Add Close Price trace to the first subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'),
                      row=1, col=1)

        # Add DMA traces to the first subplot
        for period in dma_periods:
            fig.add_trace(go.Scatter(x=df.index, y=df[f'DMA{period}'], mode='lines', name=f'{period} DMA'),
                          row=1, col=1)

        # Add OBV SMA 9 trace to the second subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV_SMA9'], mode='lines', name='OBV SMA 9', line=dict(color='purple')),
                      row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=600 # Adjust height as needed
        )

        # Update y-axis titles for subplots
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)

        fig.show()

    except Exception as e:
        print(f"Error plotting {symbol}: {e}")

# --- Code to read screened results and plot charts ---
print("\n--- Generating Charts for Screened Stocks ---")
try:
    # Load the screened results
    screened_df_for_plotting = pd.read_csv('nifty500_screener_results.csv')

    # Sort by the percentage difference for plotting order
    screened_df_sorted = screened_df_for_plotting.sort_values(by='20DMA_LTP_Diff_pct', ascending=False)

    print("Generating charts for screened stocks, sorted by 20DMA_LTP_Diff_pct...")

    # Iterate through the sorted symbols and generate charts
    for index, row in screened_df_sorted.iterrows():
        symbol_to_plot = row['Symbol']
        print(f"\nGenerating chart for {symbol_to_plot}...")
        # Call the plotting function
        plot_stock_with_dma_and_obv(symbol_to_plot)

except FileNotFoundError:
    print("Error: 'nifty500_screener_results.csv' not found. Please run the screening process first.")
except KeyError:
    print("Error: 'Symbol' or '20DMA_LTP_Diff_pct' column not found in 'nifty500_screener_results.csv'.")
except Exception as e:
    print(f"An error occurred while processing the screened results for plotting: {e}")

# Nifty 500 Momentum Screener and Analysis

This repository contains a Python script using a Jupyter Notebook (specifically designed for Google Colab) to download historical stock data for Nifty 500 companies, perform technical analysis based on specific criteria, prioritize stocks for capital allocation, and generate charts for screened stocks.

## Features

- Downloads historical stock data using the `yfinance` library.
- Implements a stock screening strategy based on:
    - ADX (Average Directional Index)
    - DMA (Simple Moving Averages)
    - OBV (On-Balance Volume)
    - Risk/Reward Ratio calculation based on ATR (Average True Range)
- Prioritizes screened stocks for capital allocation using a composite score.
- Generates interactive candlestick charts with DMA and OBV indicators using `plotly`.
- Saves screened results and capital allocation priority lists to CSV files.

## Prerequisites

To run this notebook in Google Colab, you will need:

- Access to Google Colab.
- A Google Drive account (if you choose to store data there, although the current setup saves data locally within the Colab environment).
- The necessary Python libraries, which are installed within the notebook (`ta`, `yfinance`, `pandas`, `plotly`).

## Setup and Usage

1.  **Open the Notebook in Google Colab:** Upload the `.ipynb` file to your Google Drive and open it with Google Colab.
2.  **Install Libraries:** Run the first code cell to install the required libraries (`ta`). The main notebook cell also includes commented-out lines for installing `yfinance`, `pandas`, and `plotly` in case they are not already present in your Colab environment.
3.  **Upload Symbol File:** Ensure you have the CSV file containing the Nifty 500 symbols (e.g., `MW-NIFTY-500-06-Sep-2025.csv`) uploaded to your Colab environment (or connected Google Drive). Update the `csv_path` variable in the notebook if your file path is different.
4.  **Run the Notebook:** Execute the code cells sequentially.
    - The notebook will first load the symbols.
    - Then, it will download historical data for each symbol and save it to the `nifty500_historical` directory.
    - Next, it will perform the stock screening and indicator calculations, saving the results to `nifty500_screener_results.csv`.
    - Following that, it will calculate the capital allocation priority and save the results to `nifty500_capital_priority.csv`.
    - Finally, it will generate and display interactive charts for the screened stocks.

## Code Structure

The notebook is divided into sections for clarity:

-   **Section 1: Setup and Data Loading:** Installs libraries and loads stock symbols from the provided CSV.
-   **Section 2: Download Historical Data:** Downloads historical price and volume data using `yfinance`.
-   **Section 3: Stock Screening and Indicator Calculation:** Calculates technical indicators (ADX, DMA, ATR, OBV) and applies the screening criteria.
-   **Section 4: Capital Allocation Prioritization:** Calculates a priority score for screened stocks.
-   **Section 5: Plotting Screened Stocks:** Generates and displays charts for the stocks that met the screening criteria.

## Customization

-   **Screening Criteria:** Modify the conditions in the screening section (Section 3) to adjust the technical analysis criteria.
-   **Indicator Periods:** Change the `adx_period`, `dma_20_period`, `dma_50_period`, `atr_period`, and `obv_ma_period` variables to use different time windows for the indicators.
-   **Risk/Reward Parameters:** Adjust `ATR_MULTIPLIER`, `MIN_RISK_REWARD`, and `TARGET_ATR_MULTIPLIER` to change the risk/reward calculation logic.
-   **Capital Allocation Logic:** Modify the `Priority_Score` calculation in Section 4 to change how stocks are prioritized.
-   **Plotting:** Customize the `plot_stock_with_dma_and_obv` function in Section 5 to change the appearance or indicators shown in the charts.

## Potential Enhancements

-   **Integrate Live Data:** Connect to a live stock market data API (like the Kotak API mentioned in the notebook's markdown) to perform real-time screening.
-   **Backtesting:** Develop a backtesting framework to test the performance of the screening strategy on historical data.
-   **More Indicators:** Incorporate additional technical indicators (e.g., RSI, MACD) into the screening and analysis.
-   **User Interface:** Create a simple user interface (e.g., using Gradio or Streamlit) to make the tool more interactive.
-   **Error Handling:** Add more robust error handling for data download and processing.

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository and submit a pull request.

## License

[Specify your license here, e.g., MIT License]

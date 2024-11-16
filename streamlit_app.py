import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
import time
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# Set the page configuration
st.set_page_config(
    page_title="Investment Portfolio Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("📈 Investment Portfolio Tracker")

# Option to separate realized and unrealized profits in the plot
separate_profits = st.sidebar.checkbox(
    "Separate realized and unrealized profits in the plot", value=True
)

# Instructions
st.markdown("""
This application allows you to upload your own trade and dividend logs to track your investment portfolio.

- **Trade Log CSV Template:** The CSV file should have the following columns:
  - `Date`: The date of the trade in `YYYY-MM-DD` format.
  - `Stock`: The stock symbol.
  - `Action`: `Buy` or `Sell`.
  - `Quantity`: Number of shares.
  - `Price per Share`: Price per share at the time of trade.
  - `Expenses`: Any additional expenses (e.g., commission fees).

- **Dividend Log CSV Template:** The CSV file should have the following columns:
  - `Date`: The date the dividend was received in `YYYY-MM-DD` format.
  - `Stock`: The stock symbol.
  - `Dividend Received`: Amount of dividend received.
""")

# Provide sample CSV templates for download
sample_trade_log = pd.DataFrame({
    'Date': ['2023-01-01'],
    'Stock': ['AAPL'],
    'Action': ['Buy'],
    'Quantity': [10],
    'Price per Share': [150.0],
    'Expenses': [0.0]
})

sample_dividend_log = pd.DataFrame({
    'Date': ['2023-02-01'],
    'Stock': ['AAPL'],
    'Dividend Received': [5.0]
})


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


st.sidebar.markdown("### Download Sample CSV Templates:")
st.sidebar.download_button(
    label="Download Trade Log Template",
    data=convert_df(sample_trade_log),
    file_name='trade_log_template.csv',
    mime='text/csv',
)

st.sidebar.download_button(
    label="Download Dividend Log Template",
    data=convert_df(sample_dividend_log),
    file_name='dividend_log_template.csv',
    mime='text/csv',
)

# File upload for trade log
trade_log_file = st.sidebar.file_uploader("Upload your Trade Log CSV", type=['csv'])

# File upload for dividend log
dividend_log_file = st.sidebar.file_uploader("Upload your Dividend Log CSV", type=['csv'])

# Initialize DataFrames to store trade and dividend logs
trade_columns = [
    "Date",
    "Stock",
    "Action",
    "Quantity",
    "Price per Share",
    "Expenses",
    "Total Cost",
]
dividend_columns = ["Date", "Stock", "Dividend Received"]

# Load trade log
if trade_log_file is not None:
    trade_log = pd.read_csv(trade_log_file)
    trade_log['Total Cost'] = trade_log.apply(
        lambda row: (row['Quantity'] * row['Price per Share'] + row['Expenses']) if row[
                                                                                        'Action'].lower() == 'buy' else -(
                row['Quantity'] * row['Price per Share'] - row['Expenses']),
        axis=1
    )
    trade_log['Date'] = pd.to_datetime(trade_log['Date']).dt.normalize()

    # Use session state to persist data
    st.session_state['trade_log'] = trade_log
else:
    st.warning("Please upload your Trade Log CSV file.")
    st.stop()

# Load dividend log
if dividend_log_file is not None:
    dividend_log = pd.read_csv(dividend_log_file)
    dividend_log['Date'] = pd.to_datetime(dividend_log['Date']).dt.normalize()

    # Use session state to persist data
    st.session_state['dividend_log'] = dividend_log
else:
    dividend_log = pd.DataFrame(columns=dividend_columns)
    st.session_state['dividend_log'] = dividend_log  # Assign empty DataFrame to session state

# Access the data from session state
trade_log = st.session_state['trade_log']
dividend_log = st.session_state['dividend_log']


# Function definitions (same as before, without changes)
def calculate_individual_pnl():
    # ... (same as previous code)
    all_individual_pnls = pd.DataFrame(columns=["Date", "Stock", "PnL", "Investment"])

    for stock in trade_log["Stock"].unique():
        stock_trades = trade_log[trade_log["Stock"] == stock].sort_values(by="Date")
        buys = stock_trades[stock_trades["Action"].str.lower() == "buy"].copy()
        sells = stock_trades[stock_trades["Action"].str.lower() == "sell"]
        individual_pnls = []

        for index, sell in sells.iterrows():
            sell_quantity = sell["Quantity"]
            while sell_quantity > 0 and not buys.empty:
                buy = buys.iloc[0]
                matched_quantity = min(buy["Quantity"], sell_quantity)
                sell_quantity -= matched_quantity
                buys.at[buys.index[0], "Quantity"] -= matched_quantity

                # Include expenses in investment and PnL
                buy_price_per_share = buy["Price per Share"] + buy["Expenses"] / buy["Quantity"]
                sell_price_per_share = sell["Price per Share"] - sell["Expenses"] / sell["Quantity"]
                investment = matched_quantity * buy_price_per_share
                sell_proceeds = matched_quantity * sell_price_per_share
                pnl = sell_proceeds - investment

                individual_pnls.append(
                    {
                        "Date": sell["Date"],
                        "Stock": stock,
                        "PnL": round(pnl, 2),
                        "Investment": round(investment, 2),
                    }
                )

                if buys.iloc[0]["Quantity"] == 0:
                    buys = buys.iloc[1:]

        stock_individual_pnls = pd.DataFrame(individual_pnls)
        all_individual_pnls = pd.concat([all_individual_pnls, stock_individual_pnls], ignore_index=True)

    # Sort by Date before returning
    all_individual_pnls = all_individual_pnls.sort_values(by="Date")
    return all_individual_pnls


def calculate_pnl_percentage():
    individual_pnls = calculate_individual_pnl()
    total_pnl = individual_pnls.groupby("Stock").agg({"PnL": sum, "Investment": sum})
    total_pnl["PnL Percentage"] = (total_pnl["PnL"] / total_pnl["Investment"]) * 100
    return total_pnl.round(2)


def calculate_total_pnl_and_percentage():
    individual_pnls = calculate_individual_pnl()
    total_pnl = individual_pnls["PnL"].sum()
    total_investment = individual_pnls["Investment"].sum()
    total_pnl_percentage = (total_pnl / total_investment) * 100 if total_investment != 0 else 0
    return round(total_pnl, 2), round(total_investment, 2), round(total_pnl_percentage, 2)


def calculate_dividend_profit():
    if not dividend_log.empty:
        dividend_profit = dividend_log.groupby("Stock").agg({"Dividend Received": sum})
        return dividend_profit.round(2)
    else:
        return pd.DataFrame(columns=["Dividend Received"])


def calculate_current_invested_capital_per_stock():
    # ... (same as previous code)
    invested_capital_per_stock = {}

    for stock in trade_log["Stock"].unique():
        stock_trades = trade_log[trade_log["Stock"] == stock].sort_values(by="Date")
        remaining_quantity = 0
        invested_capital = 0.0

        for index, trade in stock_trades.iterrows():
            if trade["Action"].lower() == "buy":
                total_cost = trade["Quantity"] * (
                        trade["Price per Share"] + trade["Expenses"] / trade["Quantity"]
                )
                invested_capital += total_cost
                remaining_quantity += trade["Quantity"]
            elif trade["Action"].lower() == "sell":
                if remaining_quantity > 0:
                    cost_per_share = invested_capital / remaining_quantity
                    invested_capital -= cost_per_share * trade["Quantity"]
                    remaining_quantity -= trade["Quantity"]

        # Fetch the current price of the stock
        ticker = stock_ticker_mapping.get(stock, stock)
        if ticker in successful_tickers:
            try:
                current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            except Exception:
                current_price = 0.0  # Handle missing data
        else:
            current_price = 0.0  # Set to 0 if data is not available

        # Calculate average invested price and current value
        if remaining_quantity > 0:
            average_invested_price = invested_capital / remaining_quantity
            current_value = remaining_quantity * current_price
        else:
            average_invested_price = 0.0
            current_value = 0.0

        invested_capital_per_stock[stock] = {
            "No of Stocks": remaining_quantity,
            "Invested Capital": round(invested_capital, 2),
            "Average Invested Price": round(average_invested_price, 2),
            "Current Price": round(current_price, 2),
            "Current Value": round(current_value, 2),
        }

    invested_capital_per_stock_df = pd.DataFrame.from_dict(invested_capital_per_stock, orient="index").reset_index()
    invested_capital_per_stock_df.rename(columns={"index": "Stock"}, inplace=True)
    invested_capital_per_stock_df = invested_capital_per_stock_df[
        invested_capital_per_stock_df["Invested Capital"] > 0
        ]
    return invested_capital_per_stock_df


def process_trades_and_dividends_up_to_date(date, trades, dividends):
    holdings_per_stock = defaultdict(int)
    invested_capital_per_stock = defaultdict(float)
    cumulative_realized_pnl = 0.0

    # Calculate cumulative dividends up to the given date
    if not dividends.empty:
        cumulative_dividends = dividends[dividends["Date"] <= date]["Dividend Received"].sum()
    else:
        cumulative_dividends = 0.0

    # Process trades up to the given date
    trades_up_to_date = trades[trades["Date"] <= date].sort_values(by="Date")
    for index, trade in trades_up_to_date.iterrows():
        stock = trade["Stock"]
        action = trade["Action"].lower()
        quantity = trade["Quantity"]
        price = trade["Price per Share"]
        expenses = trade["Expenses"]

        if action == "buy":
            total_cost = quantity * (price + expenses / quantity)
            invested_capital_per_stock[stock] += total_cost
            holdings_per_stock[stock] += quantity
        elif action == "sell":
            if holdings_per_stock[stock] > 0:
                cost_basis_per_share = invested_capital_per_stock[stock] / holdings_per_stock[stock]
                sell_proceeds = quantity * (price - expenses / quantity)
                realized_pnl = sell_proceeds - (cost_basis_per_share * quantity)
                cumulative_realized_pnl += realized_pnl
                invested_capital_per_stock[stock] -= cost_basis_per_share * quantity
                holdings_per_stock[stock] -= quantity

                if holdings_per_stock[stock] == 0:
                    invested_capital_per_stock[stock] = 0.0

    return holdings_per_stock, invested_capital_per_stock, cumulative_realized_pnl, cumulative_dividends



# Create a mapping of stock symbols to their Yahoo Finance tickers
stock_ticker_mapping = {
    # Greek stocks
    "AEGN": "AEGN.AT",
    "ELPE": "ELPE.AT",
    "MOH": "MOH.AT",
    "MYTIL": "MYTIL.AT",
    "OPAP": "OPAP.AT",
    "BELA": "BELA.AT",
    "ALPHA": "ALPHA.AT",
    "PPC": "PPC.AT",
    "PPA": "PPA.AT",
    "OPTIMA": "OPTIMA.AT",
    "AIA": "AIA.AT",
    # US stocks
    "AAPL": "AAPL",
    "OXY": "OXY",
    "USO": "USO",
    "BMY": "BMY",
    "CVS": "CVS",
    "SVXY": "SVXY",
    "DIS": "DIS",
    "O": "O",
    "V": "V",
    "VUAA": "VUAA.AS",  # Adjusted ticker for European market
    "PYPL": "PYPL",
    "INTC": "INTC",
    "ATEN": "ATEN",
    "UWMC": "UWMC",
    "F": "F",
    # Add other stocks as needed
}

# Remove duplicates in stock_ticker_mapping
stock_ticker_mapping = {k: v for k, v in stock_ticker_mapping.items()}

# Ensure 'Date' columns are datetime objects
trade_log["Date"] = pd.to_datetime(trade_log["Date"]).dt.normalize()
if not dividend_log.empty:
    dividend_log["Date"] = pd.to_datetime(dividend_log["Date"]).dt.normalize()

# Calculate and display PnL percentage for each stock
pnl_percentage = calculate_pnl_percentage()
st.subheader("PnL Percentage for Each Stock")
st.dataframe(pnl_percentage)

# Calculate individual PnL for all stocks
all_individual_pnls = calculate_individual_pnl()
st.subheader("Individual PnL for All Stocks")
st.dataframe(all_individual_pnls)

# Calculate the total PnL, total investment, and overall PnL percentage
total_pnl, total_investment, total_pnl_percentage = calculate_total_pnl_and_percentage()
st.markdown(
    f"""
**Total PnL:** {total_pnl}  
**Total Investment:** {total_investment}  
**Total PnL Percentage:** {total_pnl_percentage}%  
"""
)

# Calculate and display dividend profit
dividend_profit = calculate_dividend_profit()
st.subheader("Dividend Log")
st.dataframe(dividend_log)
st.subheader("Dividend Profit for each Stock")
st.dataframe(dividend_profit)
total_dividends_received = dividend_profit["Dividend Received"].sum() if not dividend_profit.empty else 0.0
st.write(f"**Total Dividends Received:** {total_dividends_received}")

# Calculate total income
total_income = round(total_pnl + total_dividends_received, 2)
st.write(f"**Total Income:** {total_income}")

# Fetch historical price data with exception handling
# Get earliest and latest dates from your logs
earliest_date = trade_log["Date"].min()
if not dividend_log.empty:
    earliest_date = min(earliest_date, dividend_log["Date"].min())
latest_date = datetime.now().date()

# Generate month-end dates between earliest and latest dates
month_ends = pd.date_range(start=earliest_date, end=latest_date, freq="M").normalize()

# Add latest date to the list of dates if not present
date_list = list(month_ends)
if pd.Timestamp(latest_date) not in date_list:
    date_list.append(pd.Timestamp(latest_date))

# Get the list of unique stocks
stocks = trade_log["Stock"].unique()

# Fetch historical prices for the stocks over the date range using the mapping
tickers = [stock_ticker_mapping.get(stock, stock) for stock in stocks]
start_date = earliest_date.strftime("%Y-%m-%d")
end_date = latest_date.strftime("%Y-%m-%d")

# Initialize lists to keep track of successful and failed tickers
successful_tickers = []
failed_tickers = []

# Initialize an empty DataFrame to store the price data
price_data = pd.DataFrame()


# Function to download data with retries
def download_data_with_retry(ticker, start_date, end_date, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)["Close"]
            if data.empty:
                print(f"No data found for {ticker}")
                return None
            else:
                return data
        except Exception as e:
            print(f"Attempt {i + 1}: Failed to download data for {ticker}: {e}")
            time.sleep(delay)
    return None


# Fetch historical price data with retry
st.write("**Downloading historical price data...**")
for ticker in tickers:
    data = download_data_with_retry(ticker, start_date, end_date)
    if data is not None:
        price_data[ticker] = data
        successful_tickers.append(ticker)
    else:
        failed_tickers.append(ticker)

if failed_tickers:
    st.write(f"Failed to download data for the following tickers after retries: {failed_tickers}")

# Proceed only with successful tickers
stocks = [stock for stock, ticker in stock_ticker_mapping.items() if ticker in successful_tickers]
if not stocks:
    st.error("No data available for any ticker. Exiting.")
    st.stop()

# Update trade_log to exclude failed tickers
trade_log = trade_log[trade_log["Stock"].isin(stocks)]
if not dividend_log.empty:
    dividend_log = dividend_log[dividend_log["Stock"].isin(stocks)]

# Calculate current invested capital per stock
current_invested_capital_per_stock = calculate_current_invested_capital_per_stock()
st.subheader("Current Invested Capital per Stock")
st.dataframe(current_invested_capital_per_stock)

current_total_invested_capital = current_invested_capital_per_stock["Invested Capital"].sum()
st.write(f"**Current Total Invested Capital:** {current_total_invested_capital}")

current_total_value = current_invested_capital_per_stock["Current Value"].sum()
st.write(f"**Current Total Value:** {current_total_value}")

# Calculate current result and true PnL
current_result = round(current_total_value - current_total_invested_capital, 2)
st.write(f"**Current Result:** {current_result}")

current_true_pnl = round(total_income + current_result, 2)
st.write(f"**Current True PnL:** {current_true_pnl}")

# Initialize lists to store results
dates = []
invested_capitals = []
realized_profits = []
unrealized_profits = []
total_profits = []
dividends_list = []

for date in date_list:
    holdings, invested_capitals_per_stock, cumulative_realized_pnl, cumulative_dividends = process_trades_and_dividends_up_to_date(
        date, trade_log, dividend_log
    )

    # Convert date to string in 'YYYY-MM-DD' format
    date_str = date.strftime("%Y-%m-%d")

    # Get the prices as of the date
    try:
        prices_on_date = price_data.loc[date_str]
    except KeyError:
        try:
            prices_on_date = price_data.loc[:date_str].iloc[-1]
        except IndexError:
            prices_on_date = pd.Series()

    # Compute current value of holdings
    current_value = 0.0
    for stock in holdings:
        quantity = holdings[stock]
        ticker = stock_ticker_mapping.get(stock, stock)
        price = prices_on_date.get(ticker, 0.0)
        current_value += quantity * price

    invested_capital = sum(invested_capitals_per_stock.values())
    unrealized_profit = current_value - invested_capital
    total_profit = cumulative_realized_pnl + unrealized_profit + cumulative_dividends

    dates.append(date)
    invested_capitals.append(invested_capital)
    realized_profits.append(cumulative_realized_pnl)
    unrealized_profits.append(unrealized_profit)
    total_profits.append(total_profit)
    dividends_list.append(cumulative_dividends)

# Create the DataFrame
df = pd.DataFrame(
    {
        "Date": dates,
        "Invested Capital": invested_capitals,
        "Realized Profit": realized_profits,
        "Unrealized Profit": unrealized_profits,
        "Dividends": dividends_list,
        "Total Profit": total_profits,
    }
)

# Set 'Date' as the index and remove time components
df["Date"] = df["Date"].dt.date
df.set_index("Date", inplace=True)

# Filter data from a specific date if needed
filtered_data = df[df.index >= earliest_date.date()]

st.subheader("Portfolio Over Time Data")
st.dataframe(filtered_data)

## Prepare data for stacked bar chart based on the option
if separate_profits:
    stacked_data = df[["Dividends", "Realized Profit", "Unrealized Profit"]]
    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    title = "Invested Capital, Realized & Unrealized Profit, and Dividends Over Time"
else:
    stacked_data = df[["Dividends", "Total Profit", "Invested Capital"]]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    title = "Invested Capital, Total Profit, and Dividends Over Time"

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
stacked_data.plot(kind="bar", stacked=True, color=colors, ax=ax)

# Formatting the plot
plt.title(title)
plt.xlabel("Date")
plt.ylabel("Amount (€)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Components")
plt.tight_layout()

st.subheader("Portfolio Over Time")
st.pyplot(fig)


# Option to download updated logs
st.sidebar.markdown("### Download Updated Logs:")
trade_csv = convert_df(trade_log)
dividend_csv = convert_df(dividend_log)

st.sidebar.download_button(
    label="Download Trade Log",
    data=trade_csv,
    file_name='trade_log.csv',
    mime='text/csv',
)

st.sidebar.download_button(
    label="Download Dividend Log",
    data=dividend_csv,
    file_name='dividend_log.csv',
    mime='text/csv',
)

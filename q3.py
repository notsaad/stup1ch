import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def q1(tickers):
    # date range
    start_date = "2000-01-01"
    end_date = "2024-10-31"

    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

    # Calculate monthly returns
    monthly_prices = data.resample("ME").last()  # Get monthly adjusted close prices
    monthly_returns = monthly_prices.pct_change().dropna()  # Calculate percentage change

    return monthly_returns

def q2(tickers, monthly_returns):
    avg_returns = []
    betas = []

    # Separate market returns
    market_returns = monthly_returns[market_ticker]
    stock_returns = monthly_returns[tickers]

# Compute AvgRet and Beta for each stock
    for stock in tickers:
        # Average return
        avg_return = stock_returns[stock].mean()
        avg_returns.append(avg_return)
        
        # CAPM Beta
        covariance = np.cov(stock_returns[stock], market_returns)[0, 1]
        market_variance = market_returns.var()
        beta = covariance / market_variance
        betas.append(beta)

# Create the result dataframe
    results_df = pd.DataFrame({
        "AvgRet": avg_returns,
        "Beta": betas
    }, index=tickers)

    return results_df

def q3(tickers, results_df):



    # Extract AvgRet and Beta from the dataframe
    avg_returns = results_df["AvgRet"]
    betas = results_df["Beta"]

    # Add a constant for the regression
    X = sm.add_constant(betas)  # Adds an intercept to the model
    y = avg_returns

    # Perform the regression
    model = sm.OLS(y, X).fit()

    # Extract the coefficient for Beta (not the constant)
    beta_coefficient = model.params["Beta"]

    # Format the coefficient to 2 decimal places
    return round((beta_coefficient*100), 2)

def q4(results_df):
    """
    Create a scatter plot of average returns against CAPM beta for individual stocks.
    """
    # Extract AvgRet and Beta
    avg_returns = results_df["AvgRet"]
    betas = results_df["Beta"]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(betas, avg_returns, color="blue", edgecolor="k", alpha=0.7)
    
    # Annotate points with stock tickers
    for ticker, beta, avg_ret in zip(results_df.index, betas, avg_returns):
        plt.text(beta, avg_ret, ticker, fontsize=9, ha="right", va="bottom")

    # Add labels, title, and grid
    plt.title("Scatter Plot of Average Returns vs. CAPM Beta", fontsize=16)
    plt.xlabel("CAPM Beta", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)
    plt.grid(alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add horizontal line at y=0
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Add vertical line at x=0

    # Show the plot
    plt.tight_layout()
    plt.show()

tickers = ["AXP", "B", "BAC", "BCS", "BMO", "C", "CL", "CNA", "JPM", "MTB", "NYT", "PFE", "PG", "SAN", "TD", "^GSPC"]
market_ticker = "^GSPC"
all_tickers = tickers + [market_ticker]

# Output the dataframe
print("Question 3, part i:")
monthly_returns = q1(tickers)
print(monthly_returns)

print("Question 3, part ii:")
results_df = q2(tickers, monthly_returns)
print(results_df)

print("Question 3, part iii:")
regression_coefficient = q3(tickers, results_df)
print(f"Regression coefficient (Beta): {regression_coefficient}")

print("Question 3, part iv:")
q4(results_df)

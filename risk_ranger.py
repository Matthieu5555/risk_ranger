import yfinance as yf
import pandas as pd
import numpy as np
import re
from colorama import Fore, Style, init
import scipy.stats

# comment code

def print_banner():
    # https://patorjk.com/software/taag/#p=display&f=Big&t=RiskRanger%20v2.2
    banner_text = r"""
  _____  _     _    _____                                    ___    ___  
 |  __ \(_)   | |  |  __ \                                  |__ \  |__ \ 
 | |__) |_ ___| | _| |__) |__ _ _ __   __ _  ___ _ __  __   __ ) |    ) |
 |  _  /| / __| |/ /  _  // _` | '_ \ / _` |/ _ \ '__| \ \ / // /    / / 
 | | \ \| \__ \   <| | \ \ (_| | | | | (_| |  __/ |     \ V // /_ _ / /_ 
 |_|  \_\_|___/_|\_\_|  \_\__,_|_| |_|\__, |\___|_|      \_/|____(_)____|
                                       __/ |                             
                                      |___/                                                    
    """
    print(Fore.WHITE + banner_text + Style.RESET_ALL)

def get_approximate_period(ticker):
    """
    This function prompts the user to input an approximate period.
    """
    period_regex = r"^\d*(d|mo|y|max)$"
    while True:
        period = input("Enter the period (e.g. XXd, XXmo, XXy, max): ")
        if re.match(period_regex, period):
            prices = yf.download(ticker, period=period)
            if not prices.empty:
                return prices, period
        print(Fore.RED + "Invalid period format. Here is an example: 10y" + Style.RESET_ALL)

def get_specific_period(ticker):
    """
    This function prompts the user to input a specific date range.
    """
    date_regex = r"^\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}$"
    while True:
        date_range = input("Enter the period (e.g. yyyy-mm-dd to yyyy-mm-dd): ")
        if re.match(date_regex, date_range):
            start_date, end_date = date_range.split(" to ")
            prices = yf.download(ticker, start=start_date, end=end_date)
            if not prices.empty:
                return prices, f"{start_date} to {end_date}"
        print(Fore.RED + "Invalid date range format. Here is an example: 2021-01-01 to 2021-12-31" + Style.RESET_ALL)

def get_valid_data(ticker):
    """
    This function prompts the user to choose between an approximate period or a specific period
    and returns the downloaded stock data and the chosen period.
    """
    while True:
        print("Would you like to get metrics using an approximate period from now (1) or input a specific period (2)?")
        choice = input("Enter 1 or 2: ")
        if choice == '1':
            return get_approximate_period(ticker)
        elif choice == '2':
            return get_specific_period(ticker)
        print(Fore.RED + "Invalid choice. Please enter 1 or 2." + Style.RESET_ALL)

def fetch_stock_data(prices):
    """
    This function fetches stock data using yfinance.
    The user chooses either price returns or total returns.
    It returns a DataFrame or Series of prices.
    """
    while True:
        choice = input("Choose 1 for price returns or 2 for total returns: ")
        #Return price returns (Close prices), which does not include dividends
        if choice == '1':
            close_prices = prices['Close']
            break
        # Return total returns (Adjusted Close prices), automatically includes dividends
        elif choice == '2':
            close_prices = prices['Adj Close']
            break
        else:
            print(Fore.RED + "Invalid choice, please enter 1 or 2." + Style.RESET_ALL)
    return close_prices

def calculate_total_return(prices):
    """
    This function calculates the total return over the period.
    It returns the total return as a single percentage.
    """
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    return total_return * 100

def get_daily_returns(prices):
    """
    This function calculates the percentage change of the stock prices.
    It returns a DataFrame with the percentage change.
    """
    r = prices.pct_change().dropna() * 100
    return r

def standard_deviation(r):
    return r.std(ddof=0)

def skewness(r):
    deviations_returns = r - r.mean()
    std_returns = r.std(ddof=0)
    cubed_deviations = (deviations_returns ** 3).mean()
    return cubed_deviations / (std_returns ** 3)

def kurtosis(r):
    deviations_returns = r - r.mean()
    std_returns = r.std(ddof=0)
    exp4_deviations = (deviations_returns ** 4).mean()
    return exp4_deviations / (std_returns ** 4)

def get_level():
    """
    This function repeatedly prompts the user to input a level of confidence for VaRs measurements.
    It checks if the input is an integer and returns the integer if valid.
    """
    while True:
        choice = input("Input a level of confidence for VaRs measurements (99, 95 ...): ")
        if choice.isdigit():
            level = 100 - int(choice)
            return level
        else:
            print(Fore.RED + "Invalid choice, please enter a valid integer." + Style.RESET_ALL)

def var_historic(r, level):
    """
    This function calculates the historical Value at Risk (VaR) at the given confidence level.
    """
    return -np.percentile(r, level)

def cvar_historic(r, level):
    """
    This function calculates the Conditional Value at Risk (CVaR) at the given confidence level.
    """
    is_beyond = r <= -var_historic(r, level=level)
    return -r[is_beyond].mean()

def var_cornishfisher(r, level):
    """
    Cornish-Fisher VaR, semi-gaussian modified with actual skewness & kurtosis of the distribution.
    """
    z = scipy.stats.norm.ppf(level / 100)
    s = skewness(r)
    k = kurtosis(r)
    z = (z +
         (z**2 - 1) * s / 6 +
         (z**3 - 3 * z) * (k - 3) / 24 -
         (2 * z**3 - 5 * z) * (s**2) / 36
         )
    return -(r.mean() + z * r.std(ddof=0))

def get_max_drawdown(prices: pd.Series):
    """
    Takes a time series of asset prices.
    Returns the maximum drawdown and its date.
    """
    # Calculate wealth index
    wealth_index = prices / prices.iloc[0]
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Find the maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Find the date of the maximum drawdown
    max_drawdown_date = drawdowns.idxmin()
    
    return max_drawdown, max_drawdown_date
def annualize_vol(r, periods_per_year):
    # Annualizes the volatility (standard deviation) of returns.
    return r.std() * (periods_per_year ** 0.5)

def annualize_rets(r, periods_per_year):
    # Annualizes the returns from daily returns, assuming compounding.
    compounded_growth = (1 + r / 100).prod()
    n_periods = r.shape[0]
    return (compounded_growth ** (periods_per_year / n_periods) - 1) * 100

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    # Calculates the Sharpe Ratio, adjusting for the risk-free rate.
    rf_per_period = ((1 + riskfree_rate / 100) ** (1 / periods_per_year) - 1) * 100
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def get_annual_risk_free_rate():
    # Retrieves and annualizes the risk-free rate from the 13-week T-bill index.
    ticker = '^IRX'
    period = '10y'
    prices = yf.download(ticker, period=period)
    average_rate = prices['Close'].mean()
    annual_risk_free_rate = average_rate * 252 / 100
    return annual_risk_free_rate


def main():
    init()
    print_banner()

    ticker = input("Enter the stock ticker symbol: ")
    prices, period = get_valid_data(ticker)
    
    start_date = prices.index[0].strftime('%Y-%m-%d')
    end_date = prices.index[-1].strftime('%Y-%m-%d')
    
    close_prices = fetch_stock_data(prices)
    total_return = calculate_total_return(close_prices)
    daily_returns = get_daily_returns(close_prices)
    volatility = standard_deviation(daily_returns)
    level = get_level()
    
    var_hist = var_historic(daily_returns, level)
    cvar_hist = cvar_historic(daily_returns, level)
    var_cornish = var_cornishfisher(daily_returns, level)
    max_drawdown, drawdown_date = get_max_drawdown(close_prices)

    periods_per_year = 252
    annualized_returns = annualize_rets(daily_returns, periods_per_year)
    annualized_vol = annualize_vol(daily_returns, periods_per_year)
    returns_vol_ratio = annualized_returns / annualized_vol

    annual_risk_free_rate = get_annual_risk_free_rate()
    excess_returns = annualized_returns - annual_risk_free_rate
    sharpe = sharpe_ratio(daily_returns, annual_risk_free_rate, periods_per_year)
    
    print(f"\nThe data used started in: {start_date} up to {end_date}")
    print(f"Total Returns: {total_return:.4f}%")
    print(f"Average Annual Returns: {annualized_returns:.4f}%")
    print(f"Return to Volatility Ratio: {returns_vol_ratio:.4f}")
    print(f"Average Annual Risk-Free Rate: {annual_risk_free_rate:.4f}%")
    print(f"Annual Excess Returns: {excess_returns:.4f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print("")
    print(f"Volatility: {volatility:.4f}%")
    print(f"Historic VaR: {var_hist:.4f}%")
    print(f"Historic CVaR: {cvar_hist:.4f}%")
    print(f"Cornish-Fischer VaR: {var_cornish:.4f}%")
    print(f"The maximum drawdown is {max_drawdown:.1%} and it occurred on the {drawdown_date.date()}")
    print("")

if __name__ == "__main__":
    main()

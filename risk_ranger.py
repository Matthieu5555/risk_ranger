import yfinance as yf
import pandas as pd
import numpy as np
import re
from colorama import Fore, Style, init
import scipy.stats

def print_banner():
    #https://patorjk.com/software/taag/#p=display&f=Big&t=RiskRanger%20v2.1
    banner_text = r"""
  _____  _     _    _____                                    ___   __ 
 |  __ \(_)   | |  |  __ \                                  |__ \ /_ |
 | |__) |_ ___| | _| |__) |__ _ _ __   __ _  ___ _ __  __   __ ) | | |
 |  _  /| / __| |/ /  _  // _` | '_ \ / _` |/ _ \ '__| \ \ / // /  | |
 | | \ \| \__ \   <| | \ \ (_| | | | | (_| |  __/ |     \ V // /_ _| |
 |_|  \_\_|___/_|\_\_|  \_\__,_|_| |_|\__, |\___|_|      \_/|____(_)_|
                                       __/ |                          
                                      |___/                           
    """
    print(Fore.WHITE + banner_text + Style.RESET_ALL)

def get_valid_data(ticker):
    """
    Prompt the user for a time period and download the data.
    """
    period_regex = r"^\d*(d|mo|y|max)$"  # This regular expression filters for what yfinance expects
    while True:
        period = input("Enter the period (e.g., '10y', '20y', 'max'): ")
        if re.match(period_regex, period):
            prices = yf.download(ticker, period=period)  # downloads a dataframe
            if not prices.empty:
                return prices, period
        print(Fore.RED + "Invalid period format or no data for the given period. Please try again." + Style.RESET_ALL)

def fetch_stock_data(prices):
    """
    This function fetches stock data using yfinance.
    The user chooses either price returns or total returns.
    It returns a DataFrame or Series of prices.
    """
    while True:
        choice = input("Choose 1 for price returns or 2 for total returns: ")

        if choice == '1':
            # Return price returns (Close prices), which does not include dividends
            close_prices = prices['Close']
            break
        elif choice == '2':
            # Return total returns (Adjusted Close prices), automatically includes dividends
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
    """
    Calculate the standard deviation of returns.
    """
    return r.std(ddof=0)

def skewness(r):
    """
    Computes the skewness of the supplied series or DataFrame.
    Returns a float or a series.
    """
    deviations_returns = r - r.mean()
    std_returns = r.std(ddof=0)
    cubed_deviations = (deviations_returns ** 3).mean()
    return cubed_deviations / (std_returns ** 3)

def kurtosis(r):
    """
    Computes the kurtosis of the series or DataFrame.
    Returns a float or series.
    """
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

        # Check if the choice is an integer
        if choice.isdigit():
            level = 100 - int(choice)
            return level
        else:
            print(Fore.RED + "Invalid choice, please enter a valid integer." + Style.RESET_ALL)

def var_historic(r, level):
    """
    Calculate Historic VaR.
    """
    if isinstance(r, pd.DataFrame):  # if it is a dataframe, call the function on every column
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be series or DataFrame")

def cvar_historic(r, level):
    """
    Calculate Historic CVaR, aka beyond VaR.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):  # if it is a dataframe, call the function on every column
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be series or DataFrame")

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

def drawdown(prices: pd.Series):
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

def get_annualized_rets_vol(r):
    """
    Calculate the annualized returns and volatility.
    Assumes r is the daily return percentage.
    """
    daily_return_mean = r.mean() / 100  # Convert percentage to decimal
    annualized_returns = ((1 + daily_return_mean) ** 252 - 1)*100 # 252 trading days, * 100 because its printed later as a percentage
    annualized_vol = (standard_deviation(r) / 100 * np.sqrt(252))*100  # Convert percentage to decimal, annualize, * 100 for a fair comparison to annualized returns
    return annualized_returns, annualized_vol

def get_return_vol_ratio(annualized_returns, annualized_vol):
    """
    Calculate the return to volatility ratio (Sharpe Ratio without risk-free rate).
    """
    return annualized_returns / annualized_vol

def get_annual_risk_free_rate():
    """
    This function downloads the data for ^IRX, calculates the average value,
    and returns it as the annual risk-free rate.
    """
    ticker = '^IRX'
    period = '10y'
    prices = yf.download(ticker, period=period)
    average_rate = prices['Close'].mean()
    annual_risk_free_rate = average_rate * 252 / 100  # Convert to annual percentage
    return annual_risk_free_rate

def get_excess_returns(annualized_return, annual_risk_free_rate):
    excess_returns = annualized_return - annual_risk_free_rate
    return excess_returns

def get_sharpe_ratio(excess_return, annualized_vol):
    return excess_return / annualized_vol

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
    max_drawdown, drawdown_date = drawdown(close_prices)

    annualized_returns, annualized_vol = get_annualized_rets_vol (daily_returns)
    returns_vol_ratio = get_return_vol_ratio(annualized_returns, annualized_vol)

    annual_risk_free_rate = get_annual_risk_free_rate()
    excess_returns = get_excess_returns(annualized_returns, annual_risk_free_rate)
    sharpe_ratio = get_sharpe_ratio (excess_returns, annualized_vol)
    
    print(f"\nThe data used started in: {start_date} up to {end_date}")
    print(f"Total Returns: {total_return:.4f}%")
    print(f"Average Annual Returns: {annualized_returns:.4f}%")
    print(f"Return to Volatility Ratio: {returns_vol_ratio:.4f}")
    print(f"Average Annual Risk-Free Rate: {annual_risk_free_rate:.4f}%")
    print(f"Annual Excess returns: {excess_returns:.4f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print("")
    print(f"Volatility: {volatility:.4f}%")
    print(f"Historic VaR: {var_hist:.4f}%")
    print(f"Historic CVaR: {cvar_hist:.4f}%")
    print(f"Cornish-Fischer VaR: {var_cornish:.4f}%")
    print(f"The maximum drawdown is {max_drawdown:.1%} and it occurred on the {drawdown_date.date()}")
    print("")

if __name__ == "__main__":
    main()

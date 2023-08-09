# Portfolio Optimization with Mean-Variance Analysis

This project demonstrates portfolio optimization using Mean-Variance Analysis. It utilizes financial data fetched from Yahoo Finance using the `pandas_datareader` library and optimizes portfolio weights to achieve desired returns while minimizing risk.

## Overview

This script fetches financial data for a list of assets (e.g., stocks) from Yahoo Finance, calculates their historical returns and volatility, and performs various portfolio optimization tasks using Mean-Variance Analysis. The goal is to find optimal asset allocations that maximize returns given a certain level of risk or minimize risk for a given expected return.

## Features

- Fetches financial data from Yahoo Finance for specified assets.
- Calculates historical returns and volatility for each asset.
- Visualizes the relationship between returns and volatility of the assets.
- Performs portfolio optimization to find the optimal asset allocation for maximum return or minimum risk.
- Plots random portfolios and portfolios with target returns on a risk-return graph.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Agu-Ribo/Portfolio-Optimization.git

2. Navigate to the project directory:
   
    ```bash
    cd Portfolio-Optimization.py

3. Install the required Python packages:

    ```bash
    pip install numpy matplotlib pandas pandas_datareader cvxopt pypfopt

## Usage
1. Open the portfolio_optimization.py script in your preferred Python IDE or editor.
2. Modify the tickers list to include the symbols of the assets you want to analyze.
3. Adjust the start and end dates in the pdr.get_data_yahoo function to specify the data retrieval period.
4. Run the script. It will display visualizations of asset returns, volatility, and portfolio optimization results.

## Configuration
No additional configuration is required for this script. However, you can explore the code to adjust parameters related to portfolio optimization, such as the number of random portfolios to generate or the target returns.

## Contributing
Contributions to this project are welcome! If you have ideas for improvements or bug fixes, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

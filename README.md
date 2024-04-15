# Chicago Trading Competition 2024
## Team UChicago x UMass Amherst
- <a href="https://github.com/coolkite/">Divyansh Shivashok</a>, 2024 AI / ML SWE Intern @ Microsoft, 2x USNCO (US Chemistry Olympiad) Top 50, AI Research @ UMass Computer Graphics Lab, MIT CSAIL
- <a href="https://github.com/dmtrung14/">Trung Dang</a>, 2x Vietnam Math Olympiad Top 25, AI Research @ UMass Dynamic and Autonomous Robotic Systems (DARoS) Lab
- <a href="https://github.com/zaranip/">Zara Nip</a>, 2024 Risk Quant Intern @ PIMCO, AI Break Through Tech Scholar, Girls Who Invest Scholar, AnitaB.org Scholar, UChicago Financial Markets Scholar

## General Thoughts
This year's competition included a live market making simulation (Case 1) and a portfolio optimization case (Case 2) from April 12-13. This was the first time some of us had ever attempted to anything related to quantitative finance - and also the first time that UMass Amherst was invited to the competition in all 12 years of its running (hopefully not the last!).

UMass Amherst has historically sent very few people to quantitative funds, noted by the lack of inclusion on university dropdown options on some firms' career pages. Both of our UMass members are incredibly talented, and we hope to continue to breakthrough these barriers.

If you are a recruiter and / or looking to hire in the quantitative finance field, please navigate to the resumes folder of this respository to find our resumes and contact information. We are currently recruiting for 2025 internships (Trung, Zara) and full-time positions (Divyansh).

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#case-1-market-making">Case 1</a></li>
    <li><a href="#case-2-portfolio-optimization">Case 2</a></li>
    <li><a href="#contact">Recruiting</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

Instructions on setting up our project locally.
To get a local copy up and running follow these simple example steps.


### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/zaranip/Chicago-Trading-Competition-2024/
   ```
2. Navigate into the directory
   ```sh
   cd Chicago-Trading-Competition-2024
   ```
3. Install and activate relevant packages
  ```sh
  conda env create --name utc --file=environment.yaml
  conda activate utc
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Case 1 -->
## Case 1: Market Making

Strategies
1. **Penny in, Penny out with Levels**: This strategy involves placing orders at the best bid and ask prices, with the aim of capturing the bid-ask spread. The bot continuously adjusts its orders based on predefined levels to optimize its position in the order book.
2. **ETF Arbitrage**: The bot monitors the prices of exchange-traded funds (ETFs) and their underlying assets. It identifies and exploits price discrepancies between the ETF and its components, taking advantage of arbitrage opportunities.
3. **GUI Interface and Accessory Strategies**: The bot includes a graphical user interface (GUI) that allows users to monitor its performance and adjust settings in real-time. Additionally, the bot employs accessory strategies, such as placing bogus bids, to manipulate the market and gain an advantage over other participants.

The GUI allowed us to control fade (rate of selling / buying assets), edge (profit margin sensitivity), slack, and minimum margin. These can be found in our "params_gui.py" file.

### Challenges
During the development and deployment of the bot, we encountered a significant challenge posed by "hitter bots". These bots aggressively hit our orders, making it difficult for our market-making bot to function effectively. The hitter bots' actions disrupted our bot's ability to maintain its desired position in the order book and execute trades as intended.

To mitigate the impact of hitter bots, we implemented various countermeasures and refined our algorithms. However, the presence of these bots highlighted the importance of robust error handling, risk management, and continuous monitoring when operating in a highly competitive and fast-paced trading environment.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Case 2 -->
## Case 2: Portfolio Optimization

### Portfolio Optimization: Case 2

In this case study, we focus on the portfolio optimization process, specifically highlighting the passive-aggressive mean reversion strategy and the insights gained from exploratory data analysis (EDA).

#### Passive-Aggressive Mean Reversion Strategy

During the portfolio optimization phase, we implemented a passive-aggressive mean reversion strategy. This strategy aims to capitalize on the tendency of asset prices to revert to their long-term average over time. The strategy involves the following steps:

1. **Identification of Mean-Reverting Assets**: Through extensive analysis of historical price data, we identified assets that exhibited mean-reverting behavior. These assets were characterized by prices that tended to oscillate around a long-term average, deviating from it in the short term but eventually reverting back to the mean.

2. **Entry and Exit Signals**: We developed a set of entry and exit signals based on the deviation of asset prices from their long-term average. When an asset's price significantly deviated from its mean, either above or below, it triggered an entry signal. Conversely, when the price reverted back towards the mean, it generated an exit signal.

3. **Position Sizing**: The strategy employed a passive-aggressive approach to position sizing. When an entry signal was triggered, the strategy took a passive position, allocating a portion of the portfolio to the asset. If the price continued to deviate from the mean, the strategy aggressively increased the position size, capitalizing on the expected mean reversion.

#### Insights from Exploratory Data Analysis (EDA)

During the exploratory data analysis phase, we made a crucial observation that supported the implementation of the passive-aggressive mean reversion strategy. While analyzing historical price data, we noticed that certain assets exhibited strong mean-reverting characteristics.

Through visual inspection of price charts and statistical analysis, we identified patterns of prices oscillating around a central value over time. These assets displayed a tendency to deviate from their long-term average in the short term, but consistently reverted back to the mean over a longer horizon.

The EDA process involved the following steps:

1. **Data Visualization**: We created price charts and plotted the long-term average or moving average of the asset prices. This visual representation helped us identify the mean-reverting behavior of the assets.

2. **Statistical Tests**: We performed statistical tests, such as the Augmented Dickey-Fuller (ADF) test, to assess the stationarity of the price series. Stationary series are more likely to exhibit mean reversion, as they have a constant mean and variance over time.

3. **Autocorrelation Analysis**: We examined the autocorrelation of the price series to determine the presence of mean reversion. Negative autocorrelation, particularly at longer lags, indicated a tendency for prices to revert to their mean.

The insights gained from the EDA process provided strong evidence of mean-reverting behavior in certain assets. This information was instrumental in the development and implementation of the passive-aggressive mean reversion strategy within our portfolio optimization framework.

By leveraging the mean-reverting characteristics of these assets, we aimed to capitalize on short-term deviations from the long-term average, potentially generating profits as prices reverted back to their equilibrium levels.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


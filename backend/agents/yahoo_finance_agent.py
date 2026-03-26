import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceAgent:
    """
    Agent 1: Fetches stock price data from Yahoo Finance.
    Responsible for retrieving current stock price, historical data, and basic metrics.
    """

    def __init__(self):
        self.agent_name = "YahooFinanceDataFetcher"

    def get_stock_data(self, ticker: str, days_back: int = 30) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            current_data = self._get_current_price(stock)
            historical_data = self._get_historical_data(ticker, days_back)
            metrics = self._get_key_metrics(stock)
            recommendations = self._get_analyst_data(stock)

            # Get news sentiment (optional, from Yahoo Finance)
            news = self._get_recent_news(ticker)

            result = {
                "agent": self.agent_name,
                "ticker": ticker.upper(),
                "timestamp": datetime.now().isoformat(),
                "current_data": current_data,
                "historical_data": historical_data,
                "key_metrics": metrics,
                "analyst_recommendations": recommendations,
                "recent_news": news,
                "success": True
            }

            logger.info(f"Successfully fetched data for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return {
                "agent": self.agent_name,
                "ticker": ticker,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_current_price(self, stock) -> Dict:
        """Extract current price and real-time metrics."""
        try:
            # Get current price info
            info = stock.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose')

            day_change = None
            day_change_percent = None
            if current_price and previous_close:
                day_change = current_price - previous_close
                day_change_percent = (day_change / previous_close * 100)

            return {
                "current_price": current_price,
                "previous_close": previous_close,
                "day_change": day_change,
                "day_change_percent": round(day_change_percent, 2) if day_change_percent else None,
                "day_high": info.get('dayHigh'),
                "day_low": info.get('dayLow'),
                "volume": info.get('volume'),
                "avg_volume": info.get('averageVolume'),
                "market_cap": info.get('marketCap'),
                "currency": info.get('currency', 'USD')
            }
        except Exception as e:
            logger.warning(f"Error getting current price: {e}")
            return {}

    def _get_historical_data(self, ticker: str, days_back: int) -> Dict:
        """Fetch historical price data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True  # Explicitly set to handle the warning
                )

            if hist.empty:
                return {"error": "No historical data available", "summary": {}}

            hist_data = {
                "dates": hist.index.strftime('%Y-%m-%d').tolist() if not hist.empty else [],
                "prices": {
                    "open": hist['Open'].squeeze().tolist() if 'Open' in hist.columns else [],
                    "high": hist['High'].squeeze().tolist() if 'High' in hist.columns else [],
                    "low": hist['Low'].squeeze().tolist() if 'Low' in hist.columns else [],
                    "close": hist['Close'].squeeze().tolist() if 'Close' in hist.columns else [],
                    "volume": hist['Volume'].squeeze().tolist() if 'Volume' in hist.columns else []
                },
                "summary": {
                    "start_price": hist['Close'].iloc[0].item() if not hist.empty and 'Close' in hist.columns else None,
                    "end_price": hist['Close'].iloc[-1].item() if not hist.empty and 'Close' in hist.columns else None,
                    "max_price": hist['High'].max().item() if not hist.empty and 'High' in hist.columns else None,
                    "min_price": hist['Low'].min().item() if not hist.empty and 'Low' in hist.columns else None,
                    "avg_price": hist['Close'].mean().item() if not hist.empty and 'Close' in hist.columns else None,
                    "total_return_percent": ((hist['Close'].iloc[-1].item() - hist['Close'].iloc[0].item()) / hist['Close'].iloc[0].item() * 100) if not hist.empty and 'Close' in hist.columns and len(hist) > 1 else None,
                    "volatility": hist['Close'].pct_change().std().item() * 100 if not hist.empty and 'Close' in hist.columns and len(hist) > 1 else None
                }
            }

            return hist_data

        except Exception as e:
            logger.warning(f"Error getting historical data: {e}")
            return {"error": str(e), "summary": {}}

    def _get_key_metrics(self, stock) -> Dict:
        """Extract key financial metrics."""
        try:
            info = stock.info

            metrics = {
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "dividend_yield": info.get('dividendYield'),
                "beta": info.get('beta'),
                "eps": info.get('trailingEps'),
                "forward_eps": info.get('forwardEps'),
                "revenue_growth": info.get('revenueGrowth'),
                "profit_margins": info.get('profitMargins'),
                "operating_margins": info.get('operatingMargins'),
                "return_on_equity": info.get('returnOnEquity'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio')
            }

            for key, value in metrics.items():
                if value is not None and key in ['profit_margins', 'operating_margins', 'revenue_growth', 'dividend_yield']:
                    metrics[key] = round(value * 100, 2)  # Convert to percentage

            metrics = {k: v for k, v in metrics.items() if v is not None}

            return metrics

        except Exception as e:
            logger.warning(f"Error getting key metrics: {e}")
            return {}

    def _get_analyst_data(self, stock) -> Dict:
        """Get analyst recommendations and price targets."""
        try:
            info = stock.info

            recommendations = {
                "target_mean_price": info.get('targetMeanPrice'),
                "target_high_price": info.get('targetHighPrice'),
                "target_low_price": info.get('targetLowPrice'),
                "number_of_analysts": info.get('numberOfAnalystOpinions'),
                "recommendation_mean": info.get('recommendationMean'),  # 1=Strong Buy, 5=Sell
                "recommendation_key": self._map_recommendation(info.get('recommendationMean'))
            }

            return recommendations

        except Exception as e:
            logger.warning(f"Error getting analyst data: {e}")
            return {}

    def _get_recent_news(self, ticker: str) -> List[Dict]:
        """Fetch recent news articles for the ticker."""
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news

            recent_news = []
            if news_data:
                for item in news_data[:5]:
                    # Handle different possible structures of news data
                    title = item.get('title')
                    if not title:
                        # Try alternative field names
                        title = item.get('content', {}).get('title', 'No title available')

                    publisher = item.get('publisher')
                    if not publisher:
                        publisher = item.get('content', {}).get('provider', 'Unknown')

                    link = item.get('link')
                    if not link:
                        link = item.get('canonicalUrl', {}).get('url', '')

                    publish_time = item.get('providerPublishTime')
                    if publish_time:
                        try:
                            publish_time = datetime.fromtimestamp(publish_time).isoformat()
                        except:
                            publish_time = None

                    if title and title != 'No title available':  # Only add if we have meaningful data
                        recent_news.append({
                            "title": title,
                            "publisher": publisher,
                            "link": link,
                            "publish_time": publish_time
                        })

            return recent_news

        except Exception as e:
            logger.warning(f"Error getting news: {e}")
            return []

    def _map_recommendation(self, recommendation_mean: Optional[float]) -> str:
        """Map numeric recommendation to text."""
        if recommendation_mean is None:
            return "N/A"

        recommendation_map = {
            1: "Strong Buy",
            2: "Buy",
            3: "Hold",
            4: "Underperform",
            5: "Sell"
        }

        rounded = round(recommendation_mean)
        return recommendation_map.get(rounded, "Hold")

    def format_for_next_agent(self, data: Dict) -> str:
        """
        Format the output for consumption by Agent 2.
        Returns a clean, structured summary.
        """
        if not data.get("success"):
            return f"Error fetching data for {data['ticker']}: {data.get('error')}"

        ticker = data['ticker']
        current = data.get('current_data', {})
        metrics = data.get('key_metrics', {})
        recommendations = data.get('analyst_recommendations', {})
        historical = data.get('historical_data', {}).get('summary', {})

        current_price = current.get('current_price')
        current_price_str = f"${current_price:,.2f}" if current_price is not None else "N/A"

        day_change = current.get('day_change_percent')
        day_change_str = f"{day_change:.2f}%" if day_change is not None else "N/A"

        volume = current.get('volume')
        volume_str = f"{volume:,}" if volume is not None else "N/A"
        avg_volume = current.get('avg_volume')
        avg_volume_str = f"{avg_volume:,}" if avg_volume is not None else "N/A"

        market_cap = current.get('market_cap')
        market_cap_str = f"${market_cap:,.0f}" if market_cap is not None else "N/A"

        start_price_display = "N/A"
        if historical.get('start_price') is not None:
            start_price_display = f"${historical['start_price']:,.2f}"

        end_price_display = "N/A"
        if historical.get('end_price') is not None:
            end_price_display = f"${historical['end_price']:,.2f}"

        total_return_display = "N/A"
        if historical.get('total_return_percent') is not None:
            total_return_display = f"{historical['total_return_percent']:.2f}%"

        volatility_display = "N/A"
        if historical.get('volatility') is not None:
            volatility_display = f"{historical['volatility']:.2f}%"

        target_mean_price_display = "N/A"
        if recommendations.get('target_mean_price') is not None:
            target_mean_price_display = f"${recommendations['target_mean_price']:,.2f}"

        target_low_price = recommendations.get('target_low_price')
        target_high_price = recommendations.get('target_high_price')
        target_range_display = "N/A"
        if target_low_price is not None and target_high_price is not None:
            target_range_display = f"${target_low_price:,.2f} - ${target_high_price:,.2f}"


        formatted = f"""
=== YAHOO FINANCE DATA FOR {ticker} ===

CURRENT PRICE: {current_price_str}
DAILY CHANGE: {day_change_str}
VOLUME: {volume_str} (Avg: {avg_volume_str})
MARKET CAP: {market_cap_str}

KEY METRICS:
- P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
- Forward P/E: {metrics.get('forward_pe', 'N/A')}
- EPS: ${metrics.get('eps', 'N/A')}
- Profit Margin: {metrics.get('profit_margins', 'N/A') or 'N/A'}%
- Beta: {metrics.get('beta', 'N/A')}

PERFORMANCE (Last 30 Days):
- Start Price: {start_price_display}
- End Price: {end_price_display}
- Total Return: {total_return_display}
- Volatility: {volatility_display}

ANALYST COVERAGE:
- Price Target: {target_mean_price_display}
- Target Range: {target_range_display}
- Analyst Rating: {recommendations.get('recommendation_key', 'N/A')}
- Analysts Covering: {recommendations.get('number_of_analysts', 'N/A')}

RECENT NEWS:
{self._format_news(data.get('recent_news', []))}
"""
        return formatted.strip()

    def _format_news(self, news_items: List[Dict]) -> str:
        """Format news items for display."""
        if not news_items:
            return "No recent news available"

        formatted = ""
        for i, item in enumerate(news_items, 1):
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            formatted += f"{i}. {title} - {publisher}\n"

        return formatted.strip()

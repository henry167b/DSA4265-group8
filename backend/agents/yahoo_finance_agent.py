import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceAgent:
    """
    Agent 1: Fetches stock price data from Yahoo Finance.
    Responsible for retrieving current stock price, historical data, and basic metrics.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance agent."""
        self.agent_name = "YahooFinanceDataFetcher"
        
    def get_stock_data(self, ticker: str, days_back: int = 30) -> Dict:
        """
        Fetch comprehensive stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')
            days_back: Number of days of historical data to retrieve
            
        Returns:
            Dictionary containing stock data, metrics, and metadata
        """
        try:
            stock = yf.Ticker(ticker)
            
            current_data = self._get_current_price(stock)
            
            historical_data = self._get_historical_data(ticker, days_back)
            
            metrics = self._get_key_metrics(stock)
            
            recommendations = self._get_analyst_data(stock)
            
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
            info = stock.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose')
            day_change = current_price - previous_close if current_price and previous_close else None
            day_change_percent = (day_change / previous_close * 100) if day_change and previous_close else None
            
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
            
            hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if hist.empty:
                return {"error": "No historical data available"}
            
            hist_data = {
                "dates": hist.index.strftime('%Y-%m-%d').tolist() if not hist.empty else [],
                "prices": {
                    "open": hist['Open'].tolist() if not hist.empty else [],
                    "high": hist['High'].tolist() if not hist.empty else [],
                    "low": hist['Low'].tolist() if not hist.empty else [],
                    "close": hist['Close'].tolist() if not hist.empty else [],
                    "volume": hist['Volume'].tolist() if not hist.empty else []
                },
                "summary": {
                    "start_price": hist['Close'].iloc[0] if not hist.empty else None,
                    "end_price": hist['Close'].iloc[-1] if not hist.empty else None,
                    "max_price": hist['High'].max() if not hist.empty else None,
                    "min_price": hist['Low'].min() if not hist.empty else None,
                    "avg_price": hist['Close'].mean() if not hist.empty else None,
                    "total_return_percent": ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if not hist.empty else None,
                    "volatility": hist['Close'].pct_change().std() * 100 if not hist.empty else None
                }
            }
            
            return hist_data
            
        except Exception as e:
            logger.warning(f"Error getting historical data: {e}")
            return {"error": str(e)}
    
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
            news = stock.news
            
            recent_news = []
            for item in news[:5]:
                recent_news.append({
                    "title": item.get('title'),
                    "publisher": item.get('publisher'),
                    "link": item.get('link'),
                    "publish_time": datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat() if item.get('providerPublishTime') else None
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
        
        formatted = f"""
=== YAHOO FINANCE DATA FOR {ticker} ===

CURRENT PRICE: ${current.get('current_price', 'N/A')}
DAILY CHANGE: {current.get('day_change_percent', 'N/A')}%
VOLUME: {current.get('volume', 'N/A')} (Avg: {current.get('avg_volume', 'N/A')})

KEY METRICS:
- P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
- Forward P/E: {metrics.get('forward_pe', 'N/A')}
- EPS: ${metrics.get('eps', 'N/A')}
- Profit Margin: {metrics.get('profit_margins', 'N/A')}%
- Beta: {metrics.get('beta', 'N/A')}

PERFORMANCE (Last 30 Days):
- Start Price: ${historical.get('start_price', 'N/A')}
- End Price: ${historical.get('end_price', 'N/A')}
- Total Return: {historical.get('total_return_percent', 'N/A')}%
- Volatility: {historical.get('volatility', 'N/A')}%

ANALYST COVERAGE:
- Price Target: ${recommendations.get('target_mean_price', 'N/A')} (Range: ${recommendations.get('target_low_price', 'N/A')} - ${recommendations.get('target_high_price', 'N/A')})
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
            formatted += f"{i}. {item.get('title')} - {item.get('publisher')}\n"
        
        return formatted

import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging
import warnings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceAgent:
    """
    Agent 1: Fetches stock price data from Yahoo Finance AND SEC EDGAR filings.
    Responsible for retrieving current stock price, historical data, basic metrics,
    and 10-Q filings with financial data from SEC.
    """

    def __init__(self, user_email: str = "e0968923@u.nus.edu"):
        """
        Initialize the Yahoo Finance agent with SEC EDGAR support.

        Args:
            user_email: Email address for SEC EDGAR User-Agent (required by SEC)
        """
        self.agent_name = "YahooFinanceDataFetcher"
        self.user_email = user_email

        # SEC EDGAR requires a specific User-Agent format with email

        self.sec_headers = {
            'User-Agent': f'AI Investor Agent/1.0 ({user_email})',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }

        self.cik_cache = {}
        self.company_name_cache = {}

    # ============ YAHOO FINANCE METHODS ============

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

            logger.info(f"Successfully fetched Yahoo Finance data for {ticker}")
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

            # Suppress the FutureWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                if isinstance(hist.columns, pd.MultiIndex):
                  hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                return {"error": "No historical data available", "summary": {}}

            hist_data = {
                "dates": hist.index.strftime('%Y-%m-%d').tolist() if not hist.empty else [],
                "prices": {
                    "open": hist['Open'].values.tolist() if 'Open' in hist.columns else [],
                    "high": hist['High'].values.tolist() if 'High' in hist.columns else [],
                    "low": hist['Low'].values.tolist() if 'Low' in hist.columns else [],
                    "close": hist['Close'].values.tolist() if 'Close' in hist.columns else [],
                    "volume": hist['Volume'].values.tolist() if 'Volume' in hist.columns else []
                },
                "summary": {
                    "start_price": float(hist['Close'].iloc[0]) if not hist.empty and 'Close' in hist.columns else None,
                    "end_price": float(hist['Close'].iloc[-1]) if not hist.empty and 'Close' in hist.columns else None,
                    "max_price": float(hist['High'].max()) if not hist.empty and 'High' in hist.columns else None,
                    "min_price": float(hist['Low'].min()) if not hist.empty and 'Low' in hist.columns else None,
                    "avg_price": float(hist['Close'].mean()) if not hist.empty and 'Close' in hist.columns else None,
                    "total_return_percent": float(((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)) if not hist.empty and 'Close' in hist.columns and len(hist) > 1 else None,
                    "volatility": float(hist['Close'].pct_change().std() * 100) if not hist.empty and 'Close' in hist.columns and len(hist) > 1 else None
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
                    metrics[key] = round(value * 100, 2)

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
                "recommendation_mean": info.get('recommendationMean'),
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
                    title = item.get('title')
                    if not title:
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

                    if title and title != 'No title available':
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

    # ============ SEC EDGAR METHODS ============

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        """
        Convert a stock ticker to SEC CIK (10-digit number with leading zeros).

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA')

        Returns:
            10-digit CIK string or None if not found
        """
        ticker = ticker.upper()

        if ticker in self.cik_cache:
            return self.cik_cache[ticker]

        try:
            headers = {
                'User-Agent': f'AI Investor Agent/1.0 ({self.user_email})', # Consistent User-Agent
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }

            # Use the correct SEC company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"

            time.sleep(0.5)

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 403:
                logger.error(f"SEC request blocked (403). Status code: {response.status_code}")
                return None

            response.raise_for_status()

            companies = response.json()

            for company in companies.values():
                if company.get('ticker') == ticker:
                    cik = str(company.get('cik_str', '')).zfill(10)
                    self.cik_cache[ticker] = cik
                    self.company_name_cache[ticker] = company.get('title', 'Unknown')
                    logger.info(f"Found CIK for {ticker}: {cik}")
                    return cik

            logger.warning(f"No CIK found for ticker: {ticker}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error converting ticker to CIK: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error converting ticker to CIK: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error converting ticker to CIK: {e}")
            return None

    def get_recent_10q_filings(
        self,
        ticker: str,
        num_quarters: int = 1,
        include_document_html: bool = False
    ) -> Dict:
        """
        Fetch recent 10-Q filings for a company using SEC Submissions API.

        Args:
            ticker: Stock ticker symbol
            num_quarters: Number of quarters to retrieve (default 1)
            include_document_html: Whether to download and attach the filing HTML

        Returns:
            Dictionary with filing metadata, URLs, and optionally filing HTML
        """
        cik = self.ticker_to_cik(ticker)
        if not cik:
            return {"error": f"Could not find CIK for {ticker}", "filings": [], "success": False}

        try:
            time.sleep(0.5)

            # Submissions API endpoint 
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            logger.info(f"SEC Submissions API URL being requested: {url}")
            response = requests.get(url, headers=self.sec_headers, timeout=15)
            response.raise_for_status()

            data = response.json()

            filings = data.get('filings', {}).get('recent', {})
            if not filings:
                return {"error": "No filings found", "filings": [], "success": False}

            # Filter for 10-Q filings
            ten_q_filings = []
            form_types = filings.get('form', [])
            accession_numbers = filings.get('accessionNumber', [])
            filing_dates = filings.get('filingDate', [])
            primary_documents = filings.get('primaryDocument', [])

            company_name = data.get('name', self.company_name_cache.get(ticker, 'Unknown'))
            if ticker not in self.company_name_cache:
                self.company_name_cache[ticker] = company_name

            for i, form in enumerate(form_types):
                if form == '10-Q' and len(ten_q_filings) < num_quarters:
                    accession = accession_numbers[i].replace('-', '')

                    # Construct URL for the filing document
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_documents[i]}"

                    filing_record = {
                        "filing_date": filing_dates[i],
                        "accession_number": accession_numbers[i],
                        "form_type": "10-Q",
                        "document_url": document_url,
                        "quarter": self._get_quarter_from_date(filing_dates[i])
                    }

                    if include_document_html:
                        document_html = self.get_full_10q_document(document_url)
                        filing_record["document_fetch_success"] = document_html is not None
                        filing_record["document_html"] = document_html
                        filing_record["document_html_length"] = len(document_html) if document_html else 0

                    ten_q_filings.append(filing_record)

            ten_q_filings.sort(key=lambda filing: filing.get("filing_date", ""))

            result = {
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name,
                "filings": ten_q_filings,
                "success": True
            }

            logger.info(f"Retrieved {len(ten_q_filings)} 10-Q filings for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error fetching 10-Q filings for {ticker}: {e}")
            return {"error": str(e), "filings": [], "success": False}

    def get_financial_facts(self, ticker: str) -> Dict:
        """
        Fetch key financial facts from XBRL data using SEC Company Facts API.
        This gives you structured financial data from the last 4-8 quarters.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with key financial metrics over time
        """
        cik = self.ticker_to_cik(ticker)
        if not cik:
            return {"error": f"Could not find CIK for {ticker}", "success": False}

        try:
            time.sleep(0.5)

            # Company Facts API 
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            logger.info(f"SEC Company Facts API URL being requested: {url}")
            response = requests.get(url, headers=self.sec_headers, timeout=15)
            response.raise_for_status()

            data = response.json()

            company_name = data.get('entityName', self.company_name_cache.get(ticker, 'Unknown'))
            if ticker not in self.company_name_cache:
                self.company_name_cache[ticker] = company_name

            facts = data.get('facts', {}).get('us-gaap', {})

            key_metrics = {
                'Revenue': 'Revenues',
                'NetIncome': 'NetIncomeLoss',
                'GrossProfit': 'GrossProfit',
                'OperatingIncome': 'OperatingIncomeLoss',
                'EarningsPerShare': 'EarningsPerShareBasic',
                'TotalAssets': 'Assets',
                'TotalLiabilities': 'Liabilities',
                'CashAndCashEquivalents': 'CashAndCashEquivalentsAtCarryingValue',
                'OperatingCashFlow': 'NetCashProvidedByUsedInOperatingActivities',
                'ResearchAndDevelopment': 'ResearchAndDevelopmentExpense'
            }

            financial_data = {}
            for display_name, concept in key_metrics.items():
                if concept in facts:
                    units = facts[concept].get('units', {})
                    # Usually USD or USD/shares
                    for unit, values in units.items():
                        # Sort by end date, get last 8
                        sorted_values = sorted(values, key=lambda x: x.get('end', ''), reverse=True)
                        financial_data[display_name] = sorted_values[:8]
                        break  

            # To get revenue growth rate by calculating YoY
            if 'Revenue' in financial_data:
                revenue_data = financial_data['Revenue']
                for i, entry in enumerate(revenue_data):
                    if i < len(revenue_data) - 4:  # Compare with 4 quarters ago
                        current = entry.get('val', 0)
                        previous = revenue_data[i + 4].get('val', 1)
                        if previous != 0:
                            entry['yoy_growth_percent'] = round((current - previous) / previous * 100, 2)

            result = {
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name,
                "financial_metrics": financial_data,
                "success": True
            }

            logger.info(f"Retrieved financial facts for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error fetching financial facts for {ticker}: {e}")
            return {"error": str(e), "success": False}

    def get_full_10q_document(self, document_url: str) -> Optional[str]:
        """
        Fetch the full text of a 10-Q filing document (HTML).

        Args:
            document_url: The URL of the 10-Q document from SEC

        Returns:
            HTML content as string or None if failed
        """
        try:
            time.sleep(0.5)

            response = requests.get(document_url, headers=self.sec_headers, timeout=20)
            response.raise_for_status()

            logger.info(f"Successfully fetched document from {document_url}")
            return response.text

        except Exception as e:
            logger.error(f"Error fetching 10-Q document: {e}")
            return None

    def _get_quarter_from_date(self, date_str: str) -> str:
        """Convert a date string to quarter format (e.g., '2024 Q1')."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year} Q{quarter}"
        except:
            return date_str

    def get_complete_analysis_data(self, ticker: str) -> Dict:
        """
        Convenience method that combines Yahoo Finance data AND SEC filing data.
        This is the main method to call for full analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Complete dictionary with market data and SEC filing data
        """
        # Get Yahoo Finance data
        stock_data = self.get_stock_data(ticker)

        if not stock_data.get('success'):
            return stock_data

        # Get SEC 10-Q filings
        sec_filings = self.get_recent_10q_filings(
            ticker,
            num_quarters=1,
            include_document_html=True
        )

        # Get financial facts from XBRL
        financial_facts = self.get_financial_facts(ticker)

        # Combine all data
        complete_data = {
            **stock_data,
            "sec_filings": sec_filings,
            "financial_facts": financial_facts,
            "full_analysis": True
        }

        return complete_data

    # ============ FORMATTING METHODS ============

    def format_for_next_agent(self, data: Dict) -> str:
        """
        Format the output for consumption by Agent 2.
        Returns a clean, structured summary of Yahoo Finance data.
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

        # Performance data formatting
        start_price_display = f"${historical['start_price']:,.2f}" if historical.get('start_price') is not None else "N/A"
        end_price_display = f"${historical['end_price']:,.2f}" if historical.get('end_price') is not None else "N/A"
        total_return_display = f"{historical['total_return_percent']:.2f}%" if historical.get('total_return_percent') is not None else "N/A"
        volatility_display = f"{historical['volatility']:.2f}%" if historical.get('volatility') is not None else "N/A"

        # Analyst data formatting
        target_mean_price_display = f"${recommendations.get('target_mean_price'):,.2f}" if recommendations.get('target_mean_price') is not None else "N/A"

        target_low_price = recommendations.get('target_low_price')
        target_high_price = recommendations.get('target_high_price')
        target_range_display = f"${target_low_price:,.2f} - ${target_high_price:,.2f}" if target_low_price is not None and target_high_price is not None else "N/A"

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
- Profit Margin: {metrics.get('profit_margins', 'N/A')}
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

    def format_sec_data_for_next_agent(self, sec_data: Dict) -> str:
        """
        Format SEC filing data for consumption by Agent 2.
        """
        if not sec_data.get('success', False):
            return f"SEC Data Error: {sec_data.get('error', 'Unknown error')}"

        formatted = f"""
=== SEC EDGAR FILING DATA ===

Company: {sec_data.get('company_name', 'N/A')} ({sec_data.get('ticker', 'N/A')})
CIK: {sec_data.get('cik', 'N/A')}

RECENT 10-Q FILINGS:
"""
        for filing in sec_data.get('filings', []):
            html_length = filing.get('document_html_length')
            html_length_line = f"  HTML Length: {html_length:,} characters\n" if html_length is not None else ""
            formatted += f"""
- {filing.get('quarter', 'N/A')} (Filed: {filing.get('filing_date', 'N/A')})
  Document URL: {filing.get('document_url', 'N/A')}
{html_length_line}  Document Fetched: {filing.get('document_fetch_success', False)}
"""

        return formatted.strip()

    def format_financial_facts_for_next_agent(self, facts_data: Dict) -> str:
        """
        Format financial facts for consumption by Agent 2.
        """
        if not facts_data.get('success', False):
            return f"Financial Facts Error: {facts_data.get('error', 'Unknown error')}"

        formatted = f"""
=== XBRL FINANCIAL DATA ===
Company: {facts_data.get('company_name', 'N/A')}

KEY METRICS (Last 4 Quarters):
"""

        for metric_name, values in facts_data.get('financial_metrics', {}).items():
            formatted += f"\n{metric_name}:\n"
            for i, val in enumerate(values[:4]):
                date = val.get('end', 'N/A')[:7]
                amount = val.get('val', 0)
                unit = val.get('unit', 'USD')
                yoy = val.get('yoy_growth_percent', '')

                formatted += f"  {date}: ${amount:,.0f} {unit}"
                if yoy:
                    formatted += f" (YoY: {yoy}%)"
                formatted += "\n"

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

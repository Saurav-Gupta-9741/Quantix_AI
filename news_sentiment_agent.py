import yfinance as yf
from bs4 import BeautifulSoup
import requests
import json
import os

class NewsSentimentAgent:
    def __init__(self, finbert_analyzer=None):
        self.agent_name = "Quantix News Architect"
        self.finbert = finbert_analyzer  # Real FinBERT pipeline passed from main_api

    def analyze_fundamentals(self, ticker):
        """Scrapes real news and uses FinBERT transformer for genuine sentiment analysis."""
        try:
            import urllib.parse
            import xml.etree.ElementTree as ET
            
            safe_query = urllib.parse.quote_plus(ticker + " stock")
            url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-US&gl=US&ceid=US:en"
            
            res = requests.get(url, timeout=5)
            root = ET.fromstring(res.text)
            
            headlines = []
            for item in root.findall('.//item')[:8]:
                title_node = item.find('title')
                if title_node is not None and title_node.text:
                    # Clean up publisher names (e.g., '... - Bloomberg')
                    clean_title = title_node.text.split(' - ')[0]
                    headlines.append(clean_title)
            
            if not headlines:
                return "NEUTRAL", "[NEWS AGENT]: No Google News found. Signal based purely on technical analysis."
            
            # Use REAL FinBERT transformer if available
            if self.finbert:
                try:
                    results = self.finbert(headlines)
                    bull_score = 0.0
                    bear_score = 0.0
                    neutral_score = 0.0
                    
                    for result in results:
                        label = result['label']
                        score = result['score']
                        if label == 'positive':
                            bull_score += score
                        elif label == 'negative':
                            bear_score += score
                        else:
                            neutral_score += score
                    
                    total = bull_score + bear_score + neutral_score
                    if total == 0:
                        sentiment = "NEUTRAL"
                    elif bull_score > bear_score and bull_score > neutral_score:
                        sentiment = "BULLISH"
                    elif bear_score > bull_score and bear_score > neutral_score:
                        sentiment = "BEARISH"
                    else:
                        sentiment = "NEUTRAL"
                    
                    confidence = max(bull_score, bear_score, neutral_score) / total * 100 if total > 0 else 0
                    
                    headline_summary = ' | '.join(headlines[:3])
                    summary = (
                        f"[NEWS AGENT]: FinBERT analyzed {len(headlines)} real-time headlines. "
                        f"Sentiment: {sentiment} ({confidence:.0f}% confidence). "
                        f"Bull: {bull_score:.2f} | Bear: {bear_score:.2f} | Neutral: {neutral_score:.2f}. "
                        f"Headlines: {headline_summary}"
                    )
                    return sentiment, summary
                    
                except Exception as e:
                    print(f"[-] FinBERT Error: {e}")
            
            # Fallback: keyword-based (still uses real headlines, not random)
            bullish_keywords = ['surge', 'growth', 'up', 'beat', 'profit', 'dividend', 'buy', 'higher', 'record', 'gain', 'rally', 'soar', 'bull']
            bearish_keywords = ['fall', 'drop', 'miss', 'loss', 'down', 'sell', 'lower', 'crash', 'risk', 'cut', 'bear', 'decline', 'plunge']
            
            bull_score = 0
            bear_score = 0
            
            for h in headlines:
                lower_h = h.lower()
                for bk in bullish_keywords:
                    if bk in lower_h: bull_score += 1
                for rk in bearish_keywords:
                    if rk in lower_h: bear_score += 1
            
            if bull_score > bear_score + 1:
                sentiment = "BULLISH"
            elif bear_score > bull_score + 1:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            headline_summary = ' | '.join(headlines[:3])
            summary = (
                f"[NEWS AGENT]: Analyzed {len(headlines)} real headlines (keyword mode). "
                f"Sentiment: {sentiment}. Bull signals: {bull_score}, Bear signals: {bear_score}. "
                f"Headlines: {headline_summary}"
            )
            return sentiment, summary
            
        except Exception as e:
            return "NEUTRAL", f"[RAG AGENT ERROR]: {str(e)}"

if __name__ == "__main__":
    agent = NewsSentimentAgent()
    sentiment, summary = agent.analyze_fundamentals("AAPL")
    print(f"Sentiment: {sentiment}")
    print(f"Summary: {summary}")

from transformers import pipeline

class SentimentTransformer:
    def __init__(self):
        print("[*] Loading FinBERT Transformer Model...")
        # FinBERT is specifically pre-trained on financial text
        self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze_headlines(self, headlines):
        """
        Takes a list of news headlines and returns a Fear/Greed index score.
        """
        results = self.analyzer(headlines)
        
        bullish_score = 0
        bearish_score = 0
        
        for idx, result in enumerate(results):
            label = result['label']
            score = result['score']
            
            print(f"Headline: '{headlines[idx]}'")
            print(f"-> Sentiment: {label.upper()} (Confidence: {score:.2f})\n")
            
            if label == 'positive':
                bullish_score += score
            elif label == 'negative':
                bearish_score += score
                
        # Calculate a simple Fear vs Greed ratio (0 to 1)
        total = bullish_score + bearish_score
        if total == 0:
            return 0.5 # Neutral
            
        greed_index = bullish_score / total
        return greed_index

if __name__ == "__main__":
    print("--- PHASE 4: GLOBAL SENTIMENT ENGINE ---")
    
    bot = SentimentTransformer()
    
    sample_news = [
        "RBI expected to cut interest rates next quarter.",
        "Major semiconductor shortage hits global tech stocks.",
        "Unemployment rises, sparking fears of a recession."
    ]
    
    print("\n[*] Analyzing live data streams...")
    greed_index = bot.analyze_headlines(sample_news)
    
    print(f"[!] Current Market Greed Index: {greed_index:.2f} (0=Extreme Fear, 1=Extreme Greed)")

import pandas as pd
import mplfinance as mpf
import os

def generate_candlestick_images(csv_file, output_folder, window_size=30):
    """
    Reads a CSV of stock data and generates 2D candlestick images
    in rolling windows. These images will be the input for our CNN.
    """
    print(f"[*] Reading real market data from {csv_file}...")
    df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"[*] Generating {window_size}-day window candlestick images. This may take a moment...")
    
    # We will generate the full mathematical dataset
    sample_size = len(df) - window_size
    
    for i in range(sample_size):
        # Extract a window of data
        window_data = df.iloc[i : i + window_size]
        
        # The filename will contain the date of the last day in the window
        last_date = window_data.index[-1].strftime('%Y-%m-%d')
        image_path = os.path.join(output_folder, f"chart_{last_date}.png")
        
        # We strip off all human elements (axis, gridlines, labels) 
        # so the CNN only learns the pure mathematical shape of the candlesticks.
        mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='', figcolor='black', facecolor='black')
        
        # Save the image silently
        mpf.plot(window_data, type='candle', style=s, axisoff=True, 
                 savefig=dict(fname=image_path, dpi=100, bbox_inches='tight', pad_inches=0))
                 
    print(f"[+] Successfully generated {sample_size} pure mathematical candlestick images in '{output_folder}'.\n")

if __name__ == "__main__":
    print("--- PHASE 2: VISION CNN PRE-PROCESSING ---")
    print("STRICT MODE: 100% REAL MARKET DATA (NO SYNTHETICS)")
    
    # Generate images for NIFTY50
    generate_candlestick_images(
        csv_file="nifty50_processed_data.csv", 
        output_folder="dataset_nifty_images"
    )
    
    # Generate images for Bitcoin
    generate_candlestick_images(
        csv_file="bitcoin_processed_data.csv", 
        output_folder="dataset_btc_images"
    )
    
    print("--- PRE-PROCESSING COMPLETE ---")

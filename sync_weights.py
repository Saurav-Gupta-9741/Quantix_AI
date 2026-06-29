import os
import sys
import time

try:
    import gdown
except ImportError:
    print("[!] ERROR: 'gdown' library is missing.")
    print("[*] Please run: pip install gdown")
    sys.exit(1)

def sync_cloud_weights():
    print("==================================================")
    print("   QUANTIX AI - CLOUD WEIGHTS SYNCHRONIZATION     ")
    print("==================================================")
    
    # ---------------------------------------------------------
    # STEP 1: GET YOUR FILE IDs
    # ---------------------------------------------------------
    # Go to Google Drive -> Right Click your .pth file -> Get Link
    # The link looks like: https://drive.google.com/file/d/1XyZ_abcdefg123456/view?usp=sharing
    # Your FILE ID is the random string of characters: 1XyZ_abcdefg123456
    
    # Paste your Google Drive File IDs here:
    LSTM_FILE_ID = "YOUR_LSTM_FILE_ID_HERE"
    CNN_FILE_ID = "YOUR_CNN_FILE_ID_HERE"
    
    if LSTM_FILE_ID == "YOUR_LSTM_FILE_ID_HERE" or CNN_FILE_ID == "YOUR_CNN_FILE_ID_HERE":
        print("[-] ERROR: You must edit sync_weights.py and paste your Google Drive File IDs!")
        sys.exit(1)
        
    print(f"[*] Contacting Google Cloud Storage...")
    
    # Target download paths (directly into your backend folder)
    base_dir = os.path.dirname(__file__)
    lstm_output = os.path.join(base_dir, "quantix_btc_lstm.pth")
    cnn_output = os.path.join(base_dir, "quantix_cnn_v1.pth")
    
    # Download LSTM Weights
    print("\n[+] Downloading latest Bidirectional LSTM weights...")
    try:
        gdown.download(id=LSTM_FILE_ID, output=lstm_output, quiet=False)
    except Exception as e:
        print(f"[-] Failed to download LSTM: {e}")
        
    # Download CNN Weights
    print("\n[+] Downloading latest ResNet18 CNN weights...")
    try:
        gdown.download(id=CNN_FILE_ID, output=cnn_output, quiet=False)
    except Exception as e:
        print(f"[-] Failed to download CNN: {e}")
        
    print("\n==================================================")
    print("[!] Synchronization Complete!")
    print("[!] Restart your FastAPI server to load the new brains.")
    print("==================================================")

if __name__ == "__main__":
    sync_cloud_weights()

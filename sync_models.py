import os
import sys

def sync_from_drive():
    print("--- QUANTIX AI: MODEL SYNCHRONIZATION UTILITY ---")
    print("This utility requires 'gdown' to download files from Google Drive.")
    try:
        import gdown
    except ImportError:
        print("[!] 'gdown' is not installed. Please run: pip install gdown")
        sys.exit(1)
        
    print("\nTo automate this, you need to generate 'Anyone with the link can view' links for your .pth files in Google Drive.")
    print("Example Link: https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view?usp=sharing\n")
    
    lstm_link = input("Enter the Google Drive Share Link for quantix_btc_lstm.pth (or press Enter to skip): ").strip()
    cnn_link = input("Enter the Google Drive Share Link for quantix_cnn_v1.pth (or press Enter to skip): ").strip()
    
    def download_file(url, output_path):
        if not url: return
        print(f"[*] Downloading to {output_path}...")
        try:
            # gdown handles google drive sharing links automatically
            gdown.download(url, output_path, quiet=False)
            print(f"[+] Successfully synced: {output_path}")
        except Exception as e:
            print(f"[-] Failed to download from {url}: {e}")

    if lstm_link:
        download_file(lstm_link, "quantix_btc_lstm.pth")
    if cnn_link:
        download_file(cnn_link, "quantix_cnn_v1.pth")
        
    print("\n[+] Synchronization Complete! Please restart the API Server (uvicorn) to load the new weights into memory.")

if __name__ == "__main__":
    sync_from_drive()

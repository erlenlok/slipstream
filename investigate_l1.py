import json
import requests
import time
from websocket import create_connection # pip install websocket-client

def get_l1_height_via_http_list():
    """
    Attempts to get L1 height via 'blockList' explorer endpoint.
    """
    targets = [
        "https://api.hyperliquid.xyz/info",
        "https://api.hyperliquid.xyz/explorer",
    ]
    
    headers = {"Content-Type": "application/json"}
    
    print(f"1. Probing 'blockList' to find L1 Tip...")
    
    for url in targets:
        # Try blockList
        payload = {"type": "blockList"}
        try:
            print(f"   Requesting {url} with {payload}...")
            resp = requests.post(url, json=payload, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                # Expected: list of blocks? or {"data": [...]}?
                # If list
                if isinstance(data, list) and len(data) > 0:
                    # check first element
                    h = data[0].get('height')
                    if h:
                         print(f"   -> FOUND L1 HEIGHT (via blockList): {h}")
                         return h
                # If dict
                elif isinstance(data, dict):
                     # check keys
                     print(f"   Keys: {data.keys()}")
                     # maybe 'data' -> list ?
            else:
                print(f"   Failed: {resp.status_code}")
        except Exception as e:
            print(f"   Error: {e}")
            
    return None

def fetch_block_details_http(height):
    """
    Fetches the deep block data (TIF/CLOID) via HTTP using the correct height.
    """
    url = "https://api.hyperliquid.xyz/info"
    # We query height - 10 to ensure it's fully indexed/finalized
    target_height = height - 10
    
    print(f"\n2. Fetching Block {target_height} via HTTP POST...")
    
    payload = {
        "type": "blockDetails", 
        "height": target_height
    }
    
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"HTTP Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"HTTP Exception: {e}")
    return None

def find_alpha_data(block_data):
    """
    Hunts for the specific fields needed for the Alphas.
    """
    print("\n3. Analyzing Payload for Alpha 3 (TIF/CLOID)...")
    
    # Path: data -> block -> signed_action_bundles
    # Note: Structure can be nested differently depending on update
    # We will search recursively for the 't' (type) and 'c' (cloid) keys.
    
    found_ioc = 0
    found_cloid = 0
    
    # Simple recursive search for demonstration
    stack = [block_data]
    while stack:
        item = stack.pop()
        
        if isinstance(item, dict):
            # Check for Order Definition
            if "t" in item and "limit" in item["t"]:
                tif = item["t"]["limit"].get("tif")
                if tif:
                    if tif == "Ioc": found_ioc += 1
            
            # Check for Client Order ID
            if "c" in item:
                cloid = item["c"]
                if isinstance(cloid, str) and cloid.startswith("0x"):
                    print(f"   [Proof] Found CLOID: {cloid}")
                    found_cloid += 1
            
            for k, v in item.items():
                if isinstance(v, (dict, list)): stack.append(v)
                
        elif isinstance(item, list):
            for i in item:
                if isinstance(i, (dict, list)): stack.append(i)

    print("-" * 40)
    print(f"RESULTS for Block:")
    print(f"Valid CLOIDs found: {found_cloid}")
    print(f"IOC Orders found:   {found_ioc}")
    
    if found_cloid > 0:
        print("\n✅ SUCCESS: The data required for Alpha 3 is accessible.")
    else:
        print("\n⚠️ Note: No CLOIDs in this specific block. Try running again (blocks vary).")

if __name__ == "__main__":
    l1_height = get_l1_height_via_http_list()
    
    if l1_height:
        data = fetch_block_details_http(l1_height)
        if data:
            find_alpha_data(data)
    else:
        print("Could not find L1 Height. API appears locked down.")

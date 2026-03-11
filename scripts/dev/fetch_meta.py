import requests
import json

url = "https://api.hyperliquid.xyz/info"
payload = {"type": "meta"}
resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
data = resp.json()

universe = data["universe"]
# universe is list of dicts: {"name": "BTC", "szDecimals": 5, "maxLeverage": 50, "onlyIsolated": false}

print(f"Total Universe: {len(universe)} symbols")
candidates = ["WIF", "BONK", "PEPE", "DOGE", "SHIB", "POPCAT", "MEW", "MOG", "FLOKI", "BRETT"]
found = []

for asset in universe:
    name = asset["name"]
    # Check if name contains any candidate substing
    for c in candidates:
        if c in name:
            print(f"MATCH: {name} (szDecimals={asset['szDecimals']})")
            found.append(name)
            
# Also print any starting with 'k' just in case
print("\n--- Fractional Assets (k...) ---")
for asset in universe:
    if asset["name"].startswith("k") or asset["name"].startswith("1000"):
        print(f"FRACTIONAL: {asset['name']} (szDecimals={asset['szDecimals']})")

print("\n--- Inferring Tick Sizes ---")
for coin in ["WIF", "kPEPE", "kBONK"]:
    payload = {"type": "l2Book", "coin": coin}
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    book = resp.json()
    if "levels" in book:
        # book["levels"] is [[px, sz, num], ...] usually for bids/asks
        # Actually structure is {"levels": [[...], ...]}?
        # Check specific structure
        bids = book.get("levels", []) # Hyperliquid might allow diff structure
        if bids and len(bids) > 1:
            # bids[0] is best bid?
            # levels might be [ [px, sz], ...]
            # Let's print first two levels
             pass
    # Hyperliquid response: {"coin": "WIF", "levels": [[px, sz, oid?], ...], "time": ...}
    # Actually levels are split into [levels] usually implies bids/asks combined?
    # No, usually "levels" contains [ [px, sz, n], ... ] for ONE side?
    # Wait, type="l2Book" returns { "levels": [ ... ] } in old API?
    # New API: { "levels": [ [px, sz, n], ... ] } represents one side?
    # It sends both sides.
    # Actually let's assume standard response structure
    print(f"{coin} Book: {book.keys()} levels[0]: {book['levels'][0] if 'levels' in book and book['levels'] else 'empty'}")


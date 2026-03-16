"""
wifi_test_phase2.py — Phase 2: WiFi Connection Test
Runs on: ESP32S3 (XIAO ESP32S3 Sense)

Purpose:
    Connect to a WiFi network using the provided Wifi.py library,
    print the assigned IP address, and confirm connectivity.

Adapted from provided working example: Wifi.py
    Uses the Sta class exactly as provided (Sharil Tumin, MIT License).
Changes from original:
    1. Created a test wrapper around the Sta class
    2. Added retry logic with countdown
    3. Added structured debug output
    4. Added network scan to help diagnose issues
    The Wifi.py file itself is NOT modified.

Citation: Wifi.py by Sharil Tumin, MIT License.
"""

import gc
from time import sleep

gc.collect()
print(f"[INFO] Free memory: {gc.mem_free()} bytes")

# ─── Step 1: Configure your WiFi credentials ───
# CHANGE THESE to match your network
SSID = "V’s iPhone"       # ← your WiFi network name
PASSWORD = "temppwdesp" # ← your WiFi password

# ─── Step 2: Import the provided Wifi library ───
# Wifi.py must be uploaded to the ESP filesystem via Thonny
try:
    from Wifi import Sta
    print("[OK]   Wifi.py library imported")
except ImportError:
    print("[FAIL] Could not import Wifi.py")
    print("[TIP]  Upload Wifi.py to the ESP using Thonny:")
    print("       View → Files → right-click Wifi.py → Upload to /")
    raise

# ─── Step 3: Scan for available networks ───
# This helps confirm the WiFi radio works and your network is visible
print("\n[INFO] Scanning for WiFi networks...")
try:
    wif = Sta(SSID, PASSWORD)
    networks = wif.scan()
    print(f"[OK]   Found {len(networks)} networks:")
    for net in networks:
        # net is a tuple: (ssid, bssid, channel, RSSI, authmode, hidden)
        ssid_name = net[0].decode("utf-8") if isinstance(net[0], bytes) else net[0]
        rssi = net[3]
        print(f"       {ssid_name:30s}  signal: {rssi} dBm")
    print()

    # Check if our target network was found
    found = any(
        (net[0].decode("utf-8") if isinstance(net[0], bytes) else net[0]) == SSID
        for net in networks
    )
    if found:
        print(f"[OK]   Target network '{SSID}' found in scan")
    else:
        print(f"[WARN] Target network '{SSID}' NOT found in scan")
        print("[TIP]  Check SSID spelling (case-sensitive)")
        print("[TIP]  Make sure your hotspot/router is on and in range")
        print("[TIP]  5 GHz networks may not be visible — use 2.4 GHz")

except Exception as e:
    print(f"[WARN] Scan failed: {e}")
    print("[TIP]  This is non-fatal — will still try to connect")

# ─── Step 4: Connect to WiFi ───
print(f"\n[INFO] Connecting to '{SSID}'...")
MAX_RETRIES = 3
connected = False

for attempt in range(1, MAX_RETRIES + 1):
    print(f"[INFO] Attempt {attempt}/{MAX_RETRIES}")

    wif.connect()

    # Wait up to 15 seconds for connection
    for countdown in range(15, 0, -1):
        status = wif.status()
        if status:  # non-empty tuple means connected
            connected = True
            break
        print(f"       Waiting... {countdown}s")
        sleep(1)

    if connected:
        break
    else:
        print(f"[WARN] Attempt {attempt} failed")
        if attempt < MAX_RETRIES:
            print("[INFO] Retrying in 3 seconds...")
            sleep(3)

# ─── Step 5: Report results ───
print(f"\n{'='*40}")
print("PHASE 2 WIFI TEST SUMMARY")
print(f"{'='*40}")

if connected:
    ip, subnet, gateway, dns = wif.status()
    print(f"  Status:    CONNECTED")
    print(f"  SSID:      {SSID}")
    print(f"  IP:        {ip}")
    print(f"  Subnet:    {subnet}")
    print(f"  Gateway:   {gateway}")
    print(f"  DNS:       {dns}")
    print(f"  Free mem:  {gc.mem_free()} bytes")
    print(f"{'='*40}")
    print(f"\n[OK]   WiFi is working!")
    print(f"[INFO] Your ESP IP address is: {ip}")
    print(f"[INFO] Make sure your laptop is on the same network ('{SSID}')")
    print(f"[INFO] You can verify by pinging {ip} from your Mac terminal:")
    print(f"       ping {ip}")
else:
    print(f"  Status:    FAILED")
    print(f"  SSID:      {SSID}")
    print(f"{'='*40}")
    print("\n[FAIL] Could not connect to WiFi after all retries")
    print("\nTroubleshooting:")
    print("  1. Double-check SSID and PASSWORD at the top of this file")
    print("  2. SSID is case-sensitive")
    print("  3. Use a phone hotspot (simpler than campus WiFi)")
    print("  4. ESP32S3 only supports 2.4 GHz, not 5 GHz")
    print("  5. Campus WiFi with login portals (WPA-Enterprise) won't work")
    print("  6. Try moving closer to the router/hotspot")

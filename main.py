import warnings
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import os

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1491805303975448696/vQf8856EYYwmAaBUYIB5wdygG60KnFuGA5YbqqejaX9--4sPjimKYohptvd2hQHwLILv")

# ==========================================
# 1. DATA FETCHING & ATR CALCULATION
# ==========================================
def fetch_live_data(
    leader: str = "^TNX", follower: str = "BTC-USD", lookback_days: int = 200
) -> pd.DataFrame:
    raw = yf.download(
        [leader, follower], period=f"{lookback_days}d", interval="1d", auto_adjust=True
    )

    df = pd.DataFrame()
    df["Leader_Close"] = raw["Close"][leader].squeeze()
    df["Follower_Close"] = raw["Close"][follower].squeeze()
    df["Follower_High"] = raw["High"][follower].squeeze()
    df["Follower_Low"] = raw["Low"][follower].squeeze()

    df = df.ffill().dropna()

    df["Leader_Return"] = df["Leader_Close"].pct_change()
    df["Follower_Return"] = df["Follower_Close"].pct_change()

    # คำนวณ ATR 14
    high = df["Follower_High"]
    low = df["Follower_Low"]
    close = df["Follower_Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    return df

# ==========================================
# 2. DISCORD NOTIFICATION FUNCTION
# ==========================================
def send_discord_webhook(webhook_url: str, data: dict):
    """ส่งข้อมูล Signal เข้า Discord Webhook ในรูปแบบ Embed"""
    # แก้ไขเงื่อนไขเช็ค URL ให้ถูกต้อง
    if not webhook_url or webhook_url == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("[Warning] Discord Webhook URL is not set. Skipping Discord alert.")
        return

    # กำหนดสี Embed ตาม Signal
    color = 0x808080  # สีเทา (HOLD)
    if data["final_signal"] == 1:
        color = 0x2ECC71  # สีเขียว (LONG)
    elif data["final_signal"] == -1:
        color = 0xE74C3C  # สีแดง (SHORT)

    embed = {
        "title": "🤖 AUTOMATIC DAILY TRADE SIGNAL",
        "description": f"**Date Evaluated:** {data['latest_date']}\n**Follower:** BTC-USD (${data['latest_close']:,.2f})\n**Leader:** ^TNX (US 10Y Yield)",
        "color": color,
        "fields": [
            {
                "name": "📊 Analysis Details",
                "value": f"• **Best Lag Selected:** {data['best_lag']} Day(s) (Corr: {data['best_corr']:.3f})\n"
                         f"• **EMA 50 Filter:** ${data['latest_ema']:,.2f} ({'BULLISH' if data['ema_signal'] == 1 else 'BEARISH'})\n"
                         f"• **Current ATR (14):** ${data['latest_atr']:,.2f}",
                "inline": False
            },
            {
                "name": f"🎯 TODAY'S ACTION: {data['action']}",
                "value": (
                    f"• **Entry Price:** ${data['entry_price']:,.2f}\n"
                    f"• **Stop Loss (SL):** ${data['sl_price']:,.2f} (-{data['sl_pct']:.2f}%)\n"
                    f"• **Take Profit (TP):** ${data['tp_price']:,.2f} (+{data['tp_pct']:.2f}%)"
                ) if data["final_signal"] != 0 else "• **Reason:** Signals conflict or correlation below threshold (< 0.2)",
                "inline": False
            }
        ],
        "footer": {"text": "Lead-Lag Quant Strategy Alert"}
    }

    payload = {"embeds": [embed]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        if response.status_code in [200, 204]:
            print("Successfully sent signal alert to Discord!")
        else:
            print(f"Failed to send to Discord. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending Discord webhook: {e}")

# ==========================================
# 3. AUTOMATIC SIGNAL CALCULATION
# ==========================================
def calculate_today_signal(
    df: pd.DataFrame,
    max_lag: int = 5,
    window: int = 60,
    min_corr: float = 0.2,
    ema_period: int = 50,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 4.0,
):
    df = df.copy()

    leader_ret = df["Leader_Return"]
    follower_ret = df["Follower_Return"]

    lag_corrs = {}
    for lag in range(1, max_lag + 1):
        lag_corrs[lag] = (
            leader_ret.shift(lag).rolling(window=window).corr(follower_ret)
        )

    corr_df = pd.DataFrame(lag_corrs)
    ema = df["Follower_Close"].ewm(span=ema_period, adjust=False).mean()

    latest_idx = -1
    latest_date = df.index[latest_idx].strftime("%Y-%m-%d")
    latest_close = df["Follower_Close"].iloc[latest_idx]
    latest_ema = ema.iloc[latest_idx]
    latest_atr = df["ATR"].iloc[latest_idx]

    latest_corrs = corr_df.iloc[latest_idx]
    best_lag = latest_corrs.abs().idxmax()
    best_corr = latest_corrs[best_lag]

    lead_lag_signal = 0
    if abs(best_corr) >= min_corr:
        leader_past_ret = leader_ret.iloc[latest_idx - best_lag]
        if best_corr > 0:
            lead_lag_signal = 1 if leader_past_ret > 0 else -1
        else:
            lead_lag_signal = -1 if leader_past_ret > 0 else 1

    ema_signal = 1 if latest_close > latest_ema else -1

    final_signal = (
        lead_lag_signal if lead_lag_signal == ema_signal else 0
    )

    entry_price = latest_close
    sl_dist = sl_atr_mult * latest_atr
    tp_dist = tp_atr_mult * latest_atr

    sl_pct = 0.0
    tp_pct = 0.0

    if final_signal == 1:
        action = "LONG / BUY"
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
        sl_pct = (sl_dist / entry_price) * 100
        tp_pct = (tp_dist / entry_price) * 100
    elif final_signal == -1:
        action = "SHORT / SELL"
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
        sl_pct = (sl_dist / entry_price) * 100
        tp_pct = (tp_dist / entry_price) * 100
    else:
        action = "NO TRADE / HOLD CASH"
        sl_price = 0.0
        tp_price = 0.0

    # รวบรวมข้อมูลเพื่อส่งออก
    signal_data = {
        "latest_date": latest_date,
        "latest_close": latest_close,
        "best_lag": best_lag,
        "best_corr": best_corr,
        "latest_ema": latest_ema,
        "ema_signal": ema_signal,
        "latest_atr": latest_atr,
        "final_signal": final_signal,
        "action": action,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct
    }

    # ส่งเข้า Discord
    send_discord_webhook(DISCORD_WEBHOOK_URL, signal_data)


if __name__ == "__main__":
    df = fetch_live_data(leader="^TNX", follower="BTC-USD")
    calculate_today_signal(
        df,
        max_lag=5,
        window=60,
        min_corr=0.2,
        ema_period=50,
        sl_atr_mult=1.5,
        tp_atr_mult=4.0,
    )
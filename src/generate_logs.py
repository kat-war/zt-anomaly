import pandas as pd
import random
from datetime import datetime, timedelta

NUM_USERS = 50
NUM_DEVICES = 100
DAYS = 7
EVENTS_PER_DAY = 5000
OUTPUT_PATH = "data/raw/zt_logs.csv"

users = [f"user{i:03}" for i in range(1, NUM_USERS + 1)]
roles = ["employee", "contractor", "admin"]
resources = ["vpn", "email", "finance_app", "dev_repo", "hr_portal"]
countries = ["US", "IN", "SG", "UK", "DE"]
devices = [f"device{i:03}" for i in range(1, NUM_DEVICES + 1)]

def random_geo(country):
    geo = {
        "US": (37.77, -122.41),
        "IN": (19.07, 72.87),
        "SG": (1.35, 103.82),
        "UK": (51.50, -0.12),
        "DE": (52.52, 13.40),
    }
    return geo[country]

def generate_events():
    events = []
    start = datetime.now() - timedelta(days=DAYS)
    for day in range(DAYS):
        for _ in range(EVENTS_PER_DAY):
            user = random.choice(users)
            role = random.choices(roles, weights=[0.7,0.2,0.1])[0]
            device = random.choice(devices)
            country = random.choice(countries)
            lat, lon = random_geo(country)
            resource = random.choice(resources)
            action = random.choice(["login","access","logout"])
            ts = start + timedelta(days=day, seconds=random.randint(0,86400))
            device_trust = random.choices(["trusted","untrusted"], weights=[0.9,0.1])[0]
            result = random.choices(["success","failure"], weights=[0.95,0.05])[0]
            mfa = random.choices(["true","false"], weights=[0.8,0.2])[0]

            # 2% anomalies
            anomaly = random.random() < 0.02
            if anomaly:
                ts = ts.replace(hour=random.choice([2,3,4]))
                device_trust = "untrusted"
                mfa = "false"

            events.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "user": user,
                "role": role,
                "device_id": device,
                "device_trust": device_trust,
                "country": country,
                "lat": lat,
                "lon": lon,
                "resource": resource,
                "action": action,
                "result": result,
                "mfa": mfa,
                "is_anomaly": int(anomaly),
            })
    return pd.DataFrame(events)

if __name__ == "__main__":
    df = generate_events()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[+] Generated {len(df)} events -> {OUTPUT_PATH}")
    print(df.head())

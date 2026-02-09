import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = {
    "order_volume": np.random.randint(1, 10, rows),
    "warehouse_time": np.random.uniform(1, 24, rows),
    "shipment_distance": np.random.uniform(10, 2000, rows),
    "traffic_index": np.random.randint(1, 11, rows),
    "weather_score": np.round(np.random.uniform(0, 1, rows), 2),
    "past_delay_rate": np.round(np.random.uniform(0, 1, rows), 2)
}

df = pd.DataFrame(data)

# Create delivery_status using business logic
conditions = [
    (df["warehouse_time"] <= 8) &
    (df["traffic_index"] <= 4) &
    (df["weather_score"] <= 0.3),

    (df["warehouse_time"] <= 16) &
    (df["traffic_index"] <= 7)
]

choices = ["On-Time", "At Risk"]

df["delivery_status"] = np.select(
    conditions,
    choices,
    default="Delayed"
)

# Save dataset
df.to_csv("amazon_delivery_data.csv", index=False)

print("✅ Artificial dataset created successfully!")
print(df.head())

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random


def generate_zone_data(zone_id="Z-101"):
    current = random.randint(0, 40)
    previous = random.randint(0, 40)

    return {
        "zoneId": zone_id,
        "metricsLast24h": {
            "totalDetections": current,
            "avgConfidence": round(random.uniform(0.6, 0.95), 2)
        },
        "metricsPrev24h": {
            "totalDetections": previous
        },
        "environment": {
            "humidity": random.randint(40, 90),
            "crowdIndex": round(random.uniform(0.2, 0.9), 2)
        }
    }

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import predict_risk
from src.simulator import generate_zone_data


data = generate_zone_data()
result = predict_risk(data)

print("INPUT:", data)
print("OUTPUT:", result)

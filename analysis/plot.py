import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đọc dữ liệu từ file JSON
with open('poses.json', 'r') as f:
    data = json.load(f)

# Tách riêng các giá trị pitch, yaw, roll
pitch_values = [pose[0] for pose in data]
yaw_values = [pose[1] for pose in data]
roll_values = [pose[2] for pose in data]

# Hàm vẽ barplot với khoảng giá trị là 1 đơn vị
def plot_distribution(values, title):
    plt.figure(figsize=(10, 6))
    # Tạo các bin với khoảng cách 1 đơn vị, từ giá trị nhỏ nhất đến lớn nhất
    bins = np.arange(int(min(values)), int(max(values)) + 2)  # +2 để bao gồm giá trị lớn nhất
    sns.histplot(values, bins=bins, kde=False)
    plt.title(f"Distribution of {title}")
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.show()

# Vẽ barplot cho từng giá trị
plot_distribution(pitch_values, "Pitch")
plot_distribution(yaw_values, "Yaw")
plot_distribution(roll_values, "Roll")

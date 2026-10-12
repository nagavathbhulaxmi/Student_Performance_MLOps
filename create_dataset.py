import pandas as pd
from pathlib import Path

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 9, 3, 5, 7],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 78, 95, 62, 72, 85],
    "previous_marks": [35, 40, 45, 50, 55, 60, 65, 70, 42, 52, 62, 80, 48, 58, 68],
    "final_marks": [30, 38, 45, 52, 60, 68, 75, 90, 40, 55, 70, 95, 48, 63, 80],
}

df = pd.DataFrame(data)

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

output_file = data_dir / "student_data.csv"
df.to_csv(output_file, index=False)

print(f"Dataset created successfully: {output_file}")
print(df.head())
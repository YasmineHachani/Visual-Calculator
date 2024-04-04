import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('all_hands_images.csv')

# Split the data into train, validation, and test sets
train_data, val_data = train_test_split(data, test_size=0.15, random_state=14, stratify=data['label'])

# Save the divided data into separate CSV files
train_data.to_csv('train/hands_images.csv', index=False)
val_data.to_csv('val/hands_images.csv', index=False)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic dataset
def create_synthetic_data():
    np.random.seed(0)
    heights = np.random.normal(170, 10, 500)
    weights = np.random.normal(70, 15, 500)
    data = pd.DataFrame({'Height': heights, 'Weight': weights})
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    bins = [0, 18.5, 24.9, 29.9, np.inf]
    labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    data['Category'] = pd.cut(data['BMI'], bins=bins, labels=labels)
    return data

# Load Dataset
df = create_synthetic_data()

# Data Preprocessing
X = df[['Height', 'Weight']]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# GUI
def predict_bmi():
    try:
        # Get user input
        height = float(entry_height.get())
        weight = float(entry_weight.get())
        
        # Calculate BMI
        bmi = weight / (height / 100) ** 2
        
        # Predict category
        user_input = np.array([[height, weight]])
        category = knn.predict(user_input)[0]
        
        # Show result
        result = f"BMI: {bmi:.2f}\nCategory: {category}"
        messagebox.showinfo("BMI Prediction", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values.")

# Create the main window
root = tk.Tk()
root.title("BMI Prediction")

# Create labels and entries for height and weight
tk.Label(root, text="Height (cm)").grid(row=0, column=0)
entry_height = tk.Entry(root)
entry_height.grid(row=0, column=1)

tk.Label(root, text="Weight (kg)").grid(row=1, column=0)
entry_weight = tk.Entry(root)
entry_weight.grid(row=1, column=1)

# Create Predict button
tk.Button(root, text="Predict BMI", command=predict_bmi).grid(row=2, columnspan=2)

# Start the GUI event loop
root.mainloop()

# Validation (optional for the assignment)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Visualization (optional for the assignment)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

report = classification_report(y_test, y_pred)
print(report)

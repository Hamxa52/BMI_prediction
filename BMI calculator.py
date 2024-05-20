import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic dataset
def create_synthetic_data():
    np.random.seed(0)
    heights = np.random.normal(170, 10, 500)
    weights = np.random.normal(70, 15, 500)
    genders = np.random.choice(['Male', 'Female'], size=500)
    data = pd.DataFrame({'Height': heights, 'Weight': weights, 'Gender': genders})
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    bins_male = [0, 18.5, 24.9, 29.9, np.inf]
    labels_male = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    bins_female = [0, 18.4, 24.9, 29.9, np.inf]
    labels_female = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    data['Category'] = np.where(data['Gender'] == 'Male', pd.cut(data['BMI'], bins=bins_male, labels=labels_male), pd.cut(data['BMI'], bins=bins_female, labels=labels_female))
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

# Diet Plan Suggestions
def suggest_diet(weight, gender, exercise):
    if gender == 'Male':
        calories = 10 * weight + 6.25 * 170 - 5 * 30 + 5 if exercise else 0
    else:
        calories = 10 * weight + 6.25 * 160 - 5 * 30 + 5 if exercise else 0
    
    diet_plan = f"Daily Calorie Intake: {calories} kcal\n\nSuggested Diet Plan:\n- Include a variety of fruits and vegetables\n- Choose whole grains over refined grains\n- Include lean protein sources such as fish, poultry, and legumes\n- Limit saturated fats and sugary foods\n- Stay hydrated by drinking plenty of water"
    return diet_plan

# GUI
def predict_bmi():
    try:
        # Get user input
        height = float(entry_height.get())
        weight = float(entry_weight.get())
        gender = var_gender.get()  # Get selected gender
        exercise = var_exercise.get()  # Get exercise status
        
        # Calculate BMI
        bmi = weight / (height / 100) ** 2
        
        # Predict category
        user_input = np.array([[height, weight]])
        category = knn.predict(user_input)[0]
        
        # Suggest diet plan
        diet_plan = suggest_diet(weight, gender, exercise)
        
        # Show result
        result = f"BMI: {bmi:.2f}\nCategory: {category}\n\n{diet_plan}"
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

# Gender selection
var_gender = tk.StringVar()
var_gender.set('Male')  # Default selection
tk.Label(root, text="Gender").grid(row=2, column=0)
tk.OptionMenu(root, var_gender, 'Male', 'Female').grid(row=2, column=1)

# Exercise selection
var_exercise = tk.BooleanVar()
var_exercise.set(False)  # Default selection
tk.Checkbutton(root, text="Regular Exercise", variable=var_exercise).grid(row=3, columnspan=2)

# Create Predict button
tk.Button(root, text="Predict BMI", command=predict_bmi).grid(row=4, columnspan=2)

# Start the GUI event loop
root.mainloop()

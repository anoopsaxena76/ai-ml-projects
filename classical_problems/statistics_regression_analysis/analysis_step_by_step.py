import csv
# We have imported data of number of hours a student has studied and the corresponding marks obtained in an exam.
import numpy as np
from statistics import mean as builtin_mean, median as builtin_median, mode as builtin_mode
from statistics import variance as builtin_variance, stdev as builtin_stdev
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


hours_studied = []
marks_obtained = []

# This code reads data from a CSV file and prints the hours studied along with the marks obtained.
# The data is expected to be in the format:
# Hours studied, Marks obtained
with open ('student_scores.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        hours_studied.append(float(row[0]))
        marks_obtained.append(float(row[1]))

#for hour, marks in zip(hours_studied, marks_obtained):
#    print (f"Hours studied: {hour}, Marks obtained: {marks}")


def mean(data):
    total = 0
    elems = 0
    for value in data:
        total += value
        elems += 1
    return total / elems if elems > 0 else 0

def median(data):
    sorted_data = sorted (data)
    n = len(sorted_data)
    if n == 0:
        return 0
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def mode(data):
    frequency = {}
    for value in data: 
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    max_freq = max(frequency.values())
    modes = [key for key, freq in frequency.items() if freq == max_freq]
    return modes if len(modes) > 0 else None

def variance(data):
    if len(data) == 0:
        return 0
    mean_value = mean(data)
    total = sum((x - mean_value) ** 2 for x in data)
    return total / len(data)

def standard_deviation(data):
    return variance(data) ** 0.5


# Calculate and print the mean, median, mode, variance, and standard deviation of hours studied and marks obtained
''''''
#print(f"Mean of hours studied: {mean(hours_studied)}")
#print(f"Median of hours studied: {median(hours_studied)}")
#print(f"Mode of hours studied: {mode(hours_studied)}")
#print(f"Variance of hours studied: {variance(hours_studied)}")
#print(f"Standard Deviation of hours studied: {standard_deviation(hours_studied)}")
#print(f"Mean of marks obtained: {mean(marks_obtained)}")
#print(f"Median of marks obtained: {median(marks_obtained)}")
#print(f"Mode of marks obtained: {mode(marks_obtained)}")
#print(f"Variance of marks obtained: {variance(marks_obtained)}")
#print(f"Standard Deviation of marks obtained: {standard_deviation(marks_obtained)}")
''''''

# Calculate the slope and intercept for the linear regression of hours studied vs marks obtained
def linear_regression(feature, target):
    n = len(feature)
    if n == 0:
        return 0, 0
    mean_x = mean(feature)
    mean_y = mean(target)
    
    numerator = sum((feature[i] - mean_x) * (target[i] - mean_y) for i in range(n))
    denominator = sum((feature[i] - mean_x) ** 2 for i in range(n)) 
    
    if denominator == 0:
        return 0, mean_y  # Avoid division by zero
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    return slope, intercept 


# Now building the platform for turning this regression into classification so that we can look at classification matrix.
def predict(slope, intercept, x):
    return [slope * xvalue + intercept for xvalue in x] 


def to_classify (mark, threshold=50):
    return 1 if mark >= threshold else 0

actual_classification = [to_classify(mark) for mark in marks_obtained]
predicted_classification = [to_classify(mark) for mark in predict(*linear_regression(hours_studied, marks_obtained), hours_studied)]

def confusion_matrix(actual, predicted):
    TP = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    TN = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    FP = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    FN = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
    return TP, TN, FP, FN


def classification_matrix(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

TP, TN, FP, FN = confusion_matrix(actual_classification, predicted_classification)
m, c = linear_regression(hours_studied, marks_obtained)
# Reshape input for sklearn (needs 2D array)
X = np.array(hours_studied).reshape(-1, 1)
y = np.array(marks_obtained)

# Create and train the model
reg = LinearRegression()
reg.fit(X, y)

class_matrix = classification_matrix(TP, TN, FP, FN)


print(f"True Positives: {TP}, True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}")  
print("Classification Matrix:", c)
print("Slope and Intercept of the regression line:", linear_regression(hours_studied, marks_obtained))


actual_classes_np = np.array(actual_classification)
predicted_classes_np = np.array(predicted_classification)

# Built-in classification metrics
acc = accuracy_score(actual_classes_np, predicted_classes_np)
prec = precision_score(actual_classes_np, predicted_classes_np, zero_division=0)
rec = recall_score(actual_classes_np, predicted_classes_np, zero_division=0)
f1 = f1_score(actual_classes_np, predicted_classes_np, zero_division=0)

print("\n=== VALIDATION: Manual vs Built-in ===")

# Mean
print(f"Mean (Hours) → Manual: {mean(hours_studied):.2f}, Built-in: {builtin_mean(hours_studied):.2f}")
print(f"Mean (Marks) → Manual: {mean(marks_obtained):.2f}, Built-in: {builtin_mean(marks_obtained):.2f}")

# Median
print(f"Median (Hours) → Manual: {median(hours_studied):.2f}, Built-in: {builtin_median(hours_studied):.2f}")
print(f"Median (Marks) → Manual: {median(marks_obtained):.2f}, Built-in: {builtin_median(marks_obtained):.2f}")

# Mode
print(f"Mode (Hours) → Manual: {mode(hours_studied)}, Built-in: {builtin_mode(hours_studied)}")
print(f"Mode (Marks) → Manual: {mode(marks_obtained)}, Built-in: {builtin_mode(marks_obtained)}")

# Variance
print(f"Variance (Hours) → Manual: {variance(hours_studied):.2f}, Built-in: {builtin_variance(hours_studied):.2f}")
print(f"Variance (Marks) → Manual: {variance(marks_obtained):.2f}, Built-in: {builtin_variance(marks_obtained):.2f}")


print("\n--- Linear Regression Comparison ---")
print(f"Slope (m) → Manual: {m:.4f}, Built-in: {reg.coef_[0]:.4f}")
print(f"Intercept (c) → Manual: {c:.4f}, Built-in: {reg.intercept_:.4f}")

print("\n--- Classification Metrics Comparison ---")
print(f"Accuracy → Manual: {class_matrix['Accuracy']:.2f}, Built-in: {acc:.2f}")
print(f"Precision → Manual: {class_matrix['Precision']:.2f}, Built-in: {prec:.2f}")
print(f"Recall → Manual: {class_matrix['Recall']:.2f}, Built-in: {rec:.2f}")
print(f"F1 Score → Manual: {class_matrix['F1 Score']:.2f}, Built-in: {f1:.2f}")

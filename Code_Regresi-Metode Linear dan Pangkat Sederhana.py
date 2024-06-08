import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Memuat data dari file CSV
data = pd.read_csv('Student_Performance.csv')
TB = data['Hours Studied']
NT = data['Performance Index']

# Metode 1: Model Linear
def linear_regression(TB, NT):
    TB = TB.values.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(TB, NT)
    NT_pred = linear_model.predict(TB)
    return linear_model.coef_[0], linear_model.intercept_, NT_pred

# Metode 2: Model Pangkat Sederhana
def power_law(x, a, b):
    return a * np.power(x, b)

def power_regression(TB, NT):
    popt, pcov = curve_fit(power_law, TB, NT)
    NT_pred = power_law(TB, *popt)
    return popt, NT_pred

# Menjalankan regresi linear
slope, intercept, NT_pred_linear = linear_regression(TB, NT)
print(f"Linear Regression: slope = {slope}, intercept = {intercept}")

# Menjalankan regresi pangkat sederhana
popt, NT_pred_power = power_regression(TB, NT)
print(f"Power Regression: a = {popt[0]}, b = {popt[1]}")

# Visualisasi hasil regresi
plt.figure(figsize=(14, 7))

# Plot regresi linear
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, color='black', label='Data Siswa')
plt.plot(TB, NT_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear')
plt.legend()

# Plot regresi pangkat sederhana
plt.subplot(1, 2, 2)
plt.scatter(TB, NT, color='black', label='Data Siswa')
plt.plot(TB, NT_pred_power, color='green', label='Regresi Pangkat Sederhana')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Pangkat Sederhana')
plt.legend()

plt.tight_layout()
plt.show()

# Hitung galat RMS
galat_rms_linear = np.sqrt(np.mean((NT - NT_pred_linear) ** 2))
galat_rms_power = np.sqrt(np.mean((NT - NT_pred_power) ** 2))
print(f"Galat RMS Regresi Linear: {galat_rms_linear}")
print(f"Galat RMS Regresi Pangkat Sederhana: {galat_rms_power}")
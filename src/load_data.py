import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path)
    print("=== PODSTAWOWE INFORMACJE O DANYCH ===")
    print("\n--- Pierwsze 5 wierszy ---")
    print(df.head())
    print("\n--- Informacje o typach danych ---")
    print(df.info())
    print("\n--- Statystyki opisowe ---") 
    print(df.describe())
    print("\n--- Lista kolumn ---")
    print(df.columns.tolist())
    return df


def identify_data_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print("\n=== TYPY DANYCH ===")
    print(f"Kolumny numeryczne: {num_cols.tolist()}")
    print(f"Kolumny kategoryczne: {cat_cols.tolist()}")
    return num_cols, cat_cols


def check_missing_values(df):
    print("\n=== BRAKI DANYCH ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Liczba braków': missing,
        'Procent braków': missing_percent
    })
    print(missing_info[missing_info['Liczba braków'] > 0].sort_values('Liczba braków', ascending=False))
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_info.index, y=missing_info['Procent braków'])
    plt.title('Procent brakujących danych w każdej kolumnie')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


path_to_data = 'jobpredictor.csv'
df = load_data(path_to_data)
num_cols, cat_cols = identify_data_types(df)
check_missing_values(df)

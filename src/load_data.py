import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_data(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine='python', encoding='latin1')
    
    print("\n=== PODSTAWOWE INFORMACJE O DANYCH ===\n")

    print("--- Pierwsze 5 wierszy ---")
    print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n--- Informacje o typach danych ---")
    info_df = pd.DataFrame({
        'Kolumna': df.columns,
        'Typ danych': df.dtypes.astype(str),
        'Liczba braków': df.isna().sum(),
        'Unikalne wartości': [df[col].nunique() for col in df.columns]
    })
    print(tabulate(info_df, headers='keys', tablefmt='github', showindex=False))

    print("\n--- Statystyki opisowe (numeryczne) ---")
    desc = df.describe().T
    print(tabulate(desc, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))

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


path_to_data = 'data/SalaryDataFinall.csv'
df = load_data(path_to_data)
num_cols, cat_cols = identify_data_types(df)
check_missing_values(df)

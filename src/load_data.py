import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_titanic_data(path):
    """
    Wczytuje dane Titanic i wyświetla podstawowe informacje
    
    Args:
        path (str): Ścieżka do pliku CSV z danymi Titanic
        
    Returns:
        pd.DataFrame: Wczytane dane
    """
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
    """
    Identyfikuje typy danych - numeryczne i kategoryczne
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        tuple: (numerical_columns, categorical_columns)
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print("\n=== TYPY DANYCH ===")
    print(f"Kolumny numeryczne: {num_cols.tolist()}")
    print(f"Kolumny kategoryczne: {cat_cols.tolist()}")
    
    return num_cols, cat_cols


def check_missing_values(df):
    """
    Sprawdza braki danych w każdej kolumnie
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
    """
    print("\n=== BRAKI DANYCH ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Liczba braków': missing,
        'Procent braków': missing_percent
    })
    
    print(missing_info[missing_info['Liczba braków'] > 0].sort_values('Liczba braków', ascending=False))
    
    plt.tight_layout()
    plt.show()

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


def visualize_data_distribution(df):
    """
    Tworzy wizualizacje rozkładu danych
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
    """
    plt.figure(figsize=(15, 10))
    
    # Rozkład wieku
    plt.subplot(2, 3, 1)
    sns.histplot(df['Age'].dropna(), kde=True, bins=30)
    plt.title('Rozkład wieku pasażerów')
    plt.xlabel('Wiek')
    plt.ylabel('Liczebność')
    
    # Przeżywalność
    plt.subplot(2, 3, 2)
    sns.countplot(data=df, x='Survived')
    plt.title('Rozkład przeżywalności')
    plt.xlabel('Przeżył (0=Nie, 1=Tak)')
    plt.ylabel('Liczba pasażerów')
    
    # Cena biletu według klasy
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='Pclass', y='Fare')
    plt.title('Cena biletu według klasy')
    plt.xlabel('Klasa')
    plt.ylabel('Cena biletu')
    
    # Rozkład płci
    plt.subplot(2, 3, 4)
    sns.countplot(data=df, x='Sex')
    plt.title('Rozkład płci')
    plt.xlabel('Płeć')
    plt.ylabel('Liczba pasażerów')
    
    # Port zaokrętowania
    plt.subplot(2, 3, 5)
    sns.countplot(data=df, x='Embarked')
    plt.title('Port zaokrętowania')
    plt.xlabel('Port')
    plt.ylabel('Liczba pasażerów')
    
    # Mapa korelacji (tylko dla danych numerycznych)
    plt.subplot(2, 3, 6)
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Mapa korelacji cech numerycznych')
    
    plt.tight_layout()
    plt.show()

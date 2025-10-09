import pandas as pd
import numpy as np


def create_new_features(df):
    """
    Tworzy nowe cechy na podstawie istniejących danych
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z nowymi cechami
    """
    df = df.copy()
    
    print("\n=== TWORZENIE NOWYCH CECH ===")
    
    # Czy pasażer jest seniorem (powyżej 60 lat)
    df['is_senior'] = (df['Age'] > 60).astype(int)
    print("✓ Utworzono cechę 'is_senior' (1 jeśli wiek > 60)")
    
    # Czy pasażer jest dzieckiem (poniżej 12 lat)
    df['is_child'] = (df['Age'] < 12).astype(int)
    print("✓ Utworzono cechę 'is_child' (1 jeśli wiek < 12)")
    
    # Rozmiar rodziny (SibSp + Parch + 1)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    print("✓ Utworzono cechę 'family_size' (SibSp + Parch + 1)")
    
    # Czy pasażer podróżuje sam
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    print("✓ Utworzono cechę 'is_alone' (1 jeśli podróżuje sam)")
    
    # Czy ma kabinę (sprawdzenie czy Cabin nie jest null)
    if 'Cabin' in df.columns:
        df['has_cabin'] = df['Cabin'].notnull().astype(int)
        print("✓ Utworzono cechę 'has_cabin' (1 jeśli znana kabina)")
    
    # Wyciągnięcie tytułu z nazwiska
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+\.)', expand=False)
        print("✓ Utworzono cechę 'Title' (tytuł wyciągnięty z nazwiska)")
        
        # Grupowanie rzadkich tytułów
        title_counts = df['Title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        print("✓ Pogrupowano rzadkie tytuły w kategorię 'Rare'")
    
    # Kategoryzacja wieku
    def categorize_age(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 12:
            return 'Child'
        elif age < 18:
            return 'Teen'
        elif age < 60:
            return 'Adult'
        else:
            return 'Senior'
    
    df['age_group'] = df['Age'].apply(categorize_age)
    print("✓ Utworzono cechę 'age_group' (Child/Teen/Adult/Senior/Unknown)")
    
    # Kategoryzacja ceny biletu
    def categorize_fare(fare):
        if pd.isna(fare):
            return 'Unknown'
        elif fare <= 7.91:
            return 'Low'
        elif fare <= 14.454:
            return 'Medium_Low'
        elif fare <= 31:
            return 'Medium'
        else:
            return 'High'
    
    df['fare_range'] = df['Fare'].apply(categorize_fare)
    print("✓ Utworzono cechę 'fare_range' (Low/Medium_Low/Medium/High)")
    
    print(f"\nLiczba nowych cech: {len(df.columns) - len(df.columns)}")
    print(f"Aktualna liczba kolumn: {len(df.columns)}")
    
    return df


def extract_cabin_features(df):
    """
    Wydobywa dodatkowe informacje z kolumny Cabin
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z nowymi cechami z kabiny
    """
    if 'Cabin' not in df.columns:
        return df
    
    df = df.copy()
    
    # Wyciągnięcie litery z numeru kabiny (pokład)
    df['cabin_deck'] = df['Cabin'].str.extract('([A-Za-z])', expand=False)
    
    # Liczba pokoi (liczba przestrzeni w Cabin)
    df['cabin_multiple'] = df['Cabin'].apply(
        lambda x: 0 if pd.isna(x) else len(str(x).split(' '))
    )
    
    print("✓ Utworzono cechy związane z kabiną: 'cabin_deck', 'cabin_multiple'")
    
    return df


def create_interaction_features(df):
    """
    Tworzy cechy interakcyjne (kombinacje istniejących cech)
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z cechami interakcyjnymi
    """
    df = df.copy()
    
    # Interakcja wieku i płci
    if 'Age' in df.columns and 'Sex' in df.columns:
        df['age_sex'] = df['Age'].astype(str) + '_' + df['Sex'].astype(str)
    
    # Interakcja klasy i płci
    if 'Pclass' in df.columns and 'Sex' in df.columns:
        df['class_sex'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str)
    
    # Interakcja rozmiaru rodziny i klasy
    if 'family_size' in df.columns and 'Pclass' in df.columns:
        df['family_class'] = df['family_size'].astype(str) + '_' + df['Pclass'].astype(str)
    
    print("✓ Utworzono cechy interakcyjne")
    
    return df

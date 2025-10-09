import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler


def handle_missing_values(df, threshold=0.3):
    """
    Obsługuje braki danych w różnych kolumnach
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        threshold (float): Próg braków - jeśli kolumna ma więcej braków, zostanie usunięta
        
    Returns:
        pd.DataFrame: DataFrame z obsłużonymi brakami
    """
    df = df.copy()
    
    print("\n=== OBSŁUGA BRAKÓW DANYCH ===")
    
    # Sprawdzenie procentu braków w każdej kolumnie
    missing_percent = df.isnull().mean()
    
    # Usunięcie kolumn z dużą liczbą braków
    cols_to_drop = missing_percent[missing_percent > threshold].index
    if len(cols_to_drop) > 0:
        print(f"Usuwanie kolumn z więcej niż {threshold*100}% braków: {list(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    # Uzupełnienie braków w kolumnie Age
    if 'Age' in df.columns and df['Age'].isnull().sum() > 0:
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
        print(f"✓ Uzupełniono braki w 'Age' medianą: {median_age:.1f}")
    
    # Uzupełnienie braków w kolumnie Fare
    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        median_fare = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(median_fare)
        print(f"✓ Uzupełniono braki w 'Fare' medianą: {median_fare:.2f}")
    
    # Uzupełnienie braków w kolumnie Embarked
    if 'Embarked' in df.columns and df['Embarked'].isnull().sum() > 0:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode_embarked)
        print(f"✓ Uzupełniono braki w 'Embarked' dominantą: {mode_embarked}")
    
    # Sprawdzenie czy zostały jakieś braki
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"⚠️  Pozostało {remaining_nulls} braków w danych")
    else:
        print("✓ Wszystkie braki zostały obsłużone")
    
    return df


def encode_categorical_features(df, encoding_method='onehot'):
    """
    Koduje zmienne kategoryczne
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        encoding_method (str): Metoda kodowania ('onehot' lub 'label')
        
    Returns:
        pd.DataFrame: DataFrame z zakodowanymi zmiennymi
    """
    df = df.copy()
    
    print(f"\n=== KODOWANIE ZMIENNYCH KATEGORYCZNYCH ({encoding_method.upper()}) ===")
    
    # Znajdź kolumny kategoryczne
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Usuń kolumny, które nie powinny być kodowane (np. Name, jeśli istnieje)
    cols_to_exclude = ['Name', 'Ticket']  # Te kolumny często zawierają unikalne wartości
    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
    
    if encoding_method == 'onehot':
        # One-Hot Encoding
        for col in categorical_cols:
            if col in df.columns:
                print(f"✓ Kodowanie One-Hot dla kolumny: {col}")
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
    
    elif encoding_method == 'label':
        # Label Encoding
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                print(f"✓ Kodowanie Label dla kolumny: {col}")
    
    return df


def scale_numerical_features(df, scaler_type='standard', columns_to_scale=None):
    """
    Skaluje cechy numeryczne
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        scaler_type (str): Typ skalowania ('standard', 'minmax', 'robust')
        columns_to_scale (list): Lista kolumn do skalowania. Jeśli None, skaluje wszystkie numeryczne
        
    Returns:
        tuple: (scaled_dataframe, scaler_object)
    """
    df = df.copy()
    
    print(f"\n=== SKALOWANIE CECH NUMERYCZNYCH ({scaler_type.upper()}) ===")
    
    # Znajdź kolumny numeryczne
    if columns_to_scale is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # Usuń zmienną docelową jeśli istnieje
        numerical_cols = [col for col in numerical_cols if col not in ['Survived']]
    else:
        numerical_cols = columns_to_scale
    
    # Wybierz odpowiedni scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("Używam StandardScaler (średnia=0, std=1)")
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        print("Używam MinMaxScaler (zakres 0-1)")
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        print("Używam RobustScaler (odporny na outliers)")
    else:
        raise ValueError("Nieobsługiwany typ scalera. Użyj: 'standard', 'minmax', lub 'robust'")
    
    # Skalowanie kolumn
    if len(numerical_cols) > 0:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print(f"✓ Przeskalowano kolumny: {list(numerical_cols)}")
    else:
        print("⚠️  Nie znaleziono kolumn numerycznych do skalowania")
    
    return df, scaler


def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Usuwa outliers z danych numerycznych
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        columns (list): Lista kolumn do sprawdzenia. Jeśli None, sprawdza wszystkie numeryczne
        method (str): Metoda wykrywania outliers ('iqr' lub 'zscore')
        threshold (float): Próg dla IQR (1.5) lub Z-score (3.0)
        
    Returns:
        pd.DataFrame: DataFrame bez outliers
    """
    df = df.copy()
    original_shape = df.shape
    
    print(f"\n=== USUWANIE OUTLIERS (metoda: {method.upper()}) ===")
    
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
        columns = [col for col in columns if col not in ['Survived']]
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_count = len(df[z_scores > threshold])
                df = df[z_scores <= threshold]
            
            if outliers_count > 0:
                print(f"✓ Usunięto {outliers_count} outliers z kolumny '{col}'")
    
    removed_rows = original_shape[0] - df.shape[0]
    print(f"Łącznie usunięto {removed_rows} wierszy ({removed_rows/original_shape[0]*100:.1f}%)")
    
    return df


def create_polynomial_features(df, columns=None, degree=2):
    """
    Tworzy cechy wielomianowe (np. kwadrat, sześcian)
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        columns (list): Lista kolumn do przekształcenia
        degree (int): Stopień wielomianu
        
    Returns:
        pd.DataFrame: DataFrame z cechami wielomianowymi
    """
    df = df.copy()
    
    print(f"\n=== TWORZENIE CECH WIELOMIANOWYCH (stopień: {degree}) ===")
    
    if columns is None:
        columns = ['Age', 'Fare']  # Domyślne kolumny dla Titanic
    
    for col in columns:
        if col in df.columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_power_{d}"
                df[new_col] = df[col] ** d
                print(f"✓ Utworzono {new_col}")
    
    return df

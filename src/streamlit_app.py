import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Importy do Machine Learning (Nowe)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- KONFIGURACJA ---
st.set_page_config(page_title="Job Market Explorer & AI", page_icon="📊", layout="wide")

DATA_FILENAME = "SalaryDataFinall.csv"
ROWS_PER_PAGE = 20


# ---------------------------------------------------------
# FUNKCJE DANYCH (Wspólne)
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Wczytaj i wstępnie przetwórz dane z pliku CSV."""
    # Próba znalezienia pliku w folderze data/ lub bieżącym
    paths_to_check = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", DATA_FILENAME)),
        DATA_FILENAME
    ]

    data_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(f"Nie znaleziono pliku {DATA_FILENAME}")

    df = pd.read_csv(data_path, sep=";")

    # Usuwanie BOM (Byte Order Mark) jeśli istnieje
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    # Konwersja kolumn liczbowych
    num_cols = [
        c for c in df.columns if any(substr in c for substr in ["Salary", "Company Size"])
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------------------------------------
# NOWE FUNKCJE DLA MODELU ML (Random Forest)
# ---------------------------------------------------------

@st.cache_data
def prepare_training_data(df):
    """
    Transformuje dane do formatu wymaganego przez ML:
    Rozbija wiersze na osobne przypadki B2B i UoP.
    """
    data_entries = []

    for _, row in df.iterrows():
        tech = row['Technology']
        seniority = row['Seniority']
        loc = row['Location']

        # 1. Sprawdzamy B2B (bierzemy średnią z widełek)
        if row['Salary B2B Min'] > 0 and row['Salary B2B Max'] > 0:
            avg_salary = (row['Salary B2B Min'] + row['Salary B2B Max']) / 2
            data_entries.append({
                'Technology': tech,
                'Seniority': seniority,
                'Location': loc,
                'Contract Type': 'B2B',
                'Salary': avg_salary
            })

        # 2. Sprawdzamy UoP (Employment)
        if row['Salary Employment Min'] > 0 and row['Salary Employment Max'] > 0:
            avg_salary = (row['Salary Employment Min'] + row['Salary Employment Max']) / 2
            data_entries.append({
                'Technology': tech,
                'Seniority': seniority,
                'Location': loc,
                'Contract Type': 'Employment',
                'Salary': avg_salary
            })

    return pd.DataFrame(data_entries)


@st.cache_resource
def build_and_train_model(data):
    """
    Trenuje model Random Forest. Używa cache, aby nie trenować przy każdym kliknięciu.
    """
    X = data[['Technology', 'Seniority', 'Location', 'Contract Type']]
    y = data['Salary']

    # Podział
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Preprocessing
    categorical_features = ['Technology', 'Seniority', 'Location', 'Contract Type']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Metryki
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


@st.cache_resource
def build_and_train_linear(data):
    """
    Trenuje prosty model Regresji Liniowej. Używa cache, aby nie trenować przy każdym kliknięciu.
    """
    X = data[['Technology', 'Seniority', 'Location', 'Contract Type']]
    y = data['Salary']

    # Podział
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Preprocessing
    categorical_features = ['Technology', 'Seniority', 'Location', 'Contract Type']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    # Metryki
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


# ---------------------------------------------------------
# HELPERY UI (Ze starego kodu)
# ---------------------------------------------------------

def format_numeric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sformatuj wartości numeryczne do wyświetlania w tabeli."""
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[np.number]).columns:
        def _fmt(x):
            if pd.isna(x) or int(x) == -1:
                return "-"
            return f"{int(x):,}".replace(",", " ")

        formatted[col] = formatted[col].apply(_fmt)
    return formatted


def paginate(df: pd.DataFrame, page_number: int, rows_per_page: int) -> Tuple[pd.DataFrame, int, int]:
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    return df.iloc[start_idx:end_idx], start_idx, end_idx


def _current_page() -> str:
    return st.session_state.get("page", "Strona główna")


def _set_page(name: str) -> None:
    st.session_state["page"] = name
    st.session_state["page_number"] = 1


# ---------------------------------------------------------
# STRONY APLIKACJI
# ---------------------------------------------------------

def render_home(df: pd.DataFrame) -> None:
    st.title("📊 Job Market Explorer")
    
    # About Dataset
    st.markdown("""
    ### O Datasecie
    Oferty pracy z rynku IT w Polsce zebrane w latach **2023-2024**. 
    Dataset zawiera różne technologie, poziomy doświadczenia oraz dane o wynagrodzeniach dla B2B i UoP.
    """)
    
    st.markdown("**Przykładowe dane:**")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Kluczowe statystyki
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liczba ofert", f"{len(df)}")
    avg_b2b = df[df['Salary B2B Max'] > 0]['Salary B2B Max'].median()
    col2.metric("Mediana B2B", f"{int(avg_b2b):,} PLN".replace(",", " "))
    avg_uop = df[df['Salary Employment Max'] > 0]['Salary Employment Max'].median()
    col3.metric("Mediana UoP", f"{int(avg_uop):,} PLN".replace(",", " "))
    col4.metric("Technologie", f"{df['Technology'].nunique()}")
    
    st.divider()
    
    # Top 10 technologii wg średnich zarobków B2B
    st.subheader("🏆 Top 10 najlepiej płatnych technologii (B2B)")
    top_tech = df[df['Salary B2B Max'] > 0].groupby('Technology')['Salary B2B Max'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_tech)
    
    st.divider()
    
    # Zarobki vs Doświadczenie
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Zarobki vs Doświadczenie (B2B)")
        sen_b2b = df[df['Salary B2B Max'] > 0].groupby('Seniority')['Salary B2B Max'].mean().reindex(['junior', 'mid', 'senior', 'expert'])
        st.bar_chart(sen_b2b)
    
    with col2:
        st.subheader("📊 Liczba ofert wg doświadczenia")
        sen_counts = df.groupby('Seniority').size().reindex(['junior', 'mid', 'senior', 'expert'])
        st.bar_chart(sen_counts)


# ---------------------------------------------------------
# STRONA: ESTYMATORY AI (PORÓWNANIE)
# ---------------------------------------------------------
def render_estimators(raw_df: pd.DataFrame) -> None:
    st.title("🤖 Predykcja Zarobków - Porównanie Modeli")
    
    # Przygotowanie danych
    clean_df = prepare_training_data(raw_df)
    
    # Trening obu modeli
    with st.spinner('Trenowanie modeli...'):
        rf_model, rf_mae, rf_r2 = build_and_train_model(clean_df)
        lr_model, lr_mae, lr_r2 = build_and_train_linear(clean_df)
    
    # Porównanie modeli
    st.subheader("📊 Jakość Modeli")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌲 Random Forest - MAE", f"{rf_mae:,.0f} PLN")
        st.metric("🌲 Random Forest - R²", f"{rf_r2:.3f}")
    with col2:
        st.metric("📐 Linear Regression - MAE", f"{lr_mae:,.0f} PLN")
        st.metric("📐 Linear Regression - R²", f"{lr_r2:.3f}")
    
    st.divider()
    
    # Formularz predykcji
    st.subheader("🔮 Przewidywanie Wynagrodzenia")
    col1, col2 = st.columns(2)
    with col1:
        tech_input = st.selectbox("Technologia", options=sorted(clean_df['Technology'].unique()))
        seniority_input = st.selectbox("Doświadczenie", options=['junior', 'mid', 'senior', 'expert'])
    with col2:
        loc_input = st.selectbox("Lokalizacja", options=sorted(clean_df['Location'].unique()))
        contract_input = st.selectbox("Typ umowy", options=['B2B', 'Employment'])
    
    if st.button("💰 Przewiduj wynagrodzenie", type="primary"):
        input_data = pd.DataFrame({
            'Technology': [tech_input],
            'Seniority': [seniority_input],
            'Location': [loc_input],
            'Contract Type': [contract_input]
        })
        
        rf_pred = rf_model.predict(input_data)[0]
        lr_pred = lr_model.predict(input_data)[0]
        
        st.success("### Przewidywania")
        c1, c2 = st.columns(2)
        c1.metric("🌲 Random Forest", f"{rf_pred:,.0f} PLN")
        c2.metric("📐 Linear Regression", f"{lr_pred:,.0f} PLN")
        
        diff = abs(rf_pred - lr_pred)
        st.info(f"💡 Różnica między modelami: **{diff:,.0f} PLN** ({diff/rf_pred*100:.1f}%)")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    if "page" not in st.session_state: st.session_state["page"] = "Strona główna"
    if "page_number" not in st.session_state: st.session_state["page_number"] = 1

    # SIDEBAR
    with st.sidebar:
        st.title("🎯 Menu")
        if st.button("📊 Dashboard", use_container_width=True): _set_page("Strona główna")
        if st.button("🤖 Predykcja AI", use_container_width=True): _set_page("Estymatory")

    # LOAD DATA
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Błąd danych: {e}")
        return

    # ROUTING
    pg = _current_page()
    if pg == "Strona główna":
        render_home(df)
    elif pg == "Estymatory":
        render_estimators(df)


if __name__ == "__main__":
    main()
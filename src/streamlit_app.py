import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Importy do Machine Learning (Nowe)
from sklearn.ensemble import RandomForestRegressor
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
    st.markdown("Przegląd ofert pracy (2023-2024).")

    # Filtry
    col1, col2, col3 = st.columns(3)
    with col1:
        tech_filter = st.selectbox("Technologia:", ["Wszystkie"] + sorted(df["Technology"].dropna().unique().tolist()))
    with col2:
        loc_filter = st.selectbox("Lokalizacja:", ["Wszystkie"] + sorted(df["Location"].dropna().unique().tolist()))
    with col3:
        sen_filter = st.selectbox("Poziom:", ["Wszystkie"] + sorted(df["Seniority"].dropna().unique().tolist()))

    filtered = df.copy()
    if tech_filter != "Wszystkie": filtered = filtered[filtered["Technology"] == tech_filter]
    if loc_filter != "Wszystkie": filtered = filtered[filtered["Location"] == loc_filter]
    if sen_filter != "Wszystkie": filtered = filtered[filtered["Seniority"] == sen_filter]

    st.caption(f"Liczba ofert: {len(filtered)}")

    # Tabela
    page_number = st.session_state.get("page_number", 1)
    df_view = format_numeric_table(filtered)
    if "ID" in df_view.columns:
        df_view["ID"] = df_view["ID"].astype(str).apply(lambda x: (x[:8] + "…") if len(x) > 9 else x)

    slice_df, start_idx, end_idx = paginate(df_view, page_number, ROWS_PER_PAGE)
    st.dataframe(slice_df, use_container_width=True, hide_index=True)

    # Paginacja
    c_prev, _, c_next = st.columns([1, 10, 1])
    if c_prev.button("Poprzednia") and page_number > 1:
        st.session_state["page_number"] -= 1
        st.rerun()
    if c_next.button("Następna") and end_idx < len(filtered):
        st.session_state["page_number"] += 1
        st.rerun()


def render_salary_chart(df: pd.DataFrame) -> None:
    st.title("📈 Wykres zarobków wg technologii")
    contract_type = st.radio("Typ umowy:", ("Umowa o pracę", "B2B"), horizontal=True)
    seniority = st.selectbox("Poziom:", ["Wszystkie", "junior", "mid", "senior", "expert"])

    filtered = df if seniority == "Wszystkie" else df[df["Seniority"] == seniority]
    col_target = "Salary Employment Max" if contract_type == "Umowa o pracę" else "Salary B2B Max"

    # Obliczanie średniej bez -1
    avg = filtered[filtered[col_target] > 0].groupby("Technology")[col_target].mean().sort_values(ascending=False)

    if not avg.empty:
        st.bar_chart(avg)
    else:
        st.warning("Brak danych dla wybranych kryteriów.")


def render_eda(df: pd.DataFrame) -> None:
    st.title("Eksploracyjna Analiza Danych (EDA)")
    st.info("Tutaj znajdują się statyczne analizy wygenerowane wcześniej.")

    c1, c2 = st.columns(2)
    img1 = os.path.join(os.path.dirname(__file__), "company_size_vs_junior_salary.png")
    img2 = os.path.join(os.path.dirname(__file__), "salary_by_seniority.png")

    if os.path.exists(img1): c1.image(img1, caption="Junior vs Wielkość Firmy")
    if os.path.exists(img2): c2.image(img2, caption="Zarobki vs Seniority")


def render_calculator_stats(df: pd.DataFrame) -> None:
    """Stary kalkulator oparty na medianie i kwartylach."""
    st.title("🧮 Kalkulator Rynkowy (Statystyka)")
    st.markdown("Analiza historycznych danych (Mediana / Kwartyle).")

    c1, c2, c3 = st.columns(3)
    tech = c1.selectbox("Technologia", sorted(df["Technology"].unique()))
    seniority = c2.selectbox("Poziom", sorted(df["Seniority"].unique()))
    loc = c3.selectbox("Lokalizacja", ["Cała Polska"] + sorted(df["Location"].unique().tolist()))
    contract = st.radio("Umowa", ["B2B", "Umowa o pracę"], horizontal=True)

    mask = (df["Technology"] == tech) & (df["Seniority"] == seniority)
    if loc != "Cała Polska": mask &= (df["Location"] == loc)

    col_name = "Salary B2B Max" if contract == "B2B" else "Salary Employment Max"
    vals = df[mask][col_name]
    valid = vals[vals > 0]

    st.divider()
    if valid.empty:
        st.warning("Brak danych historycznych dla tych kryteriów.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Min (25%)", f"{int(valid.quantile(0.25)):,} PLN".replace(",", " "))
    m2.metric("Mediana", f"{int(valid.median()):,} PLN".replace(",", " "))
    m3.metric("Max (75%)", f"{int(valid.quantile(0.75)):,} PLN".replace(",", " "))


# ---------------------------------------------------------
# NOWA STRONA: ESTYMATOR AI (Random Forest)
# ---------------------------------------------------------
def render_ml_estimator(raw_df: pd.DataFrame) -> None:
    st.title("Estymator Zarobków AI")
    st.markdown("""
    Ten moduł wykorzystuje **uczenie maszynowe (Random Forest)**, aby oszacować potencjalne wynagrodzenie.
    Model "nauczył się" zależności między technologią, miastem a zarobkami na podstawie całego zbioru danych.
    """)

    # 1. Przygotowanie danych
    clean_df = prepare_training_data(raw_df)

    # 2. Trening (Cache zadba, by nie robić tego ciągle)
    with st.spinner('Trwa trenowanie modelu AI... (może chwilę potrwać przy pierwszym uruchomieniu)'):
        model, mae, r2 = build_and_train_model(clean_df)

    # Wyświetlenie jakości modelu
    with st.expander("ℹ️ Informacje o jakości modelu (Metryki)"):
        c1, c2 = st.columns(2)
        c1.metric("Średni błąd (MAE)", f"{mae:,.0f} PLN", help="O tyle średnio model myli się w przewidywaniach.")
        c2.metric("Dopasowanie (R2 Score)", f"{r2:.2f}",
                  help="1.0 to ideał. Powyżej 0.5 w płacach to często dobry wynik.")

    st.divider()

    st.subheader("Wprowadź parametry stanowiska")

    # Formularz
    col1, col2 = st.columns(2)
    with col1:
        tech_input = st.selectbox("Technologia", options=sorted(clean_df['Technology'].unique()), key="ml_tech")
        seniority_input = st.selectbox("Doświadczenie", options=['junior', 'mid', 'senior', 'expert'], key="ml_sen")
    with col2:
        loc_input = st.selectbox("Lokalizacja", options=sorted(clean_df['Location'].unique()), key="ml_loc")
        contract_input = st.selectbox("Typ umowy", options=['B2B', 'Employment'], key="ml_con")

    if st.button("🔮 Oszacuj wynagrodzenie", type="primary"):
        # Przygotowanie danych wejściowych
        input_data = pd.DataFrame({
            'Technology': [tech_input],
            'Seniority': [seniority_input],
            'Location': [loc_input],
            'Contract Type': [contract_input]
        })

        prediction = model.predict(input_data)[0]

        st.success(f"Szacowane wynagrodzenie: **{prediction:,.2f} PLN**")
        if contract_input == 'B2B':
            st.caption("Jest to szacowana kwota netto na fakturze (bez VAT).")
        else:
            st.caption("Jest to szacowana kwota brutto na umowie o pracę.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    if "page" not in st.session_state: st.session_state["page"] = "Strona główna"
    if "page_number" not in st.session_state: st.session_state["page_number"] = 1

    # SIDEBAR
    with st.sidebar:
        st.title("Menu")
        if st.button("🏠 Strona główna", use_container_width=True): _set_page("Strona główna")
        if st.button("🧮 Kalkulator (Statystyka)", use_container_width=True): _set_page("Kalkulator")
        if st.button("Estymator (ML)", use_container_width=True): _set_page("Estymator AI")  # <--- NOWOŚĆ
        if st.button("📈 Wykresy", use_container_width=True): _set_page("Wykresy")
        if st.button("🔍 Analiza EDA", use_container_width=True): _set_page("EDA")

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
    elif pg == "Kalkulator":
        render_calculator_stats(df)
    elif pg == "Estymator AI":
        render_ml_estimator(df)
    elif pg == "Wykresy":
        render_salary_chart(df)
    elif pg == "EDA":
        render_eda(df)


if __name__ == "__main__":
    main()
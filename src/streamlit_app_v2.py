import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Job Market Explorer", page_icon="📊", layout="wide")

DATA_FILENAME = "SalaryDataFinall.csv"
ROWS_PER_PAGE = 20


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Wczytaj i wstępnie przetwórz dane z pliku CSV."""
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", DATA_FILENAME)
    )

    df = pd.read_csv(data_path, sep=";")

    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    num_cols = [
        c for c in df.columns if any(substr in c for substr in ["Salary", "Company Size"])
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


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
    """Zwróć wycinek danych dla danej strony."""
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    return df.iloc[start_idx:end_idx], start_idx, end_idx


def _current_page() -> str:
    return st.session_state.get("page", "Strona główna")


def _set_page(name: str) -> None:
    st.session_state["page"] = name
    st.session_state["page_number"] = 1


def render_home(df: pd.DataFrame) -> None:
    st.title("📊 Job Market Explorer")

    st.subheader("O zbiorze danych")
    st.markdown(
        """
        Ten zbiór danych zawiera oferty pracy dla inżynierów oprogramowania w Polsce
        z okresu od **września 2023 do lipca 2024**.

        Główne kolumny:
        - **ID** – unikalny identyfikator oferty,
        - **Company / Company Size** – nazwa i wielkość firmy,
        - **Location** – lokalizacja stanowiska,
        - **Technology** – główna technologia / język programowania,
        - **Seniority** – poziom zaawansowania (junior, mid, senior, expert),
        - **Salary Employment Min / Max** – widełki wynagrodzenia na umowę o pracę,
        - **Salary B2B Min / Max** – widełki wynagrodzenia na kontrakt B2B.

        Dane pozwalają analizować rynek pracy pod kątem miasta, technologii,
        poziomu doświadczenia i formy zatrudnienia.
        """
    )

    st.subheader("Filtry ofert pracy")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tech_filter = st.selectbox(
            "Technologia:",
            options=["Wszystkie"] + sorted(df["Technology"].dropna().unique().tolist()),
        )
    with col_f2:
        loc_filter = st.selectbox(
            "Lokalizacja:",
            options=["Wszystkie"] + sorted(df["Location"].dropna().unique().tolist()),
        )
    with col_f3:
        sen_filter = st.selectbox(
            "Poziom zaawansowania:",
            options=["Wszystkie"] + sorted(df["Seniority"].dropna().unique().tolist()),
        )

    filtered = df.copy()
    if tech_filter != "Wszystkie":
        filtered = filtered[filtered["Technology"] == tech_filter]
    if loc_filter != "Wszystkie":
        filtered = filtered[filtered["Location"] == loc_filter]
    if sen_filter != "Wszystkie":
        filtered = filtered[filtered["Seniority"] == sen_filter]

    if filtered.empty:
        st.warning("Brak ofert dla wybranych filtrów.")
        return

    active_filters = []
    if tech_filter != "Wszystkie":
        active_filters.append(f"technologia = {tech_filter}")
    if loc_filter != "Wszystkie":
        active_filters.append(f"lokalizacja = {loc_filter}")
    if sen_filter != "Wszystkie":
        active_filters.append(f"poziom = {sen_filter}")

    if active_filters:
        st.caption("Filtry: " + ", ".join(active_filters))

    st.caption(f"Liczba ofert po zastosowaniu filtrów: {len(filtered)}")

    show_full_id = st.checkbox("Pokaż pełne ID", value=False)

    page_number = st.session_state.get("page_number", 1)

    df_view = filtered.copy()
    if not show_full_id and "ID" in df_view.columns:
        df_view["ID"] = df_view["ID"].astype(str).apply(
            lambda x: (x[:8] + "…") if len(x) > 9 else x
        )

    df_view = format_numeric_table(df_view)

    slice_df, start_idx, end_idx = paginate(df_view, page_number, ROWS_PER_PAGE)

    visible_rows = len(slice_df)
    row_height = 34
    header_extra = 70
    dynamic_height = visible_rows * row_height + header_extra

    try:
        st.dataframe(slice_df, use_container_width=True, hide_index=True, height=dynamic_height)
    except TypeError:
        st.dataframe(slice_df, use_container_width=True, height=dynamic_height)

    col_prev, col_spacer, col_next = st.columns([2, 14, 2])
    with col_prev:
        if st.button("Poprzednia", key="prev") and page_number > 1:
            st.session_state["page_number"] = page_number - 1
    with col_next:
        if st.button("Następna", key="next") and end_idx < len(filtered):
            st.session_state["page_number"] = page_number + 1


def render_salary_chart(df: pd.DataFrame) -> None:
    _c1, _cmain, _c3 = st.columns([1, 3, 1])
    with _cmain:
        st.title("📈 Wykres zarobków wg technologii")

        st.markdown(
            """
            Wykres przedstawia średnie **maksymalne** wynagrodzenie w zależności od technologii.
            Możesz wybrać typ umowy oraz poziom zaawansowania.
            """
        )

        try:
            contract_type = st.radio(
                "Wybierz typ umowy:", ("Łącznie", "Umowa o pracę", "B2B")
            )
            seniority_level = st.selectbox(
                "Wybierz poziom zaawansowania:",
                ("Wszystkie", "junior", "mid", "senior", "expert"),
            )

            if seniority_level != "Wszystkie":
                filtered_df = df[df["Seniority"] == seniority_level].copy()
            else:
                filtered_df = df.copy()

            def mean_excluding_minus_one(series: pd.Series) -> float:
                vals = series[(series != -1) & (~series.isna())]
                if len(vals) == 0:
                    return np.nan
                return vals.mean()

            if contract_type == "Umowa o pracę":
                avg_salaries = filtered_df.groupby("Technology")[
                    "Salary Employment Max"
                ].apply(mean_excluding_minus_one)
            elif contract_type == "B2B":
                avg_salaries = filtered_df.groupby("Technology")[
                    "Salary B2B Max"
                ].apply(mean_excluding_minus_one)
            else:
                stacked = pd.DataFrame(
                    {
                        "Technology": np.concatenate(
                            [
                                filtered_df["Technology"].values,
                                filtered_df["Technology"].values,
                            ]
                        ),
                        "Salary": np.concatenate(
                            [
                                filtered_df["Salary Employment Max"].values,
                                filtered_df["Salary B2B Max"].values,
                            ]
                        ),
                    }
                )
                stacked = stacked[(stacked["Salary"] != -1) & (~stacked["Salary"].isna())]
                avg_salaries = stacked.groupby("Technology")["Salary"].mean()

            avg_salaries = avg_salaries.dropna().sort_values(ascending=False)

            if avg_salaries.empty:
                st.warning("Brak danych do wyświetlenia wykresu dla wybranych filtrów.")
                return

            avg_salaries_int = avg_salaries.astype(int)

            st.caption(
                f"Liczba technologii na wykresie: {len(avg_salaries_int)} | Poziom: {seniority_level} | Typ umowy: {contract_type}"
            )

            st.bar_chart(avg_salaries_int)
        except Exception as e:
            st.error(f"Nie udało się wygenerować wykresu: {e}")


def render_eda(df: pd.DataFrame) -> None:
    _c1, _cmain, _c3 = st.columns([1, 3, 1])
    with _cmain:
        st.title("Eksploracyjna Analiza Danych (EDA)")

        st.markdown(
            """
            Eksploracyjna analiza danych (EDA) pozwala zrozumieć rozkład wynagrodzeń
            w zależności od wielkości firmy oraz poziomu zaawansowania.

            Poniżej prezentujemy dwa wybrane diagramy przygotowane wcześniej:
            - pensje juniorów względem wielkości firmy,
            - zarobki względem poziomu zaawansowania.
            """
        )

        img1_path = os.path.join(os.path.dirname(__file__), "company_size_vs_junior_salary.png")
        img2_path = os.path.join(os.path.dirname(__file__), "salary_by_seniority.png")

        if os.path.exists(img1_path):
            st.image(img1_path, caption="Pensje juniorów względem wielkości firmy")
        else:
            st.info("Brak pliku company_size_vs_junior_salary.png w katalogu src.")

        if os.path.exists(img2_path):
            st.image(img2_path, caption="Zarobki względem zaawansowania")
        else:
            st.info("Brak pliku salary_by_seniority.png w katalogu src.")

        st.markdown(
            """
            Pierwszy wykres pokazuje, jak średnia pensja juniorów zmienia się wraz z wielkością firmy
            (od mikro firm po bardzo duże organizacje).

            Drugi wykres porównuje rozkład wynagrodzeń między poziomami junior, mid, senior i expert,
            co pomaga zobaczyć, jak rośnie wynagrodzenie wraz z doświadczeniem.
            """
        )
def render_calculator(df: pd.DataFrame) -> None:
    st.title("Kalkulator Rynkowy")
    st.markdown("""
    Sprawdź, jak kształtują się stawki rynkowe dla konkretnego stanowiska.
    Wybierz parametry, aby zobaczyć medianę oraz zakres wynagrodzeń.
    """)

    # --- Filtry ---
    c1, c2, c3 = st.columns(3)
    with c1:
        tech = st.selectbox("Technologia", options=sorted(df["Technology"].unique()))
    with c2:
        seniority = st.selectbox("Poziom", options=sorted(df["Seniority"].unique()))
    with c3:
        # Dodajemy opcję "Cała Polska" dla lokalizacji
        loc_options = ["Cała Polska"] + sorted(df["Location"].unique().tolist())
        location = st.selectbox("Lokalizacja", options=loc_options)

    # --- Wybór typu umowy ---
    contract_type = st.radio("Typ umowy do analizy:", ["B2B", "Umowa o pracę"], horizontal=True)

    # --- Przygotowanie danych ---
    mask = (df["Technology"] == tech) & (df["Seniority"] == seniority)
    if location != "Cała Polska":
        mask = mask & (df["Location"] == location)
    
    filtered = df[mask].copy()

    # Wybór odpowiedniej kolumny z zarobkami (analizujemy górne widełki - Max)
    col_name = "Salary B2B Max" if contract_type == "B2B" else "Salary Employment Max"

    # Czyszczenie danych (usuwamy -1 i NaN)
    salaries = filtered[col_name]
    valid_salaries = salaries[(salaries > 0) & (salaries.notna())]

    st.divider()

    # --- Wyświetlanie wyników ---
    if valid_salaries.empty:
        st.warning("Niestety, brak wystarczających danych dla wybranych kryteriów.")
        return

    # Obliczenia statystyczne
    median_val = int(valid_salaries.median())
    q25_val = int(valid_salaries.quantile(0.25))
    q75_val = int(valid_salaries.quantile(0.75))
    count_val = len(valid_salaries)

    # Wyświetlenie metryk
    st.subheader(f"Wyniki dla: {tech} / {seniority}")
    st.caption(f"Analiza na podstawie {count_val} ofert.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Dolny kwartyl (25%)", value=f"{q25_val:,}".replace(",", " "))
        st.caption("Początek stawek rynkowych")
    with m2:
        st.metric(label="Mediana (Środek)", value=f"{median_val:,}".replace(",", " "), delta_color="off")
        st.caption("Najczęstsza stawka rynkowa")
    with m3:
        st.metric(label="Górny kwartyl (75%)", value=f"{q75_val:,}".replace(",", " "))
        st.caption("Oferty powyżej średniej")

    # Opcjonalnie: Prosty wykres rozkładu (histogram)
    st.write("")
    st.write("Rozkład stawek w ofertach:")
    try:
        # Używamy prostego bar_chart, sortując dane dla lepszej czytelności
        st.bar_chart(valid_salaries.value_counts().sort_index(), color="#4CAF50")
    except Exception:
        pass

def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "Strona główna"
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 1

    with st.sidebar:
        st.markdown("### Nawigacja")
        st.button(
            "Strona główna",
            key="home_button",
            on_click=lambda: _set_page("Strona główna"),
            use_container_width=True,
        )
        # --- NOWY PRZYCISK ---
        st.button(
            "🧮 Kalkulator Rynkowy",
            key="calc_button",
            on_click=lambda: _set_page("Kalkulator"),
            use_container_width=True,
        )
        # ---------------------
        st.button(
            "Wykres zarobków wg technologii",
            key="chart_button",
            on_click=lambda: _set_page("Wykres zarobków wg technologii"),
            use_container_width=True,
        )
        # ... (reszta przycisków)
        st.button(
            "EDA - Eksploracyjna Analiza Danych",
            key="eda_button",
            on_click=lambda: _set_page("EDA"),
            use_container_width=True,
        )

    try:
        df = load_data()
    except FileNotFoundError:
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", DATA_FILENAME)
        )
        st.error(
            "Nie znaleziono pliku z danymi. Upewnij się, że plik znajduje się w folderze 'data'.\n"
            f"Ścieżka sprawdzana: {data_path}"
        )
        return
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}")
        return

    current = _current_page()
    if current == "Strona główna":
        render_home(df)
    elif current == "Kalkulator":      # <--- NOWY WARUNEK
        render_calculator(df)          # <--- WYWOŁANIE FUNKCJI
    elif current == "Wykres zarobków wg technologii":
        render_salary_chart(df)
    elif current == "EDA":
        render_eda(df)
    else:
        st.error("Nieznana strona.")


if __name__ == "__main__":
    main()

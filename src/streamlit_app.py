import streamlit as st
import pandas as pd
import os
import numpy as np

# Ustaw szeroki układ strony, aby tabela mogła zająć maksymalną szerokość
st.set_page_config(page_title="Job Predictor", page_icon="📊", layout="wide")


def main():
    # nie ustawiamy tutaj tytułu globalnie, tylko per-strona niżej

    if "page" not in st.session_state:
        st.session_state["page"] = "Strona główna"
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 1

    def set_page(name):
        st.session_state["page"] = name
        st.session_state["page_number"] = 1

    current_page = st.session_state.get("page", "Strona główna")

    # ---------- SIDEBAR (NAWIGACJA) ----------
    with st.sidebar:
        st.markdown("### Nawigacja")
        st.button("Strona główna", key="home_button", on_click=lambda: set_page("Strona główna"), use_container_width=True)
        st.button("Wykres zarobków wg technologii", key="chart_button", on_click=lambda: set_page("Wykres zarobków wg technologii"), use_container_width=True)
        st.button("EDA - Eksploracyjna Analiza Danych", key="eda_button", on_click=lambda: set_page("EDA"), use_container_width=True)

    # ---------- GŁÓWNA TREŚĆ ----------
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'SalaryDataFinall.csv'))

    try:
        df = pd.read_csv(data_path, sep=';')
    except FileNotFoundError:
        st.error("Nie znaleziono pliku z danymi. Upewnij się, że plik znajduje się w folderze 'data'.\nŚcieżka sprawdzana: %s" % data_path)
        return
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}")
        return

    df.columns = df.columns.str.replace('\ufeff', '')
    num_cols = [c for c in df.columns if any(substr in c for substr in ['Salary', 'Company Size'])]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # ---------- STRONA GŁÓWNA ----------
    if current_page == "Strona główna":
        st.title("📊 Job Predictor")

        st.subheader("O zbiorze danych")
        st.markdown("""
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
        """)

        # --- FILTRY NAD TABELĄ ---
        st.subheader("Filtry ofert pracy")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            tech_filter = st.selectbox(
                "Technologia:",
                options=["Wszystkie"] + sorted(df["Technology"].dropna().unique().tolist())
            )
        with col_f2:
            loc_filter = st.selectbox(
                "Lokalizacja:",
                options=["Wszystkie"] + sorted(df["Location"].dropna().unique().tolist())
            )
        with col_f3:
            sen_filter = st.selectbox(
                "Poziom zaawansowania:",
                options=["Wszystkie"] + sorted(df["Seniority"].dropna().unique().tolist())
            )

        filtered = df.copy()
        if tech_filter != "Wszystkie":
            filtered = filtered[filtered["Technology"] == tech_filter]
        if loc_filter != "Wszystkie":
            filtered = filtered[filtered["Location"] == loc_filter]
        if sen_filter != "Wszystkie":
            filtered = filtered[filtered["Seniority"] == sen_filter]

        # licznik rekordów po filtrach
        st.caption(f"Liczba ofert po zastosowaniu filtrów: {len(filtered)}")
        # Opcja pokazywania pełnego ID
        show_full_id = st.checkbox("Pokaż pełne ID", value=False)
        # Tabela z paginacją na przefiltrowanych danych
        page_number = st.session_state.get("page_number", 1)
        rows_per_page = 20
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        df_view = filtered.copy()
        # Skracanie ID jeśli wyłączone pełne
        if not show_full_id and 'ID' in df_view.columns:
            df_view['ID'] = df_view['ID'].astype(str).apply(lambda x: (x[:8] + '…') if len(x) > 9 else x)
        for col in df_view.select_dtypes(include=[np.number]).columns:
            df_view[col] = df_view[col].apply(lambda x: '-1' if pd.isna(x) or int(x) == -1 else f"{int(x):,}")

        slice_df = df_view.iloc[start_idx:end_idx]
        # Dynamiczna wysokość aby wszystkie rekordy (do 20) były w pełni widoczne bez scrolla
        visible_rows = len(slice_df)
        row_height = 34  # przybliżona wysokość jednego wiersza w px
        header_extra = 70  # miejsce na nagłówek i marginesy
        dynamic_height = visible_rows * row_height + header_extra
        try:
            st.dataframe(slice_df, use_container_width=True, hide_index=True, height=dynamic_height)
        except TypeError:
            st.dataframe(slice_df, use_container_width=True, height=dynamic_height)

        # Szersze odstępy: "Następna" maksymalnie po prawej stronie
        col_prev, col_spacer, col_next = st.columns([2, 14, 2])
        with col_prev:
            if st.button("Poprzednia", key="prev") and page_number > 1:
                st.session_state["page_number"] = page_number - 1
        with col_next:
            if st.button("Następna", key="next") and end_idx < len(filtered):
                st.session_state["page_number"] = page_number + 1

    # ---------- WYKRES ZAROBKÓW WG TECHNOLOGII ----------
    elif current_page == "Wykres zarobków wg technologii":
        st.title("📈 Wykres zarobków juniorów wg technologii")

        st.markdown("""
        Wykres przedstawia średnie maksymalne wynagrodzenie dla juniorów w zależności od technologii.
        Możesz wybrać, czy chcesz zobaczyć dane łączne, dla umów o pracę, czy kontraktów B2B.
        """)

        try:
            contract_type = st.radio("Wybierz typ umowy:", ("Łącznie", "Umowa o pracę", "B2B"))
            seniority_level = st.selectbox("Wybierz poziom zaawansowania:", ("Wszystkie", "junior", "mid", "senior", "expert"))

            if seniority_level != "Wszystkie":
                filtered_df = df[df["Seniority"] == seniority_level].copy()
            else:
                filtered_df = df.copy()

            # Funkcja pomocnicza: średnia ignorując wartości -1 (ale nie usuwając wierszy)
            def mean_excluding_minus_one(series):
                vals = series[series != -1].dropna()
                if len(vals) == 0:
                    return np.nan
                return vals.mean()

            if contract_type == "Umowa o pracę":
                avg_salaries = filtered_df.groupby("Technology")["Salary Employment Max"].apply(mean_excluding_minus_one)
            elif contract_type == "B2B":
                avg_salaries = filtered_df.groupby("Technology")["Salary B2B Max"].apply(mean_excluding_minus_one)
            else:  # Łącznie - traktujemy każdą nie- -1 wartość z dwóch kolumn jako osobny rekord
                stacked = pd.DataFrame({
                    'Technology': np.concatenate([filtered_df['Technology'].values, filtered_df['Technology'].values]),
                    'Salary': np.concatenate([filtered_df['Salary Employment Max'].values, filtered_df['Salary B2B Max'].values])
                })
                stacked = stacked[stacked['Salary'] != -1]
                avg_salaries = stacked.groupby('Technology')['Salary'].mean()

            # Usuń NaN przed rysowaniem i konwertuj do int
            avg_salaries = avg_salaries.dropna().sort_values(ascending=False)
            avg_salaries = avg_salaries.astype(int)

            st.bar_chart(avg_salaries)
        except Exception as e:
            st.error(f"Nie udało się wygenerować wykresu: {e}")

    # ---------- STRONA EDA ----------
    elif current_page == "EDA":
        st.title("Eksploracyjna Analiza Danych (EDA)")

        st.markdown("""
        Eksploracyjna analiza danych (EDA) pozwala zrozumieć rozkład wynagrodzeń
        w zależności od wielkości firmy oraz poziomu zaawansowania.

        Poniżej prezentujemy dwa wybrane diagramy przygotowane wcześniej:
        - pensje juniorów względem wielkości firmy,
        - zarobki względem poziomu zaawansowania.
        """)

        img1_path = os.path.join(os.path.dirname(__file__), 'company_size_vs_junior_salary.png')
        img2_path = os.path.join(os.path.dirname(__file__), 'salary_by_seniority.png')

        if os.path.exists(img1_path):
            st.image(img1_path, caption="Pensje juniorów względem wielkości firmy", use_column_width=True)
        else:
            st.info("Brak pliku company_size_vs_junior_salary.png w katalogu src.")

        if os.path.exists(img2_path):
            st.image(img2_path, caption="Zarobki względem zaawansowania", use_column_width=True)
        else:
            st.info("Brak pliku salary_by_seniority.png w katalogu src.")

        st.markdown("""
        Pierwszy wykres pokazuje, jak średnia pensja juniorów zmienia się wraz z wielkością firmy
        (od mikro firm po bardzo duże organizacje).

        Drugi wykres porównuje rozkład wynagrodzeń między poziomami junior, mid, senior i expert,
        co pomaga zobaczyć, jak rośnie wynagrodzenie wraz z doświadczeniem.
        """)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import os

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "Strona główna"

    def set_page(name):
        st.session_state["page"] = name

    current_page = st.session_state.get("page", "Strona główna")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("📊 Nawigacja")

        st.button("🏠 Strona główna", key="home_button", on_click=lambda: set_page("Strona główna"), use_container_width=True)
        st.button("📈 Wykres zarobków wg technologii", key="chart_button", on_click=lambda: set_page("Wykres zarobków wg technologii"), use_container_width=True)

    # ---------- GŁÓWNA TREŚĆ ----------
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'SalaryDataFinall.csv'))

    try:
        df = pd.read_csv(data_path, sep=';')
    except FileNotFoundError:
        st.error("Nie znaleziono pliku z danymi. Upewnij się, że plik znajduje się w folderze 'data'.")
        return
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
        return

    if current_page == "Strona główna":
        st.title("📊 Job Predictor")

        st.markdown("""
        ### O zbiorze danych
        Zbiór danych przedstawia oferty pracy dla programistów w Polsce od września 2023 do lipca 2024. Kluczowe informacje:
        - **ID**: Unikalny identyfikator oferty.
        - **Firma**: Nazwa firmy rekrutującej.
        - **Wielkość firmy**: Liczba pracowników.
        - **Lokalizacja**: Miasto, w którym znajduje się praca.
        - **Technologia**: Wymagany język programowania lub stos technologiczny.
        - **Doświadczenie**: Poziom doświadczenia (np. junior, mid, senior).
        - **Wynagrodzenie (Umowa o pracę)**: Zakres wynagrodzenia dla tradycyjnych umów.
        - **Wynagrodzenie (B2B)**: Zakres wynagrodzenia dla kontraktów B2B.
        
        Zbiór danych dostarcza wglądu w polski rynek pracy IT, uwzględniając rozkład ofert według miast, technologii, poziomu doświadczenia i wynagrodzenia.
        """)

        page_number = st.session_state.get("page_number", 1)
        rows_per_page = 10
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if x != -1 else "-1")

        st.dataframe(df.iloc[start_idx:end_idx].style.hide(axis='index'), use_container_width=True)

        col1, col2, col3 = st.columns([2, 6, 2])
        with col1:
            if st.button("⬅️ Poprzednia strona") and page_number > 1:
                st.session_state["page_number"] = page_number - 1
        with col3:
            if st.button("Następna strona ➡️") and end_idx < len(df):
                st.session_state["page_number"] = page_number + 1

    elif current_page == "Wykres zarobków wg technologii":
        st.title("📈 Wykres zarobków juniorów wg technologii")

        st.markdown("""
        ### Opis wykresu
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

            if contract_type == "Umowa o pracę":
                salary_column = "Salary Employment Max"
                filtered_df[salary_column] = filtered_df[salary_column].apply(lambda x: x if x != -1 else 0)
                avg_salaries = filtered_df.groupby("Technology")[salary_column].mean()
            elif contract_type == "B2B":
                salary_column = "Salary B2B Max"
                filtered_df[salary_column] = filtered_df[salary_column].apply(lambda x: x if x != -1 else 0)
                avg_salaries = filtered_df.groupby("Technology")[salary_column].mean()
            else:  # Łącznie
                filtered_df["Salary Employment Max"] = filtered_df["Salary Employment Max"].apply(lambda x: x if x != -1 else None)
                filtered_df["Salary B2B Max"] = filtered_df["Salary B2B Max"].apply(lambda x: x if x != -1 else None)
                avg_salaries = filtered_df.groupby("Technology")[
                    ["Salary Employment Max", "Salary B2B Max"]].mean().mean(axis=1)

            st.bar_chart(avg_salaries)
        except Exception as e:
            st.error(f"Nie udało się wygenerować wykresu: {e}")

if __name__ == "__main__":
    main()

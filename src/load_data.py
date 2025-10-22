import os

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



def plot_salary_by_seniority(df):
    """Wykres średnich zarobków według porządkowej zmiennej Seniority."""
    salary_cols = ['Salary Employment Min', 'Salary Employment Max', 'Salary B2B Min', 'Salary B2B Max']
    df = df.copy()

    # Oblicz średnią tylko z dodatnich wartości
    df['Avg_Salary'] = df[salary_cols].where(df[salary_cols] > 0).mean(axis=1)

    # Normalizacja i ustawienie porządku poziomów Seniority
    df['Seniority'] = df.get('Seniority', '').astype(str).str.lower().str.strip()
    seniority_order = ['junior', 'mid', 'senior', 'expert']
    df['Seniority'] = pd.Categorical(df['Seniority'], categories=seniority_order, ordered=True)

    df = df.dropna(subset=['Avg_Salary', 'Seniority'])
    if df.empty:
        print("Brak danych do wykresu `salary_by_seniority`.")
        return

    # Liczby rekordów dla każdej kategorii (używamy observed=False aby uniknąć FutureWarning)
    counts = df.groupby('Seniority', observed=False)['Avg_Salary'].count().reindex(seniority_order).fillna(0).astype(int)
    print("Liczba rekordów wg Seniority:")
    for lvl, cnt in counts.items():
        print(f"  {lvl}: {int(cnt)}")

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df,
        x='Seniority',
        y='Avg_Salary',
        order=seniority_order,
        hue='Seniority',    # eliminuje FutureWarning przy użyciu palette
        palette='Set2',
        dodge=False,
        showfliers=True
    )
    # Usuń legendę powstałą przez ustawienie hue
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # dodaj zapas na osi Y (+15000)
    try:
        max_val = float(df['Avg_Salary'].max())
        ax.set_ylim(0, max_val + 15000)
    except Exception:
        pass

    # Dodaj adnotacje z liczebnością nad każdym boxem (tylko dla kategorii z danymi)
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.03
    for i, lvl in enumerate(seniority_order):
        cnt = int(counts.get(lvl, 0))
        if cnt == 0:
            continue
        mask = df['Seniority'] == lvl
        if mask.any():
            local_max = df.loc[mask, 'Avg_Salary'].max()
            if pd.isna(local_max):
                ypos = y_min + y_offset
            else:
                ypos = local_max + y_offset
        else:
            ypos = y_min + y_offset
        ax.text(i, ypos, f'n={cnt}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    plt.title('Zarobki względem zaawansowania', fontsize=14, fontweight='bold')
    plt.xlabel('Zawansowanie', fontsize=12)
    plt.ylabel('Średnia pensja', fontsize=12)
    plt.tight_layout()
    plt.savefig('salary_by_seniority.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_company_size_vs_junior_salary(df):
    """Bar chart: średnie pensje juniorów wg wielkości firmy."""
    df = df.copy()
    df['Seniority'] = df.get('Seniority', '').astype(str).str.lower().str.strip()
    df_junior = df[df['Seniority'] == 'junior'].copy()

    salary_cols = ['Salary Employment Min', 'Salary Employment Max', 'Salary B2B Min', 'Salary B2B Max']
    df_junior['Avg_Salary'] = df_junior[salary_cols].where(df_junior[salary_cols] > 0).mean(axis=1)
    df_junior = df_junior.dropna(subset=['Avg_Salary', 'Company Size'])

    def categorize_company_size(size):
        if pd.isna(size) or size == -1:
            return None
        try:
            size = float(size)
        except Exception:
            return None
        if size < 50:
            return 'Micro (< 50)'
        elif size < 250:
            return 'Mała (50-250)'
        elif size < 1000:
            return 'Średnia (250-1K)'
        elif size < 10000:
            return 'Duża (1K-10K)'
        else:
            return 'Bardzo Duża (> 10K)'

    df_junior['Company_Size_Category'] = df_junior['Company Size'].apply(categorize_company_size)
    df_junior = df_junior.dropna(subset=['Company_Size_Category'])

    size_order = ['Micro (< 50)', 'Mała (50-250)', 'Średnia (250-1K)', 'Duża (1K-10K)', 'Bardzo Duża (> 10K)']
    grouped = df_junior.groupby('Company_Size_Category').agg(
        Avg_Salary=('Avg_Salary', 'mean'),
        Count=('Avg_Salary', 'count')
    ).reindex(size_order)

    # Zapełnij brakujące kategorie Count=0, Avg_Salary=NaN
    grouped['Count'] = grouped['Count'].fillna(0).astype(int)
    grouped['Avg_Salary'] = grouped['Avg_Salary'].where(grouped['Count'] > 0, np.nan)

    if grouped['Count'].sum() == 0:
        print("Brak danych do wykresu `company_size_vs_junior_salary`.")
        return

    # Wypisz liczby rekordów dla wszystkich kategorii (nawet jeśli 0)
    print("Liczba rekordów wg wielkości firmy (junior):")
    for cat, row in grouped.iterrows():
        print(f"  {cat}: {int(row['Count'])}")

    # Rysuj tylko kategorie z Count > 0
    grouped_nonzero = grouped[grouped['Count'] > 0]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(grouped_nonzero)), grouped_nonzero['Avg_Salary'].values,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4'][:len(grouped_nonzero)],
                   edgecolor='black', linewidth=1.2, alpha=0.9)

    # adnotacje z wartością i liczebnością
    for idx, (val, cnt) in enumerate(zip(grouped_nonzero['Avg_Salary'].values, grouped_nonzero['Count'].values)):
        plt.text(idx, val + max(1, val * 0.02), f'${val:,.0f}\n(n={int(cnt)})',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.xticks(range(len(grouped_nonzero)), grouped_nonzero.index, fontsize=11, rotation=0)
    plt.title('Pensje juniorów względem wielkości firmy', fontsize=14, fontweight='bold')
    plt.xlabel('Wielkość firmy', fontsize=12, fontweight='bold')
    plt.ylabel('Średnia pensja ($)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # dodaj zapas na osi Y (+15000)
    try:
        max_val = float(grouped_nonzero['Avg_Salary'].max())
        plt.ylim(0, max_val + 15000)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig('company_size_vs_junior_salary.png', dpi=300, bbox_inches='tight')
    plt.show()




path_to_data = '../data/SalaryDataFinall.csv'
df = load_data(path_to_data)
num_cols, cat_cols = identify_data_types(df)
check_missing_values(df)
plot_salary_by_seniority(df)
plot_company_size_vs_junior_salary(df)

# policz i wypisz łączną liczbę juniorów (bez błędu gdy brak kolumny)
if 'Seniority' in df.columns:
    ser = df['Seniority'].fillna('').astype(str).str.lower().str.strip()
    total_juniors = int(ser.eq('junior').sum())
else:
    total_juniors = 0
print(f"Liczba juniorów: {total_juniors}")



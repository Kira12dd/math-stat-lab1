import numpy as np
import stat_calc as sc
from plots import show_statistical_plots

# Вхідні дані: 12 варіант (кількість пошкоджених зерен у пробі)
raw_data = [
    4, 3, 4, 3, 4, 5, 3, 2, 3, 5, 4,
    3, 5, 3, 2, 3, 4, 5, 3, 3, 3, 3,
    4, 3, 3, 3, 3, 6, 3, 3, 2, 3, 3,
    2, 6, 1, 1, 4, 5, 3, 4, 1, 3, 3,
    4, 3, 2, 3, 2, 4, 3, 3, 5, 5, 5,
    2, 3, 3, 2, 4, 5, 2, 5, 4, 3, 2,
    5, 3, 3, 4, 4, 5, 3, 7, 3
]

def main():
    # Розрахунок характеристик для вихідної вибірки (обсяг n = 75)
    orig = sc.analyze_sample(raw_data)
    st = orig['stat']
    
    # Пункт 1-2. Вихідна вибірка, варіаційний ряд, межі та розмах R
    print("1. Початкова вибірка (n = 75):")
    print(raw_data)
    
    print("\n2. Варіаційний ряд:")
    print(orig['var_series'])
    print(f"\nОбсяг n = {orig['n']}, x_min = {orig['x_min']}, x_max = {orig['x_max']}, Розмах R = {orig['R']}")
    
    # Пункт 3. Таблиця статистичного розподілу частот і накопичених частот
    freq_table = sc.get_freq_table(orig['x_vals'], orig['n_vals'], orig['w_vals'])
    print("\n3. Статистичний розподіл частот і відносних частот:")
    print(freq_table.to_string(index=False, col_space=10, justify='center'))
    
    # Пункт 5. Аналітичний запис емпіричної функції розподілу за проміжками
    print("\n5. Аналітичний запис емпіричної функції розподілу F*(x):")
    print(sc.get_empir(orig['x_vals'], orig['w_vals']))
    
    # Пункт 6. Числові характеристики вихідної вибірки
    print("\n6. Числові характеристики вихідної вибірки:")
    print(f" - Середнє x_bar: {st['x_bar']:.4f}")
    print(f" - Дисперсія d_bar (спосіб 1): {st['d_bar_1']:.4f}, (спосіб 2): {st['d_bar_2']:.4f}")
    print(f" - Середнє квадр. відхилення sigma: {st['sigma_bar']:.4f}")
    print(f" - Виправлена дисперсія s^2: {st['s2']:.4f}, Стандарт s: {st['s']:.4f}")
    print(f" - Мода Mo: {st['mo']}, Медіана Me: {st['me']}")
    
    # Пункт 7. Квартилі та інтерквартильний розмах
    q1 = sc.get_quantile(orig['var_series'], 25)
    q3 = sc.get_quantile(orig['var_series'], 75)
    print(f"\n7. Квартилі: Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {q3 - q1:.2f}")
    
    # Пункт 8. Перерахунок вибірки через квантилі до обсягу m = 100
    y_series, y_st = sc.resample_and_analyze(orig['var_series'], target_size=100)
    print(f"\n8. Перерахована вибірка y_1, ..., y_100 (обсяг m = {len(y_series)}):")
    for i in range(0, len(y_series), 10):
        print("   " + ", ".join(f"{val:.2f}" for val in y_series[i:i+10]))
    
    # Пункт 9. Порівняльна таблиця вихідної та перерахованої вибірок
    comp_df = sc.get_comparison_table(st, orig['n'], y_st, len(y_series))
    print("\n9. Порівняльна таблиця вихідної та перерахованої вибірок:")
    print(comp_df.to_string(index=False))
    
    # Пункт 4-5. Побудова графіків полігонів та F*(x)
    show_statistical_plots(orig['x_vals'], orig['n_vals'], orig['w_vals'])

if __name__ == "__main__":
    main()
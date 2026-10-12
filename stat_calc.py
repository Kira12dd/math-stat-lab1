from collections import Counter
import numpy as np
import pandas as pd

def analyze_sample(raw_data):
    """Аналіз вибірки: Побудова варіаційного ряду, частоти та числові характеристики."""
    n = len(raw_data)
    var_series = sorted(raw_data)
    
    counts = Counter(var_series)
    x_vals = sorted(counts.keys())
    n_vals = [counts[x] for x in x_vals]
    w_vals = [count / n for count in n_vals]
    
    stat = calc_stat(raw_data, x_vals, n_vals, n)
    
    return {
        'n': n,
        'var_series': var_series,
        'x_min': var_series[0],
        'x_max': var_series[-1],
        'R': var_series[-1] - var_series[0],
        'x_vals': x_vals,
        'n_vals': n_vals,
        'w_vals': w_vals,
        'stat': stat
    }

def calc_stat(data, x_vals, n_vals, n):
    """Обчислення числових характеристик."""
    x_bar = np.mean(data)
    d_bar_1 = np.sum(n_vals * (x_vals - x_bar)**2) / n
    d_bar_2 = np.sum(n_vals * (np.array(x_vals)**2)) / n - (x_bar**2)
    sigma_bar = np.sqrt(d_bar_1)
    
    return {
        'x_bar': x_bar,
        'd_bar_1': d_bar_1,
        'd_bar_2': d_bar_2,
        'sigma_bar': sigma_bar,
        's2': (n / (n - 1)) * d_bar_1,
        's': np.sqrt((n / (n - 1)) * d_bar_1),
        'mo': x_vals[np.argmax(n_vals)],
        'me': sorted(data)[n // 2] if n % 2 != 0 else (sorted(data)[n // 2 - 1] + sorted(data)[n // 2]) / 2
    }

def get_quantile(data_sorted, m, K=100):
    """Обчислення квантиля."""
    n = len(data_sorted)
    L_m = (n + 1) * m / K
    int_L = int(np.floor(L_m))
    frac_L = L_m - int_L
    
    if int_L >= n: return float(data_sorted[-1])
    if int_L < 1:  return float(data_sorted[0])
    return data_sorted[int_L - 1] + frac_L * (data_sorted[int_L] - data_sorted[int_L - 1])

def resample_and_analyze(var_series, target_size=100):
    """Перерахунок до m=100 та обчислення характеристик нової вибірки."""
    y_series = [get_quantile(var_series, j, target_size) for j in range(1, target_size + 1)]
    counts = Counter(y_series)
    x_vals = sorted(counts.keys())
    n_vals = [counts[x] for x in x_vals]
    
    stat = calc_stat(y_series, x_vals, n_vals, target_size)
    return y_series, stat

def get_empir(x_vals, w_vals):
    """Формування кускового запису емпіричної функції F*(x)."""
    cum_w = np.cumsum(w_vals)
    lines = []
    lines.append(f"         0,         при x <= {x_vals[0]}")
    
    for i in range(len(x_vals) - 1):
        lines.append(f"         {cum_w[i]:.4f},    при {x_vals[i]} < x <= {x_vals[i+1]}")
        
    lines.append(f"         1,         при x > {x_vals[-1]}")
    return "F*(x) = {\n" + "\n".join(lines)

def get_freq_table(x_vals, n_vals, w_vals):
    """Створює таблицю розподілу частот і відносних частот."""
    return pd.DataFrame(
        {
            "x_i": x_vals,
            "n_i": n_vals,
            "w_i": [round(w, 4) for w in w_vals],
            #"F*(x)": [round(f, 4) for f in np.cumsum(w_vals)],
        }
    )

def get_comparison_table(st, n, y_st, n_new=100):
  """Створює підсумкову таблицю порівняння характеристик."""
  return pd.DataFrame({
      "Характеристика": [
          "Вибіркове середнє",
          "Вибіркова дисперсія",
          "Середнє квадр. відхилення",
          "Медіана Me",
          "Мода Mo",
      ],
      f"Вихідна (n={n})": [
          f"{st['x_bar']:.4f}",
          f"{st['d_bar_1']:.4f}",
          f"{st['sigma_bar']:.4f}",
          f"{st['me']:.4f}",
          f"{st['mo']:.4f}",
      ],
      f"Перерахована (n={n_new})": [
          f"{y_st['x_bar']:.4f}",
          f"{y_st['d_bar_1']:.4f}",
          f"{y_st['sigma_bar']:.4f}",
          f"{y_st['me']:.4f}",
          f"{y_st['mo']:.4f}",
      ],
      "Різниця, %": [
          f"{abs(y_st['x_bar'] - st['x_bar']) / st['x_bar'] * 100:.2f}%",
          f"{abs(y_st['d_bar_1'] - st['d_bar_1']) / st['d_bar_1'] * 100:.2f}%",
          f"{abs(y_st['sigma_bar'] - st['sigma_bar']) / st['sigma_bar'] * 100:.2f}%",
          f"{abs(y_st['me'] - st['me']) / st['me'] * 100:.2f}%",
          f"{abs(y_st['mo'] - st['mo']) / st['mo'] * 100:.2f}%",
      ]
  })
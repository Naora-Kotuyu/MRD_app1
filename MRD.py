#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from scipy.stats import pearsonr, linregress,skew, kurtosis,shapiro
plt.rcParams['font.family'] = "MS Gothic"


def get_user_input():
    root = tk.Tk()
    root.title("データ入力")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 数値データの数を入力するテキストボックス
    num_numeric_label = ttk.Label(main_frame, text="数値データの数を入力してください:")
    num_numeric_label.grid(row=0, column=0, padx=5, pady=5)

    num_numeric_entry = ttk.Entry(main_frame)
    num_numeric_entry.grid(row=0, column=1, padx=5, pady=5)

    # 数値データの名前を入力するテキストボックス
    numeric_names_label = ttk.Label(main_frame, text="数値データの名前をスペースで区切って入力してください:")
    numeric_names_label.grid(row=1, column=0, padx=5, pady=5)

    numeric_names_entry = ttk.Entry(main_frame)
    numeric_names_entry.grid(row=1, column=1, padx=5, pady=5)

    # カテゴリカルデータの数を入力するテキストボックス
    num_categorical_label = ttk.Label(main_frame, text="カテゴリカルデータの数を入力してください:")
    num_categorical_label.grid(row=2, column=0, padx=5, pady=5)

    num_categorical_entry = ttk.Entry(main_frame)
    num_categorical_entry.grid(row=2, column=1, padx=5, pady=5)

    # カテゴリカルデータの名前を入力するテキストボックス
    categorical_names_label = ttk.Label(main_frame, text="カテゴリカルデータの名前をスペースで区切って入力してください:")
    categorical_names_label.grid(row=3, column=0, padx=5, pady=5)

    categorical_names_entry = ttk.Entry(main_frame)
    categorical_names_entry.grid(row=3, column=1, padx=5, pady=5)

    result_numeric = None
    result_categorical = None

    def open_numeric_input_window():
        nonlocal result_numeric
        num_numeric_features = int(num_numeric_entry.get())
        numeric_feature_names = numeric_names_entry.get().split()

        numeric_input_window = tk.Toplevel(root)
        numeric_input_window.title("数値データの入力")

        numeric_input_frame = ttk.Frame(numeric_input_window)
        numeric_input_frame.pack(fill=tk.BOTH, expand=True)

        numeric_data = {}

        for feature_name in numeric_feature_names:
            ttk.Label(numeric_input_frame, text=f"{feature_name} の値をスペースで区切って入力してください:").pack(padx=5, pady=5)
            numeric_entry = ttk.Entry(numeric_input_frame)
            numeric_entry.pack(padx=5, pady=5)
            numeric_data[feature_name] = numeric_entry

        def submit_numeric_data():
            nonlocal result_numeric
            data = {}

            for feature_name, entry in numeric_data.items():
                try:
                    values = [float(value) for value in entry.get().split()]
                except ValueError:
                    messagebox.showerror("エラー", f"{feature_name} の入力が無効です。有効な数値を入力してください。")
                    return

                data[feature_name] = values

            result_numeric = pd.DataFrame(data)
            numeric_input_window.destroy()

        submit_button = ttk.Button(numeric_input_frame, text="Submit", command=submit_numeric_data)
        submit_button.pack(pady=10)

    def open_categorical_input_window():
        nonlocal result_categorical
        num_categorical_features = int(num_categorical_entry.get())
        categorical_feature_names = categorical_names_entry.get().split()

        categorical_input_window = tk.Toplevel(root)
        categorical_input_window.title("カテゴリカルデータの入力")

        categorical_input_frame = ttk.Frame(categorical_input_window)
        categorical_input_frame.pack(fill=tk.BOTH, expand=True)

        categorical_data = {}

        for feature_name in categorical_feature_names:
            ttk.Label(categorical_input_frame, text=f"{feature_name} の値をスペースで区切って入力してください:").pack(padx=5, pady=5)
            categorical_entry = ttk.Entry(categorical_input_frame)
            categorical_entry.pack(padx=5, pady=5)
            categorical_data[feature_name] = categorical_entry

        def submit_categorical_data():
            nonlocal result_categorical
            data = {}

            for feature_name, entry in categorical_data.items():
                values = entry.get().split()
                data[feature_name] = values

            result_categorical = pd.DataFrame(data)
            categorical_input_window.destroy()

        submit_button = ttk.Button(categorical_input_frame, text="Submit", command=submit_categorical_data)
        submit_button.pack(pady=10)

    submit_button_numeric = ttk.Button(main_frame, text="数値データ入力", command=open_numeric_input_window)
    submit_button_numeric.grid(row=4, column=0, columnspan=2, pady=10)

    submit_button_categorical = ttk.Button(main_frame, text="カテゴリカルデータ入力", command=open_categorical_input_window)
    submit_button_categorical.grid(row=5, column=0, columnspan=2, pady=10)

    complete_button = ttk.Button(main_frame, text="完了", command=root.destroy)
    complete_button.grid(row=6, column=0, columnspan=2, pady=10)

    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return result_numeric, result_categorical

df_numeric, df_categorical = get_user_input()
if df_numeric is not None:
    print("数値データ:")
    print(df_numeric)

if df_categorical is not None:
    print("\nカテゴリカルデータ:")
    print(df_categorical)

# 数値データとカテゴリカルデータを結合して1つのデータフレームにまとめる
df = pd.concat([df_numeric, df_categorical], axis=1)


def show_plot_in_new_window(fig, feature_i, feature_j):
    new_window = tk.Toplevel()
    new_window.title("拡大表示")

    main_frame = ttk.Frame(new_window)
    main_frame.pack(fill=tk.BOTH, expand=True)
    fig.set_size_inches(9, 9)  # グラフの図面のサイズを変更

    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT)
    
    if feature_i != feature_j and pd.api.types.is_numeric_dtype(df[feature_i]) and not pd.api.types.is_numeric_dtype(df[feature_j]):
        add_category_stats(main_frame, df, feature_i, feature_j)
    if feature_i != feature_j and pd.api.types.is_numeric_dtype(df[feature_j]) and not pd.api.types.is_numeric_dtype(df[feature_i]):
        add_category_stats(main_frame, df, feature_j, feature_i)
    
    # ヒストグラムの場合、統計情報を計算して表示
    if feature_i == feature_j and pd.api.types.is_numeric_dtype(df[feature_i]):
        min_val = df[feature_i].min()
        min_label = ttk.Label(main_frame, text=f"最小値: {min_val:.4f}")
        min_label.pack(side=tk.TOP)
        
        max_val = df[feature_i].max()
        max_label = ttk.Label(main_frame, text=f"最大値: {max_val:.4f}")
        max_label.pack(side=tk.TOP)
        
        mean_val = df[feature_i].mean()
        average_label = ttk.Label(main_frame, text=f"平均値: {mean_val:.4f}")
        average_label.pack()
        
        std_dev_val = df[feature_i].std()
        std_dev_label = ttk.Label(main_frame, text=f"標準偏差: {std_dev_val:.4f}")
        std_dev_label.pack(side=tk.TOP)
        
        # ひがみ（歪度）を計算
        skewness = skew(df[feature_i])
        skew_label = ttk.Label(main_frame, text=f"歪度: {skewness:.4f}")
        skew_label.pack(side=tk.TOP)

        # とがり（尖度）を計算
        kurt = kurtosis(df[feature_i])
        kurt_label = ttk.Label(main_frame, text=f"尖度: {kurt:.4f}")
        kurt_label.pack(side=tk.TOP)
        
        # 正規性の検定（シャピロ-ウィルク検定）
        stat, p_value = shapiro(df[feature_i])
        normality_label = ttk.Label(main_frame, text=f"正規性検定 p-value: {p_value:.4f}")
        normality_label.pack(side=tk.TOP)

        
    # 散布図の場合のみ相関係数を計算して表示
    if feature_i != feature_j and pd.api.types.is_numeric_dtype(df[feature_i]) and pd.api.types.is_numeric_dtype(df[feature_j]):
        correlation_coefficient = calculate_correlation(df, feature_i, feature_j)
        correlation_label = ttk.Label(main_frame, text=f"相関係数: {correlation_coefficient:.4f}")
        correlation_label.pack(side=tk.TOP)

    # 散布図の場合のみ軸の説明を表示
        axis_label = ttk.Label(main_frame, text=f"横軸: {feature_i}\n縦軸: {feature_j}")
        axis_label.pack(side=tk.TOP)

    # 散布図の場合のみ平均値と標準偏差を計算して表示
        average_label = ttk.Label(main_frame, text=f"横軸平均値: {df[feature_i].mean():.4f}\n縦軸平均値: {df[feature_j].mean():.4f}")
        average_label.pack()

        std_dev_label = ttk.Label(main_frame, text=f"横軸標準偏差: {df[feature_i].std():.4f}\n縦軸標準偏差: {df[feature_j].std():.4f}")
        std_dev_label.pack(side=tk.TOP)

    # 散布図の場合のみ回帰直線の情報を取得して表示
        slope, intercept, r_value, p_value, std_err = linregress(df[feature_i], df[feature_j])
        regression_label = ttk.Label(main_frame, text=f"回帰直線: y = {slope:.4f}x + {intercept:.4f}")
        regression_label.pack(side=tk.TOP)

    # 散布図の場合のみ回帰直線を追加
        add_regression_line(fig, df[feature_i], df[feature_j])
        
def add_regression_line(fig, x, y):
    # 回帰直線の追加
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax = fig.gca()
    ax.plot(x, intercept + slope * x, color='red', label='Regression Line')
    ax.legend()

def calculate_correlation(dataframe, feature_i, feature_j):
    # 相関係数を計算
    correlation_coefficient, _ = pearsonr(dataframe[feature_i], dataframe[feature_j])
    return correlation_coefficient
    
def add_category_stats(frame, dataframe, feature_i, feature_j):
    unique_categories = dataframe[feature_j].unique()

    # スクロールバーを持つフレームを作成
    stats_frame_scrollable = tk.Frame(frame)
    stats_frame_scrollable.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

    # フレーム内にスクロールバーを追加
    stats_frame_canvas = tk.Canvas(stats_frame_scrollable)
    stats_frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    stats_frame_scrollbar = ttk.Scrollbar(stats_frame_scrollable, orient=tk.VERTICAL, command=stats_frame_canvas.yview)
    stats_frame_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    stats_frame_canvas.configure(yscrollcommand=stats_frame_scrollbar.set)

    # スクロール可能なフレームを作成
    stats_frame = ttk.Frame(stats_frame_canvas)
    stats_frame_canvas.create_window((0, 0), window=stats_frame, anchor="nw")

    for category in unique_categories:
        subset = dataframe[dataframe[feature_j] == category][feature_i]

        # フレーム内に統計情報を表示
        stats_subframe = ttk.Frame(stats_frame)
        stats_subframe.pack(side=tk.TOP, padx=5, pady=5)


        # 最小値
        min_val = subset.min()
        min_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 最小値: {min_val:.4f}")
        min_label.pack(side=tk.TOP)

        # 最大値
        max_val = subset.max()
        max_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 最大値: {max_val:.4f}")
        max_label.pack(side=tk.TOP)

        # 平均値
        mean_val = subset.mean()
        mean_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 平均値: {mean_val:.4f}")
        mean_label.pack(side=tk.TOP)

        # 標準偏差
        std_dev_val = subset.std()
        std_dev_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 標準偏差: {std_dev_val:.4f}")
        std_dev_label.pack(side=tk.TOP)

        # ひがみ（歪度）
        skewness = skew(subset)
        skew_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 歪度: {skewness:.4f}")
        skew_label.pack(side=tk.TOP)

        # とがり（尖度）
        kurt = kurtosis(subset)
        kurt_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 尖度: {kurt:.4f}")
        kurt_label.pack(side=tk.TOP)

        # 正規性の検定（シャピロ-ウィルク検定）
        stat, p_value = shapiro(subset)
        normality_label = ttk.Label(stats_frame, text=f"{feature_i} ({feature_j}={category}) 正規性検定 p-value: {p_value:.4f}")
        normality_label.pack(side=tk.TOP)
        
    # スクロールバーの設定
    stats_frame.bind("<Configure>", lambda e: stats_frame_canvas.configure(scrollregion=stats_frame_canvas.bbox("all")))

    
def plot_individual_matrices(dataframe):
    features = dataframe.columns
    num_features = len(features)

    root = tk.Tk()
    root.title("プロット")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    x_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    y_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    for i in range(num_features):
        for j in range(num_features):
            fig = Figure(figsize=(3, 3))
            ax = fig.add_subplot(111)

            if i == j:
                if pd.api.types.is_numeric_dtype(dataframe[features[i]]):
                    ax.hist(dataframe[features[i]], bins=10)
                    ax.set_xlabel(features[i])
                    ax.set_ylabel('Frequency')
                elif pd.api.types.is_string_dtype(dataframe[features[i]]):
                    count_table = dataframe[features[i]].value_counts(normalize=True)
                    count_table.plot(kind='bar', ax=ax)
                    ax.set_xlabel(features[i])
                    ax.set_ylabel('Percentage')
                    ax.set_ylim(0, 1)
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            else:
                if pd.api.types.is_numeric_dtype(dataframe[features[i]]) and pd.api.types.is_numeric_dtype(dataframe[features[j]]):
                    ax.scatter(dataframe[features[i]], dataframe[features[j]])
                    ax.set_xlabel(features[i])
                    ax.set_ylabel(features[j])

                elif pd.api.types.is_numeric_dtype(dataframe[features[i]]) and not pd.api.types.is_numeric_dtype(dataframe[features[j]]):
                    ax.set_frame_on(False)
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)

                    unique_categories = dataframe[features[j]].unique()

                    for idx, category in enumerate(unique_categories, start=1):
                        ax = fig.add_subplot(len(unique_categories), 1, idx)
                        subset = dataframe[dataframe[features[j]] == category][features[i]]
                        ax.hist(subset, bins=10, alpha=0.5, label=f"{features[j]}={category}")
                        ax.set_xlabel(features[i])
                        ax.set_ylabel('Frequency')
                        ax.legend()

                elif not pd.api.types.is_numeric_dtype(dataframe[features[i]]) and not pd.api.types.is_numeric_dtype(dataframe[features[j]]):
                    count_table = pd.crosstab(dataframe[features[i]], dataframe[features[j]])
                    count_table.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_xlabel(features[i])
                    ax.set_ylabel('Count')  # y軸のラベルを「Count」に設定
                    ax.set_ylim(0, count_table.sum(axis=1).max())  # y軸の範囲を0から各行の合計値の最大値に設定
                    ax.legend(title=features[j])
                    for p in ax.patches:
                        width = p.get_width()
                        height = p.get_height()
                        x, y = p.get_xy() 
                        ax.annotate(f'{height} ({height / count_table.sum(axis=1)[int(x)]:.2%})',
                                    (x + width/2, y + height/2), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

                elif not pd.api.types.is_numeric_dtype(dataframe[features[i]]) and pd.api.types.is_numeric_dtype(dataframe[features[j]]):
                    ax.set_frame_on(False)
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)

                    unique_categories = dataframe[features[i]].unique()

                    for idx, category in enumerate(unique_categories, start=1):
                        ax = fig.add_subplot(len(unique_categories), 1, idx)
                        subset = dataframe[dataframe[features[i]] == category][features[j]]
                        ax.hist(subset, bins=10, alpha=0.5, label=f"{features[i]}={category}")
                        ax.set_xlabel(features[j])
                        ax.set_ylabel('Frequency')
                        ax.legend()

            ax.set_title("")
            plot_widget = FigureCanvasTkAgg(fig, master=frame)
            plot_widget.draw()
            plot_widget.get_tk_widget().grid(row=i, column=j)

            plot_widget.get_tk_widget().bind("<Button-3>",
                                            lambda event, fig=fig, feature_i=features[i], feature_j=features[j]: show_plot_in_new_window(
                                                fig, feature_i, feature_j))

            fig.tight_layout()

    root.mainloop()

# 関数を呼び出してプロットを生成します
plot_individual_matrices(df)


# データの入力は質的データ(非数的データ)と量的データ(数的データ)でわけて入力する必要があります。

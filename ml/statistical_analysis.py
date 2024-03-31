import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Read the data
data = pd.read_csv("./dataset/features.csv")

# preprocessing the nan values
data = data.dropna()

def perform_tests(data, column1, column2, output_file_name):

    with open(output_file_name, 'w') as f:
        # Redirecting print statements to the file
        original_stdout = sys.stdout
        sys.stdout = f

        # Correlation analysis
        print("Correlation matrix:")
        correlation_matrix = data.corr()
        print(correlation_matrix)
        print("---------------------------------", file=f)
        
        # Perform a t-test for gender differences in the 'word_count' column
        male_word_count = data[data['gender'] == 'male'][column1]
        female_word_count = data[data['gender'] == 'female'][column1]
        t_stat, p_value = stats.ttest_ind(male_word_count, female_word_count)
        print("T-test:")
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)

        # Perform a chi-square test
        print("Chi-square test:")
        contingency_table = pd.crosstab(data[column1], data[column2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-square statistic: {chi2}")
        print(f"p-value: {p}")
        print("---------------------------------", file=f)

        # Perform a one-way ANOVA
        print("One-way ANOVA:")
        grouped_data = data.groupby(column1)
        grouped_data = grouped_data[column2].apply(list)
        f_stat, p_value = stats.f_oneway(*grouped_data)
        print(f"F-statistic: {f_stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)

        # Perform a Mann-Whitney U test
        print("Mann-Whitney U test:")
        u_stat, p_value = stats.mannwhitneyu(data[column1], data[column2])
        print(f"U-statistic: {u_stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)

        # Perform a Kruskal-Wallis H test
        print("Kruskal-Wallis H test:")
        h_stat, p_value = stats.kruskal(*grouped_data)
        print(f"H-statistic: {h_stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)

        # Perform a paired t-test
        print("Paired t-test:")
        t_stat, p_value = stats.ttest_rel(data[column1], data[column2])
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)

        # Perform a sign test
        print("Sign test:")
        stat, p_value = stats.wilcoxon(data[column1], data[column2])
        print(f"Test statistic: {stat}")
        print(f"p-value: {p_value}")
        print("---------------------------------", file=f)
    sys.stdout = original_stdout


label_encoder = LabelEncoder()
data['gender_encoded'] = label_encoder.fit_transform(data['gender'])

# Now, you can use the perform_tests method to analyze the gender differences
perform_tests(data, 'word_count', 'gender_encoded', 'statistical_analysis.txt')

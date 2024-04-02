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

    with open(output_file_name, 'a') as f:
        # Redirecting print statements to the file
        original_stdout = sys.stdout
        sys.stdout = f

        # Feature name
        print("Feature:" + column1)
        print("---------------------------------", file=f)

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
        print("---------------------------------", file=f)
        print("---------------------------------", file=f)
        print("---------------------------------", file=f)
        print("---------------------------------", file=f)
        print("---------------------------------", file=f)
        
    sys.stdout = original_stdout


label_encoder = LabelEncoder()
data['gender_encoded'] = label_encoder.fit_transform(data['gender'])


column_names = [
    'word_count', 'average_sentence_length',
    'lexical_diversity', 'count_foreign_words', 'count_wh_words',
    'sentiment_score', 'subjectivity_score', 'count_slang', 'count_VERB',
    'count_NOUN', 'count_ADJ', 'count_ADV', 'count_PRON', 'count_CCONJ', 'count_ADP', 'count_DET',
    'count_NUM', 'count_X', 'count_SYM', 'count_PART', 'count_SPACE', 
    'count_SCONJ', 'count_PROPN', 'count_AUX', 'pos_distribution_ADJ', 'pos_distribution_ADP',
    'pos_distribution_ADV', 'pos_distribution_AUX', 'pos_distribution_DET',
    'pos_distribution_NOUN', 'pos_distribution_NUM', 'pos_distribution_PART',
    'pos_distribution_PRON', 'pos_distribution_PROPN', 'pos_distribution_SCONJ',
    'pos_distribution_SYM', 'pos_distribution_VERB', 'pos_statistics_mean_count',
    'pos_statistics_median_count', 'pos_statistics_std_deviation', 'pos_statistics_max_count',
    'pos_statistics_min_count'
]

for column_name in column_names:
    print(column_name)
    perform_tests(data, column_name, 'gender_encoded', 'statistical_analysis.txt')


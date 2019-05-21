import pandas as pd
import matplotlib.pyplot as plt
import squarify
from pycountry import languages

def codes_distribution(df):
    df['code_group'] = (df['code'] // 100).values.astype(str)
    df['code_group'] = df['code_group'] + 'xx'
    group = pd.DataFrame(df.groupby(df['code_group']).size(), columns=['count'])
    group.plot.pie(y='count', autopct='%1.1f%%')
    plt.show()


def lang_by_countries(df):
    group = pd.DataFrame(df.groupby(['language', 'region']).size().groupby('language').size().nlargest(20), columns=['regions_count'])
    group['language_name'] = group.apply(lambda x: languages.get(alpha_2=x.name).name, axis=1)
    group['label'] = group['language_name'] + ' (' + group['regions_count'].values.astype(str) + ')'
    colors = ["#006e90", "#F18F01", "#ADCAD6", "#99C24D", "#41BBD9", "#8ACEA9", "#A8E3FF", "#CEE5FF", "#B3B5E0", "#BA9FC4"]
    squarify.plot(sizes=group['regions_count'], label=group['label'], color=colors)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('./output/data.csv', header=0)
    lang_by_countries(df)
    codes_distribution(df)

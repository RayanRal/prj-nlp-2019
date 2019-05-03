## Курсовий проект
Для свого курсового проекту:

- поділіть свої дані на тренувальні та тестувальні
- побудуйте бейзлайн (правила чи простий класифікатор)
- поміряйте якість роботи системи раніше побудованими метриками

Код для курсового проекту повинен бути у вашому репозиторії. У директорії students/ в теці з вашим іменем збережіть файл з посиланням на код бейзлайну. Опишіть ваші результати.

## Tesufr
Multilingual Text Summarizing Framework - https://github.com/serge-sotnyk/tesufr

На 13-квітня-2019 зроблено каркас фреймворку, який:

1. Дозволяє використовувати різні ядра для анотації (виділення ключових слів та саммарі) текстів на різних мовах.
2. Додано baseline ядро, яке майже незалежне від мови (індо-європейської групи), у якості саммарі віддає перщі N речень текста, а у якості ключових слів - M самих росповсюджених в тексті термів.
3. Розроблено систему підтримки абстрактних корпусів, та зроблено два провайдери для реальних корпусів - BBC News (саммарі) та Krapivin2009 (саммарі та ключові слова). Фреймворк, якщо для цих корпусів ще нема локальних файлів, загружає їх з мого Google Drive, та надалі віддає у стандартизованному виді. Далі будуть підтримані ще корпуси для саммарі, та ключових слів.
4. Перенесено розроблену раніше систему оцінки якості на базі метрик ROUGE/BLEU для саммарі, та F1 для ключових слів.

### Типовий код:

Анотація тексту

```
from nltk.corpus import udhr
...
processor = Processor()
text = udhr.raw('Italian-Latin1')
text_process_params = TextProcessParams(SummarySize.new_relative(0.1), keywords_number=10)
document = processor.process_text(text, text_process_params)
```

* Processor - клас, який керує обробкою тексту
* text_process_params - параметри обробки тексту. Зараз це кількість ключових слів, та речень в саммарі.
* Document - клас, який є контейнером для переробленого тексту, та має поля для колекції знайдених ключових слів, самарі, Named Entities, оригінального тексту, деталі обробки, ворнінги та помилки, які можуть бути на етапі обробки.

Для того, щоб створити нове ядро, треба реалізувати абстрактний клас CoreBase - https://github.com/serge-sotnyk/tesufr/blob/789b37fc5947bdcd5379aa8187eeba52dfe8d656/tesufr/cores/core_base.py

Прикладом такої реалізації є поки єдиний клас FallbackCore - https://github.com/serge-sotnyk/tesufr/blob/789b37fc5947bdcd5379aa8187eeba52dfe8d656/tesufr/cores/fallback_core.py. Але далі буде...

Перевірка якості анотації по двом підтриманим корпусам (https://github.com/serge-sotnyk/tesufr/blob/789b37fc5947bdcd5379aa8187eeba52dfe8d656/evaluate_with_corpora.py):

```
processor = Processor()
corpus = BbcNewsProvider()
metrics = evaluate_processor_on_corpus(processor, corpus, "BBC News")
print(metrics)

corpus = Krapivin2009Provider()
metrics = evaluate_processor_on_corpus(processor, corpus, "Krapivin2009")
print(metrics)
```

Результати першого запуску з бейзлайн-ядром:

```
C:\Users\ssotn\Anaconda3\envs\tesurf\python.exe D:/git-nlp/tesufr/evaluate_with_corpora.py
WARNING:root:Could not import signal.SIGPIPE (this is expected on Windows machines)
Check metrics for BBC News
100%|██████████| 445/445 [00:40<00:00,  9.50it/s]
{'rouge_1': 0.5489569227407518, 'rouge_2': 0.44733345610434844, 'rouge_3': 0.4047945705861156, 'rouge_4': 0.3695602606075601, 'bleu': 41.611864189349895}
Downloading ['1zRP0sKH0tn3P2hWRyE2E3yXjjbkYLHtR'] into D:\git-nlp\tesufr\corpora_data\krapivin2009.zip... Done.
  0%|          | 0/461 [00:00<?, ?it/s]Check metrics for Krapivin2009
100%|██████████| 461/461 [28:08<00:00,  3.17s/it]
{'precision': 0.4030932949690004, 'recall': 0.14229306838749253, 'f1': 0.20524058808381612, 'rouge_1': 0.024238965325505984, 'rouge_2': 0.008135379808691693, 'rouge_3': 0.003927411070078904, 'rouge_4': 0.0025979649815830105, 'bleu': 0.5387344660957267}
```

### Плани на майбутнє
1. Викладка як пакет Pypi (щоб можна було встановлювати за допомогою pip)
2. Нові ядра
3. Нові корпуси
4. REST API

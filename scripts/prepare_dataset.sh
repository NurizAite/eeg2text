#!/bin/bash

echo "This script constructs .pickle files from .mat files from ZuCo dataset."
echo "Processing only available tasks: task1-SR (v1.0) and task2-NR-2.0 (v2.0)."
echo "This script also generates ternary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

# Обработка task1-SR (ZuCo v1.0)
#python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task2-NR
#python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task3-TSR
# Обработка task2-NR-2.0 (ZuCo v2.0)
#python3 ./util/construct_dataset_mat_to_pickle_v2.py -t task2-NR-2.0

# Генерация файлов настроений (нужны для task1-SR и дальнейшего анализа)
#python3 ./util/get_sentiment_labels.py
#python3 ./util/get_SST_ternary_dataset.py
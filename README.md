В файле ModelQA.py содержиться интерфейся для создания чат бота
В файле Trainning.py код для дообучения модели и преобразования входных данных (Обучался в Google Colab на GPU)
Файл Annoy_index необходимый для работы дообученной модели, необходимо скачать, так как он превышает 100мб (git lfs не помог)
(https://drive.google.com/file/d/1gufxOUqmlSdKEqWSMUzflGKvj8OTdo2e/view?usp=sharing)
## Описание проекта
Данный проект представляет собой чат-бота для ответа на вопросы пользователей. Бот использует модель SentenceTransformer и построенный на ее основе индекс Annoy для быстрого поиска наиболее подходящего ответа на заданный вопрос.

## Используемая модель
В данном проекте используется модель SentenceTransformer с названием "all-MiniLM-L6-v2". Эта модель обучена на большом корпусе текстов и способна выделять семантические признаки из текстов, что позволяет использовать ее для задачи поиска наиболее подходящего ответа на заданный вопрос.

## Используемая метрика
Для измерения сходства между вопросами и ответами используется метрика cosine similarity, которая позволяет измерять косинус угла между векторами, полученными из текстов с помощью модели SentenceTransformer.

## Функция сравнения
Для поиска наиболее подходящего ответа на заданный вопрос используется индекс Annoy, который строится на основе функции сравнения angular distance. Эта функция сравнения позволяет измерять угол между векторами и определять, насколько они близки друг к другу по смыслу.

Обучение проводилось в Google Colab https://colab.research.google.com/drive/1K3xRY9uQWrjuLdSUF-B6nONKMar_m5YE?authuser=0#scrollTo=VHAXGAbxYqEr
Модель на HuggingFace в открытом доступе (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

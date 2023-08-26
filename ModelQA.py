# -*- coding: utf-8 -*-руг
"""
Установить!
pip install transformer
pip install annoy
pip install transformers[torch]
pip install -U sentence-transformers
"""
from annoy import AnnoyIndex
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle as pkl

data = pd.read_csv('QAdata')
question = list(data.QUESTION)
answer = list(data.ANSWER)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with open('model.pkl', 'rb') as f:
    embeddings = pkl.load(f)
annoy_index = AnnoyIndex(len(embeddings[1]), 'angular')
annoy_index.load('Annoy_index')

def question_response(embeddings):
    top_k_hits = 5
    inp_question = input("Введите ваш вопрос: ")
    question_embedding = model.encode(inp_question)
    corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
    hits = []
    for id, score in zip(corpus_ids, scores):
        hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})
    hits_above_threshold = [hit for hit in hits if hit['score'] > 0.8]
    if hits_above_threshold:
        print("Возможные ответы на ваш вопрос:")
        for hit in hits_above_threshold:
            print("\t{:.3f}\t{}".format(hit['score'], question[hit['corpus_id']]))
        correct_hits = util.semantic_search(question_embedding, embeddings, top_k=top_k_hits)[0]
        correct_hits_ids = list([hit['corpus_id'] for hit in correct_hits])
        print("Наиболее подходящий ответ на ваш вопрос:")
        print(answer[correct_hits_ids[0]])
        with open('user_questions.txt', 'a') as f:
            f.write(f"Вопрос пользователя: {inp_question}\n")
        return answer[correct_hits_ids[0]]
    else:
        print("К сожалению, мы не поняли ваш вопрос. Возможно, вы имели в виду один из следующих вопросов:")
        for hit in hits[:3]:
            print("\t{}".format(question[hit['corpus_id']]))
        return "Не понятен вопрос"

if __name__ == '__main__':
    while True:
        try:
            question_response(embeddings)
        except KeyboardInterrupt:
            print('Пока!')
            break
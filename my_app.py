# -*- coding: utf-8 -*-

from flask import Flask
from flask import url_for, render_template, request, redirect
from pymystem3 import Mystem
import nltk
from nltk.collocations import *
import re
import json
import requests
import time

app = Flask(__name__)
m = Mystem()

users_data = dict()
pets_data = dict()

pets_data['cats'] = 0
pets_data['dogs'] = 0

data = None
sentences = None
bigrams = None


# 1. Flask Intro

@app.route('/')
def form():
    if request.args:
        name = request.args['name']
        pets = request.args['pets']

        if name in users_data:
            users_data[name] += 1
        else:
            users_data[name] = 1

        pets_data[pets] += 1

        return redirect(url_for('result'))

    return render_template('form.html')


@app.route('/result')
def result():
    num_for_cats = pets_data['cats']
    num_for_dogs = pets_data['dogs']

    return render_template('result.html', for_cats=num_for_cats,
                           for_dogs=num_for_dogs, users=users_data)


# 2. Pymystem3

def count_verbs(text):
    num_of_words = 0
    num_of_verbs = 0

    num_of_trans = 0
    num_of_imtrans = 0

    num_of_perfect = 0
    num_of_imperfect = 0

    dict_of_verbs = dict()

    ana = m.analyze(text)
    for i in ana:
        if i['text'].strip() and 'analysis' in i and i['analysis']:
            num_of_words += 1

            gr = i['analysis'][0]['gr']
            if gr.split('=')[0].split(',')[0] == 'V':
                num_of_verbs += 1

                lex = i['analysis'][0]['lex']
                if lex in dict_of_verbs:
                    dict_of_verbs[lex] += 1
                else:
                    dict_of_verbs[lex] = 1

                if 'сов' in gr:
                    num_of_perfect += 1
                else:
                    num_of_imperfect += 1

                if 'пе' in gr:
                    num_of_trans += 1
                else:
                    num_of_imtrans += 1

    part_of_verbs = round(num_of_verbs / num_of_words, 2)

    result_dict = {'Всего глаголов': num_of_verbs, 'Доля глаголов в тексте':
                   part_of_verbs, 'Переходных глаголов': num_of_trans,
                   'Непереходных глаголов': num_of_imtrans,
                   'Глаголов совершенного вида': num_of_perfect,
                   'Глаголов несовершенного вида': num_of_imperfect}

    return result_dict, dict_of_verbs


@app.route('/pymystem', methods=['get', 'post'])
def index_pymystem():
    if request.form:
        text = request.form['text']
        result_dict, verb_lex = count_verbs(text)
        verb_lex = sorted(verb_lex.items(), key=lambda x: x[1], reverse=True)
        return render_template('pymystem.html', input=text,
                               verb_info=result_dict, lex=verb_lex)
    return render_template('pymystem.html')

# 4. VK API

def vk_api(method, **kwargs):
    api_request = 'https://api.vk.com/method/' + method + '?'
    api_request += '&'.join(['{}={}'.format(key, kwargs[key])
                            for key in kwargs])
    return json.loads(requests.get(api_request).text)


def get_users(group_id):
    group_users = []

    page = 0
    limit = 1000
    offset = 0

    response = vk_api('groups.getMembers', group_id=group_id, offset=offset,
                      count=limit)
    num_users = response['response']['count']

    if num_users <= limit:
        group_users = response['response']['users']
    else:
        while num_users > offset + limit:
            offset = page * limit
            print(offset)

            response = vk_api('groups.getMembers', group_id=group_id,
                              offset=offset, count=limit)

            time.sleep(1)

            group_users.extend(response['response']['users'])

            page += 1

    return (num_users, group_users)


@app.route('/VK', methods=['get', 'post'])
def index():
    if request.form:
        group1 = request.form['group1']
        group2 = request.form['group2']

        group1_users = get_users(group1)
        group2_users = get_users(group2)

        num1_users = group1_users[0]
        num2_users = group2_users[0]

        num_intersected = len(set.intersection(set(group1_users[1]),
                                               set(group2_users[1])))

        return render_template('vk.html', group1=group1, group2=group2,
                               num1_users=num1_users, num2_users=num2_users,
                               num_intersected=num_intersected)

    return render_template('vk.html')


# 7. NLTK

def remove_timing_and_nums(sentences):
    def good_string(string):
        return '-->' not in string

    return list(filter(lambda x: good_string(x) and
                       not x.isdigit(), sentences))


def sentences_with_word(word, sentences):
    regexp = re.compile("(^|\s){}($|\W)".format(word.lower()))
    return list(filter(lambda x: regexp.search(x.lower()), sentences))


def sentences_with_stem(stem, sentences):
    stem = stem.lower()
    return list(filter(lambda x: stem in x.lower(), sentences))


def bigrams_with_word(word, bigrams):
    def good_bigram(bigram):
        return word.lower() == bigram[0].lower() or \
               word.lower() == bigram[1].lower()

    tmp = list(filter(lambda x: good_bigram(x[0]), bigrams))

    answer = []
    for item in tmp:
        answer.append((' '.join(item[0]), item[1]))

    return answer


@app.route('/nltk', methods=['get', 'post'])
def index_nltk():
    if request.form:
        word = request.form['text']

        sents_word = sentences_with_word(word, sentences)
        sents_stem = sentences_with_stem(word, sentences)
        bgrms_word = bigrams_with_word(word, bigrams)

        return render_template('nltk.html', input=word,
                               sents_word=sents_word, sents_stem=sents_stem,
                               bgrms_word=bgrms_word)

    return render_template('nltk.html')


if __name__ == '__main__':
    fname = "data/friends.srt"
    with open(fname, 'rb') as f:
        data = f.read()

    data = data.decode('cp1251')
    tmp = remove_timing_and_nums(data.split('\r\n'))
    data = ' '.join(tmp)

    sentences = nltk.sent_tokenize(data)

    finder = BigramCollocationFinder.from_words(nltk.word_tokenize(data))
    bigrams = list(finder.ngram_fd.items())
    bigrams = sorted(bigrams, key=lambda x: x[1], reverse=True)
    
if __name__ == '__main__':
	app.run(debug=True)

'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:

@ author:
Zewei Chu (zeweichu@gmail.com)

@modified:
8/4/2020

'''
import os
import json
import pandas as pd
import pickle
import numpy as np

import code

TPS_DIR = './data'
TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')
OUT_DIR = "./data"

users_id = []
items_id = []
ratings = []
reviews = []
np.random.seed(2020)

with open(TP_file) as f:
    for line in f:
        js=json.loads(line)
        if str(js['reviewerID'])=='unknown':
            print("unknown")
            continue
        if str(js['asin'])=='unknown':
            print("unknown2")
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']))
        items_id.append(str(js['asin']))
        ratings.append(str(js['overall']))

data = pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]


# convert user and item IDs to numeric values
rating_columns = ['user_id','item_id','ratings']

n_ratings = data.shape[0]
devtest = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
devtest_idx = np.zeros(n_ratings, dtype=bool)
devtest_idx[devtest] = True

devtest_data = data[devtest_idx]
train_data = data[~devtest_idx]

n_ratings = devtest_data.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

test_data = devtest_data[test_idx]
dev_data = devtest_data[~test_idx]
train_data[rating_columns].to_csv(os.path.join(OUT_DIR, 'train.csv'))
dev_data[rating_columns].to_csv(os.path.join(OUT_DIR, 'valid.csv'))
test_data[rating_columns].to_csv(os.path.join(OUT_DIR, 'test.csv'))

user_reviews={}
item_reviews={}

# user_id, item_id, rating, review
for i in train_data.values:
    if i[0] not in user_reviews:
        user_reviews[i[0]] = []

    user_reviews[i[0]].append(i[3].strip().replace("\t", " "))

    if i[1] not in item_reviews:
        item_reviews[i[1]] = []

    item_reviews[i[1]].append(i[3].strip().replace("\t", " "))


# code.interact(local=locals())
# print(item_reviews[11])

# with open(os.path.join(TPS_DIR, 'user_review'), 'w') as fout:


with open(os.path.join(OUT_DIR, 'user_review.tsv'), 'w') as fout:
    for uid, reviews in user_reviews.items():
        fout.write(uid + "\t" + "\t".join(reviews) + "\n")


with open(os.path.join(OUT_DIR, 'item_review.tsv'), 'w') as fout:
    for uid, reviews in item_reviews.items():
        fout.write(uid + "\t" + "\t".join(reviews) + "\n")


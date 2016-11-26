# coding: utf-8
"""
This py file will realize two recommendation systems(item-based/user-based)\
using datasets in our local directory.
Why don't we upload our datasets into GitHub? Because the size limitation for
a file in GitHub is only 100 size!
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import math


# Load data
def LoadData():
	print 'Loading data...'
	# movies
	dir = "/Users/apple/Desktop/Graduate/course/datamining/datamining-sets"
	#mnames = ['movieid', 'title', 'genres']
	movies = pd.read_csv(dir + '/movies.csv', engine='python')

	# ratings
	#rnames = ['userid', 'movieid', 'rating', 'timestamp']
	ratings = pd.read_csv(dir + '/ratings.csv', engine='python')
	ratings = ratings.drop('timestamp', axis=1)  # drop column of "timestamp"
	print 'Data loaded.'
	return movies, ratings


# Preprocess the datasets
def Preprocess(movies, ratings):
	print 'Data preprocessing...'
	# Convert the column of "genres" into dummies
	genre_iter = (set(x.split('|')) for x in movies.genres)
	genres = sorted(set.union(*genre_iter))  # get all genres
	dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
	# iterate "genres" each row, assign "1" to corresponding location
	for i, gen in enumerate(movies.genres):
		dummies.ix[i, gen.split('|')] = 1
	movies_dummies = movies.join(dummies.add_prefix('Genres_'))  # merged with movies

	data_merged = pd.merge(movies_dummies, ratings, on='movieid')  # merge two tables by 'movieid'
	data = data_merged.drop_duplicates()  # drop duplications
	print 'Preprocessing completed.'
	return data


# Create the item-user dict
def TransformData(data):
	print 'Transform data to item-user dict...'
	item_user = dict()  # create empty dictionary
	# iterate merged dataset, getting the item-user dict
	for i in range(len(data)):
		item_user.setdefault(data.ix[i]['movieid'], {})
		item_user[data.ix[i]['movieid']][data.ix[i]['userid']] = float(data.ix[i]['rating'])
		if i % 500 == 0:
			print('\tFinshed %s %' % round((i*100.0/len(data)), 2))
	print 'Item-user dict has done.'
	return item_user


# ============================== Algorithm ==============================
# Pearson coefficient
def SimPearson(prefs, p1, p2):
	si = {}
	for item in prefs[p1]:
		if item in prefs[p2]:
			si[item] = 1
	n = len(si)
	if n == 0:
		return -1

	sum1 = sum([prefs[p1][it] for it in si])
	sum2 = sum([prefs[p2][it] for it in si])

	sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
	sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

	pSum = sum([prefs[p1][it]*prefs[p2][it] for it in si])

	num = pSum - (sum1*sum2/n)
	den = math.sqrt((sum1Sq-pow(sum1, 2)/n)*(sum2Sq-pow(sum2, 2)/n))
	if den == 0:
		return 0
	r = num/den
	return r


# Matches
def Matches(prefs, person, similarity=SimPearson):
	scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
	scores.sort()  # sort
	scores.reverse()  # decreasing sort
	return scores


# Item similarity
def ItemSimilarity(train):
	print 'Calculating item similarities...'
	itemsim = {}
	n = 0  # counter
	# iterate datasets
	for item in train:
		n += 1
		if n % 500 == 0:
			print "\tFinished %s %" % (round((n*100.0/len(train)), 2))
		scores = Matches(train, item)
		itemsim[item] = scores
	print 'Item-sim has done.'
	return itemsim


if __name__ == "__main__":
	movies, ratings = LoadData()
	data = Preprocess(movies, ratings)
	item_user = TransformData(data)
	itemsim = ItemSimilarity(item_user)
	with open('/Users/apple/Desktop/Graduate/course/datamining/test.txt') as f:
		for item in itemsim:
			f.write(item)
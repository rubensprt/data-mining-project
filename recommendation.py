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
import random


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

# Load data test
def LoadDataTest():
	print 'Loading data...'
	# movies
	dir = "/Users/apple/Desktop/test"
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
	data = data_merged.drop_duplicates()
	data = data.dropna()  # drop duplications
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
		if i % 500000 == 0:
			print('\tFinshed %s%%' % round((i*100.0/len(data)), 2))
	print('\tFinshed 100%')
	print 'Item-user dict has done.'
	return item_user


# Transform datasets from item_user to user_item
def Transform(train):
	user_item = dict()
	for item in train:
		for user in train[item]:
			user_item.setdefault(user, {})
			user_item[user][item] = train[item][user]
	return user_item


def ItemSim(train):
	# Comatrix
	comatrix = dict()
	# the frequency of movies
	movie_num = dict()
	# iterate
	for movies in train.values():
		# calculate the comatrix
		for i in movies:
			# calculate the frequency
			movie_num.setdefault(i, 0)
			movie_num[i] += 1
			for j in movies:
				if j == i:
					continue
				comatrix.setdefault(i, {})
				comatrix[i].setdefault(j, 0)
				comatrix[i][j] += 1

	# calculate the similarity
	Sim = dict()
	for i, related_movies in comatrix.items():
		for j in related_movies:
			Sim.setdefault(i, {})
			Sim[i][j] = (comatrix[i][j] * 1.0)/math.sqrt(movie_num[i] * movie_num[j])

	# sort
	Sim_result = dict()
	for k, v in Sim.items():
		sim = sorted(v.iteritems(), key=lambda x: x[1], reverse=True)
		Sim_result[k] = sim
	return Sim_result


# Recommend(IBFC)
def Recommend_I(itemsim, train, user, k=100, n=10):
	userRatings = train[user]
	rank = dict()
	totalsim = dict()
	# Iterate the movies that a certain user has rated
	for (item, rating) in userRatings.items():
		# find the most xth similar movies in datasets
		for i in itemsim[item][:k]:
			item2 = i[0]
			similarity = i[1]
			if item2 in userRatings:
				continue
			if similarity == 0 or similarity == -1:
				continue
			rank.setdefault(item2, 0)
			rank[item2] += similarity * rating
			totalsim.setdefault(item2, 0)
			totalsim[item2] += similarity
	#rankings = [(round(v/totalsim[k],2), k) for k, v in rank.items()]
	rankings = [(round(v,2),k) for k, v in rank.items()]
	rankings.sort()
	rankings.reverse()
	return rankings[:n]

# Recommend(UBFC)
def Recommend_U(usersim, train, user, k=10, n=10):
	rank = dict()
	totalsim = dict()
	for (user2, sim) in usersim[user][:k]:
		if user2 == user:
			continue
		if sim == 0 or sim == -1:
			continue
		for item in train[user2]:
			if item not in train[user]:
				rank.setdefault(item, 0)
				rank[item] += sim * train[user2][item]
				totalsim.setdefault(item, 0)
				totalsim[item] += sim
	rankings = [(round(v/totalsim[k],2), k) for k, v in rank.items()]
	#rankings = [(round(v,2),k) for k, v in rank.items()]
	rankings.sort()
	rankings.reverse()
	return rankings[:n]



# ============================== Algorithm Evaluation ==============================
# Split the data
def SplitData(data, M, k, seed):
	test = dict()
	train = dict()
	random.seed(seed)
	for i in range(len(data)):
		if random.randint(0, M) == k:
			test.setdefault(data.ix[i]['userid'], {})
			test[data.ix[i]['userid']][data.ix[i]['movieid']] = float(data.ix[i]['rating'])
		else:
			train.setdefault(data.ix[i]['userid'], {})
			train[data.ix[i]['userid']][data.ix[i]['movieid']] = float(data.ix[i]['rating'])
	return train, test

# Split the data
def SplitData2(data, M, k, seed):
	test = dict()
	train = dict()
	random.seed(seed)
	for i in range(len(data)):
		if random.randint(0, M) == k:
			test.setdefault(data.ix[i]['movieid'], {})
			test[data.ix[i]['movieid']][data.ix[i]['userid']] = float(data.ix[i]['rating'])
		else:
			train.setdefault(data.ix[i]['movieid'], {})
			train[data.ix[i]['movieid']][data.ix[i]['userid']] = float(data.ix[i]['rating'])
	return train, test


# Recall
def Recall(recommend_result, test, user):
	# hit represents the number of predicted one in test
	hit = 0
	for item in recommend_result:
		if item[1] in test[user]:
			hit += 1

	all = len(test[user])
	return hit/(all*1.0)


# Precise
def Precise(recommend_result, test, user):
	hit = 0
	for item in recommend_result:
		if item[1] in test[user]:
			hit += 1

	all = len(recommend_result)
	return hit/(all*1.0)


# Coverage
def Coverage(recommend_result, movies_num):
	recommend_items = []
	for user in recommend_result:
		for item in recommend_result[user]:
			recommend_items.append(item[1])

	items = set(recommend_items)

	return len(items)/(movies_num * 1.0)


# =============================Get the result==========================
def GetAllRecommendations(Sim_result, train, k=10):
	recommend_result = dict()
	c = 0
	for user in train:
		recommend_result[user] = Recommend_I(Sim_result, train, user, k)
		if c%1000 == 0: print "%d / %d" % (c, len(train))
		c += 1

	return recommend_result

def TestRecommend(recommend_result, test, movies_num=10325):
	recall = []
	precise = []
	for user in recommend_result:
		if user in test:
			recall.append(Recall(recommend_result[user], test, user))
			precise.append(Precise(recommend_result[user], test, user))
		else:
			continue

	# get the average
	recall_r = sum(recall)/(len(recall)*1.0)
	precise_r = sum(precise)/(len(precise)*1.0)

	# get the coverage
	coverage_r = Coverage(recommend_result, movies_num)

	return round(recall_r, 4), round(precise_r, 4), round(coverage_r, 4)


if __name__ == "__main__":
	# movies, ratings = LoadData()
	movies, ratings = LoadDataTest()
	data = Preprocess(movies, ratings)
	#item_user = TransformData(data)
	#user_item = Transform(item_user)
	#itemsim = ItemSim(user_item)
	#print Recommend_I(itemsim, user_item, 1)

	# Test IBCF
	#train, test = SplitData(data, 8, 4, 123)
	#itemsim = ItemSim(train)
	#for k in [5, 10, 20, 40, 80, 160, 320]:
	#	recommend_result = GetAllRecommendations(itemsim, train, k)
	#	r, p, c = TestRecommend(recommend_result, test)
	#	print('k=%s: recall=%s, precise=%s, coverage=%s' % (k, r, p, c))

	#Test UBCF
	train, test = SplitData2(data, 8, 7, 123)
	usersim = ItemSim(train)
	for k in [5, 10, 20, 40, 80, 160, 320]:
		recommend_result = GetAllRecommendations(usersim, train, k)
		r, p, c = TestRecommend(recommend_result, test)
		print('k=%s: recall=%s, precise=%s, coverage=%s' % (k, r, p, c))

import pandas
import numpy

#Program acts as a simple recommender system
#culmination of the exploration i've been doing into SVD and recomender systems

#take in the movie data as a pd dataframe / delim is split / names is cols
dataMan = pandas.read_csv('/Users/ChristianWalsh/Downloads/ml-100k/u.data', delimiter='\t',names=['UserID', 'ItemID', 'Rating', 'Timestamp'])
#print(dataMan.head)


#creating my sparse "dense" matrix A 
#PIVOTED DATA 
userRatingPerItem = dataMan.pivot(index='UserID', columns='ItemID', values='Rating')
#print(userRatingPerItem.head())

#loop through the sparce matrix and fill it in with the mean of the column
for column in userRatingPerItem:
    mean = userRatingPerItem[column].mean() #taking the mean for each col
    userRatingPerItem[column] = userRatingPerItem[column].fillna(value=mean) #fillna - any empty only
#print(userRatingPerItem.head())

#a user matrix is now needed to calculate the "distance" or relativity between each user
#this method uses the Euclidean distance. Other methods like cosine similarity may also be used 

def distance(v1,v2):
    return numpy.sqrt(numpy.sum((v1-v2)**2))
u1 = userRatingPerItem.iloc[1]
u2 = userRatingPerItem.iloc[2]
distance(u1,u2)

#THIS PORTION WAS "BORROWED" from a github repo i studied
#This takes a while to run
# Beware, this kind of matrix is very time-consuming, and the one you are using is not optimized. 
# Consider using default dictionary if you have to use user similarity matrix.
userMatrix = []
for i, row in enumerate(userRatingPerItem.index):
    u1 = userRatingPerItem[row] # I believe it should be userRatingPerItem.loc[row] u2 has the same problem
    #symmetric matrix fill all values previously examined
    userDistanes = [entry[i] for entry in userMatrix] # I believe this one is incorrect, but I have not tested it using any data. 
    for j, row2 in enumerate(userRatingPerItem.index[i:]):
        u2 = userRatingPerItem[row2]
        dist = distance(u1,u2)
        userDistanes.append(dist)
    userMatrix.append(userDistanes)
userSimilarities = pandas.DataFrame(userMatrix)
#print(userSimilarities.head())

#now all that is left to do is make a recommend function to rec movies to user
def recommendMovies(user, userSimilarities, userRatingPerItem, dataMan, nUsers=20, nItems=10):
    #n is num of similar uses i want to gen data for
    #User similarities are offset by 1 so remove the current user
    topNSimilarUsers = userSimilarities[user-1].drop(user-1).sort_values().index[:nUsers]
    #offset fix here
    topNSimilarUsers = [i+1 for i in topNSimilarUsers]
    alreadyWatched = set(dataMan[dataMan.UserID == 0].ItemID.unique()) # I think, maybe, this UserID should be user instead of 0. 
                                                                       # Because you are trying to find all movies that have been watched by the current user instead of user 0. 
    unseen = set(dataMan.ItemID.unique()) - alreadyWatched
    projectedReviews = userRatingPerItem[userRatingPerItem.index.isin(topNSimilarUsers)].mean()[list(unseen)].sort_values(ascending=False)
    return projectedReviews[:nItems]

#example of some recs
print(recommendMovies(1, userSimilarities, userRatingPerItem, dataMan)) # this is recommendation only for one user, if you could, can you generate the recommendations for all users?

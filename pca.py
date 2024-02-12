#from ctypes import sizeof # what... is this?

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # you don't need the whole package
import matplotlib.pyplot as plt
import numpy as np


def calculate_biases(ratings_matrix):
    global_mean = ratings_matrix.stack().mean()
    user_bias = ratings_matrix.mean(axis=1) - global_mean # axis is like do you want mean on row or mean on column, like the direction of x, y, z
    item_bias = ratings_matrix.mean(axis=0) - global_mean # it is like squeezing them
    return global_mean, user_bias, item_bias

# this is actually slow, not just becasue of the nested for-loop, but because of this: ratings_matrix.loc[user, item]
# I'm curious if you can find better solusion 
def impute_ratings(ratings_matrix, global_mean, user_bias, item_bias):
    imputed_matrix = ratings_matrix.copy() # important, try figure out why
    for user in ratings_matrix.index:
        for item in ratings_matrix.columns:
            if pd.isnull(ratings_matrix.loc[user, item]):
                imputed_matrix.loc[user, item] = global_mean + user_bias[user] + item_bias[item]
    return imputed_matrix


#performing PCA on the data set
def main():
    
    # UHHHH get the file to be the data set yo
    # Using pandas to make my data structure of the set
    data = pandas.read_csv('/Users/ChristianWalsh/Downloads/ml-100k/u.data', sep='\t', header=None, names=['UserID', 'ItemID', 'Rating', 'Timestamp'])
    

    #centering and scaling the data
    #scaled_data = preprocessing.scale(data.T) #passing the transpose
    # Are you trying to scale features instead of samples? For PCA, typically, you scale samples, not features (columns).
    #StandardScaler().fit_transform(data.T)

    # I am guessing you want to do this but don't know how. 
    # Pivot the table instead of transposing the table? 
    ratings_matrix = data.pivot_table(index='UserID', columns='ItemID', values='Rating')

    
    # Befure you use PCA, you need to make sure the matrix is dense.
    # So you need to use imputation
    # Or you need to use this "from sklearn.decomposition import SparsePCA"
    # I challenge you to figure out what this is and what is so different. 

    # Calculate biases and thne impute missing ratings
    global_mean, user_bias, item_bias = calculate_biases(ratings_matrix)
    imputed_ratings_matrix = impute_ratings(ratings_matrix, global_mean, user_bias, item_bias)
    pca = PCA(n_components=2) # you might want to specify the number of components you want

    #scaling is really trivial in our case, but to follow the mandane rules...
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_ratings_matrix.fillna(0))  # Impute with 0 (should be minimal if any)
    pca.fit(scaled_data) #does the math -loading scores, variation
    pca_data = pca.transform(scaled_data) #generate coords for the pca graph based on data


    # I am not a master of visualization, so I will leave this part to you for now. If I have more time, I will start correcting the part. 
    #start with a scree plot 
    perc_variation = np.round(pca.explained_variance_ratio_* 100, decimals=1)#percent variation of each component
    labels = ['PC' + str(x) for x in range(1, len(perc_variation)+1)] #labels for plot

    #make a bar plot with matplotlib
    plt.bar(x=range(1,len(perc_variation)+1), height=perc_variation, tick_label=labels)
    plt.ylabel('Percent Explained Variance')
    plt.xlabel('Principle Component')
    plt.title('Scree Plot')
    #plt.show()
    
    #put the new coords into a matrix
    pca_df = pandas.DataFrame(pca_data, index=[ ItemID, Rating], columns=labels)
    # hmmmm...  I think you were trying to tell PCA the indices, but the syntax is wrong. 
    
    #draws a scatter plot with title and axis lables
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('My PCA GRAPH')
    plt.xlabel('PC1 - {0}%'.format(perc_variation[0]))
    plt.ylabel('PC2 - {0}%'.format(perc_variation[1]))
    
    for sample in pca_df.index: #add sample names to graph
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
    plt.show()

main()

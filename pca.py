from ctypes import sizeof
import pandas
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np



#performing PCA on the data set
def main():
    
    # UHHHH get the file to be the data set yo
    # Using pandas to make my data structure of the set
    data = pandas.read_csv('/Users/ChristianWalsh/Downloads/ml-100k/u.data', sep='\t', header=None, names=['UserID', 'ItemID', 'Rating', 'Timestamp'])
    

    #centering and scaling the data
    scaled_data = preprocessing.scale(data.T) #passing the transpose
    #StandardScaler().fit_transform(data.T)
    
    pca = PCA()
    pca.fit(scaled_data) #does the math -loading scores, variation
    pca_data = pca.transform(scaled_data) #generate coords for the pca graph based on data
    
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
    
    #draws a scatter plot with title and axis lables
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('My PCA GRAPH')
    plt.xlabel('PC1 - {0}%'.format(perc_variation[0]))
    plt.ylabel('PC2 - {0}%'.format(perc_variation[1]))
    
    for sample in pca_df.index: #add sample names to graph
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
    plt.show()

main()
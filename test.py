from ctypes import sizeof
import pandas

#Access the file and read it 
dataFile = open("u.data", "r")
dataFileRead = dataFile.read()


#split it into seprate lines
fileByLine = dataFileRead.splitlines()


#split each line of the file into individual UserID, ItemID, Rating, and Timestamp
totalRating = 0
for line in fileByLine:
    individualThings = line.split()
    userId = individualThings[0]
    itemId = individualThings[1]
    rating = individualThings[2]
    timestamp = individualThings[3]
    
    #calculate the total rating of all entries
    totalRating = totalRating + int(rating)

#get number of ratings in the data set
numOfEntries = len(fileByLine)
#calculate the global mean of the data set, total rating of all entries / number of entries 
globalMean = totalRating/numOfEntries


print(globalMean)
    
    
    
    
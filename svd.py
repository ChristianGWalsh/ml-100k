import numpy
from numpy.linalg import svd
from numpy.linalg import multi_dot
from numpy.linalg import norm

# A = the array of stuff
#not the correct data just testing

A = numpy.array([
    [1,-0.8],
    [0,1],
    [1,0]
])

# i can print(A)
# A.shape tells me rows and cols

#doing the svd calc splitting A into U S VT
#full matrices for this case, change if not full
U, S, VT = svd(A, full_matrices=True)
#in this case, U will be one row shorter if i use false

#U contains left singular vectors
#S contains singular vaules
#VT contains the right singular vaules

#To display going from A(mxn) to U(mxm) S(mxn) VT(nxn)
#print(U.shape, S.shape, VT.shape)
#S is 1 dimensional hence the (2,)

#----------
#Turning U, S, and VT back into A
#making a 2d array of 0's to put my 1d S into
smat = numpy.zeros((3,2)) #size same as A
smat[:2, :2] = numpy.diag(S) #putting the S values into the 0 array
#print(smat)
print()

#use multidot to multiply the 3 matrices and get A
print(multi_dot([U,smat,VT]))

#to compare my remade A to the og matrix, i can take the "norm"
print(norm(A - multi_dot([U,smat,VT])))
#this number is basically 0, so I can consider it equal to the og

#-----------
#LOW RANK APPROX 
#great for compressing data and getting rid of useless aspects
#ex array
B = numpy.array([
    [.1649, .6164, .2730],
    [.2317, .8663, .3837],
    [.0334, .1249, .0553],
    [.2284, .8537, .3781],
    [.0769, .2874, .1273]
])

U1, S1, VT1 = svd(B, full_matrices=False) #FALSE cuz low rank
#print(U1.shape, S1.shape, VT1.shape)

#can see that the first singular value carries much more 
#importance than the rest, so just use the first
#this IS low rank approx
print(S1)

rank =1
U1_sub = U1[:,:rank]
VT1_sub = VT1[:rank,:]
S1_sub = numpy.diag(S1[:rank])
#using .dot here for just 2 matrix multi
B_low_rank = numpy.dot(numpy.dot(U1_sub, S1_sub), VT1_sub)

#use the norm to see how accurate it is
print(norm(B-B_low_rank))
#same thing, if basically 0, all is good






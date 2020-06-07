# save numpy array as csv file
import numpy as np
# define data
data = np.array([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
# save to csv file
np.savetxt('data.csv', data, delimiter=',')
data2 = np.loadtxt('data.csv', delimiter=',')
# print the array
print(data2)
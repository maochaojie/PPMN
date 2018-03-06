import scipy.io as sio
import sys
mat_file = sys.argv[1]
a=sio.loadmat(mat_file)
b = a['ours']
for iter in range(2000,16000,1000):
	key = 'single_yaqing_%d'%iter
	c = b[key]
	print '%f %s'%(c[0][0][0][0],key)

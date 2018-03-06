import sys
import time
def progresslog(s,i):
	sys.stdout.write(s+':{0:3}/{1:3}:'.format(i, 100)+'#'*(i/2)+'->'+'\r')
	sys.stdout.flush()
class ProgressBar:
	def __init__(self,count = 0, total = 0, width = 100):
		self.count = count
		self.total = total
		self.width = width
	def move(self):
		self.count += 1
	def log(self, s=''):
		# sys.stdout.flush()
		# sys.stdout.write('\r')
		
		progress = self.width * self.count/self.total
		sys.stdout.write('{0:3}/{1:3}:'.format(self.count, self.total)+'#'*progress+' '*(self.width - progress)+'\r')
		if progress == self.width:
			sys.stdout.write('\n')
		sys.stdout.flush()
		
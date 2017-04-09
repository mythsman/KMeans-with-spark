#!/usr/bin/env python
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score

if __name__=='__main__':
	if len(sys.argv)<3:
		print("please input two files.")
		exit(0)
	res=[]
	for i in xrange(20):
		arr1=[]
		arr2=[]
		with open(sys.argv[1]) as f:
			for line in f:
				arr1.append(int(line))
		with open(sys.argv[2]+str(i)+"/part-00000") as f:
			for line in f:
				arr2.append(int(line))
		ans=normalized_mutual_info_score(arr1,arr2)
		res.append(ans)
		print(ans)
	print(sum(res)/len(res))

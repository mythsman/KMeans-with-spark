#!/usr/bin/env python
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score

if __name__=='__main__':
	if len(sys.argv)<3:
		print("please input two files.")
		exit(0)
	arr1=[]
	arr2=[]
	with open(sys.argv[1]) as f:
		for line in f:
			arr1.append(int(line))
	with open(sys.argv[2]) as f:
		for line in f:
			arr2.append(int(line))
	print(normalized_mutual_info_score(arr1,arr2))

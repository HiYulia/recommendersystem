 
from pyspark import SparkConf, SparkContext
import json
import sys
import time
import itertools
from collections import Counter
from itertools import chain
from collections import OrderedDict
import random
import collections
import math
from statistics import mean 
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
def fillmissing(pairs):
	if pairs[1]==None:
		pairs=(pairs[0],3.5)
	return pairs

def predictcase2(userid,businessid):
	if userid in umb_maps and businessid in bmu_maps:
		otheritems_thisuser=dict(umb_maps[userid])
		meannumber=mean(otheritems_thisuser.values())
		related_users=bmu_maps[businessid]
		nom=0
		de=0 
		related_users=related_users[:40]
		for i in related_users:
			othersingleuser_items=dict(umb_maps[i[0]])
			commen_items=set(othersingleuser_items.keys()) & set(otheritems_thisuser.keys())
			commen_number=len(commen_items)
			if commen_number<=1:
				w_value=1
				mean_two=i[1]
			else:
				nominater=0
				denominater1=0
				denominater2=0
				one=0
				two=0
				for item in commen_items:
					one=one+otheritems_thisuser[item]
					two=two+othersingleuser_items[item]
				mean_one=one/commen_number
				mean_two=two/commen_number
				for item in commen_items:
					normalize_one=otheritems_thisuser[item]-mean_one
					normalize_two=othersingleuser_items[item]-mean_two
					nominater=nominater+normalize_one*normalize_two
					denominater1=denominater1+normalize_one**2
					denominater2=denominater2+normalize_two**2
				denominater=math.sqrt(denominater1*denominater2)
				if denominater==0:w_value=0
				else: w_value=nominater/denominater
			nom=nom+w_value*(i[1]-mean_two)
			de=de+abs(w_value)
		if de==0:predictvalue=meannumber
		else:predictvalue=meannumber+(nom/de)
	else: predictvalue=3.5
	return (userid,businessid,predictvalue)

def predictcase3(userid,businessid):
	if userid in umb_maps and businessid in bmu_maps:
		timestamp1= time.time()
		otherusers_thisitem=dict(bmu_maps[businessid])
		related_businesses=umb_maps[userid]
		nom=0
		de=0 
		timestamp2= time.time()
		related_businesses=related_businesses[:50]
		for i in related_businesses:
			othersingleitem_users=dict(bmu_maps[i[0]])
			commen_user=set(othersingleitem_users.keys()) & set(otherusers_thisitem.keys())
			commen_number=len(commen_user)
			if commen_number<=1:
				w_value=1
			else:
				nominater=0
				denominater1=0
				denominater2=0
				one=0
				two=0
				for user in commen_user:
					one=one+otherusers_thisitem[user]
					two=two+othersingleitem_users[user]
				mean_one=one/commen_number
				mean_two=two/commen_number
				for user in commen_user:
					normalize_one=otherusers_thisitem[user]-mean_one
					normalize_two=othersingleitem_users[user]-mean_two
					nominater=nominater+normalize_one*normalize_two
					denominater1=denominater1+normalize_one**2
					denominater2=denominater2+normalize_two**2
				denominater=math.sqrt(denominater1*denominater2)
				if nominater<0:w_value=0
				elif denominater==0:w_value=0
				else: w_value=nominater/denominater
			nom=nom+w_value*i[1]
			de=de+abs(w_value)

		timestamp3= time.time()
		if de==0:predictvalue=3.5
		else:predictvalue=nom/de	
	else: predictvalue=3.5
	return (userid,businessid,predictvalue)

def task1(train_file_name,test_file_name,case_id,output_file_name):
	conf = SparkConf().setMaster("local").setAppName("HW3")
	sc=SparkContext(conf=conf)
	startTime = time.time()
	data=sc.textFile(train_file_name)
	header = data.first()
	data = data.filter(lambda row:row != header).map(lambda line: line.split(",")).persist()
	val_data=sc.textFile(test_file_name)
	val_header = val_data.first()
	val_data = val_data.filter(lambda row:row != val_header).map(lambda line: line.split(",")).persist()
	#dict
	userid_dic_val = val_data.map(lambda x: x[0])
	userid_dic = data.map(lambda x: x[0]).union(userid_dic_val).distinct().zipWithIndex().persist()
	userid_dic_pos=userid_dic.collectAsMap()
	userid_dic_neg=userid_dic.map(lambda x: (x[1],x[0])).collectAsMap()

	businessid_dic_val = val_data.map(lambda x: x[1])
	businessid_dic = t=data.map(lambda x: x[1]).union(businessid_dic_val).distinct().zipWithIndex().persist()#24731
	businessid_dic_pos=businessid_dic.collectAsMap()
	businessid_dic_neg=businessid_dic.map(lambda x: (x[1],x[0])).collectAsMap()
	global umb_maps,bmu_maps
	if int(case_id)==1:
		ratings=data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2]))\
		.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
		val_ratings=val_data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2]))\
		.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
		rank = 20
		numIterations = 20
		model = ALS.train(ratings, rank, numIterations,lambda_=0.1)
		valdata = val_ratings.map(lambda p: (p[0], p[1]))
		predictions = model.predictAll(valdata).map(lambda r: ((r[0], r[1]), r[2]))
		ratesAndPreds = val_ratings.map(lambda r: ((r[0], r[1]), r[2])).leftOuterJoin(predictions).map(lambda x: (x[0],fillmissing(x[1]))).persist()
		output=ratesAndPreds.map(lambda x: (userid_dic_neg[x[0][0]],businessid_dic_neg[x[0][1]],x[1][1])).collect()
		MSE=ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
		RMSE=math.sqrt(MSE)
		print("Root Mean Squared Error = " + str(RMSE))
	
	elif int(case_id)==2:

		ratings=data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2])).distinct().persist()
		umb_maps=ratings.map(lambda l: (int (l[0]), (int(l[1]), float(l[2])))).groupByKey().map(lambda e: (e[0],list(e[1]))).collectAsMap()
		bmu_maps=ratings.map(lambda l: (int (l[1]), (int(l[0]), float(l[2])))).groupByKey().map(lambda e: (e[0],list(e[1]))).collectAsMap()
		val_ratings=val_data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2])).distinct()\
		.map(lambda l: (int(l[0]), int(l[1]), float(l[2]))).persist()

		valdata = val_ratings.map(lambda p: (predictcase2(p[0], p[1]),p[2])).persist()
		output=valdata.map(lambda x: (userid_dic_neg[x[0][0]],businessid_dic_neg[x[0][1]],x[0][2])).collect()
		MSE=valdata.map(lambda x:(x[0][2]-x[1])**2).mean()
		RMSE=math.sqrt(MSE)
		print("Root Mean Squared Error = " + str(RMSE))

	elif int(case_id)==3:

		ratings=data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2])).distinct().persist()
		umb_maps=ratings.map(lambda l: (int (l[0]), (int(l[1]), float(l[2])))).groupByKey().map(lambda e: (e[0],list(e[1]))).collectAsMap()
		bmu_maps=ratings.map(lambda l: (int (l[1]), (int(l[0]), float(l[2])))).groupByKey().map(lambda e: (e[0],list(e[1]))).collectAsMap()
		val_ratings=val_data.map(lambda line: (userid_dic_pos[line[0]], businessid_dic_pos[line[1]],line[2])).distinct()\
		.map(lambda l: (int(l[0]), int(l[1]), float(l[2]))).persist()
		
		valdata = val_ratings.map(lambda p: (predictcase3(p[0], p[1]),p[2])).persist()
		output=valdata.map(lambda x: (userid_dic_neg[x[0][0]],businessid_dic_neg[x[0][1]],x[0][2])).collect()
		MSE=valdata.map(lambda x:(x[0][2]-x[1])**2).mean()
		RMSE=math.sqrt(MSE)
		print("Root Mean Squared Error = " + str(RMSE))

	else: 
		return print("This is an invaild case number, please enter 1~3")

	with open(output_file_name, "w") as outfile1:
		outfile1.write("user_id, business_id, prediction:\n")
		for k in output:
			i=','.join(str(w) for w in k)
			outfile1.write(i+"\n")
	endTime1 = time.time()
	time1=endTime1-startTime
	print("Duration " + str(time1))

if __name__ == "__main__":
		task1(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])

#print(result)

#spark-submit xinyue_niu_task2.py yelp_train.csv yelp_val.csv 1 task2_case1.csv
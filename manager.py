from __future__ import print_function
from properties import Properties
from classificationModel import Model
from stream import Stream
from sklearn import svm, grid_search
import time, sys, datetime
import numpy as np
import random, math
from jda import Jda
import matlab.engine

class Manager(object):
	def __init__(self, sourceFile, targetFile):
		self.SDataBufferArr = None #2D array representation of self.SDataBuffer
		self.SDataLabels = None
		self.TDataBufferArr = None #2D array representation of self.TDataBuffer
		self.TDataLabels = None

		self.useKliepCVSigma = Properties.useKliepCVSigma

		self.kliep = None


		self.initialWindowSize = int(Properties.INITIAL_DATA_SIZE)
		self.maxWindowSize = int(Properties.MAX_WINDOW_SIZE)

		self.enableForceUpdate = int(Properties.enableForceUpdate)
		self.forceUpdatePeriod = int(Properties.forceUpdatePeriod)
		self.jda = None
		self.jdaEng = matlab.engine.start_matlab()
		self.ClsModel = Model()


		"""
		- simulate source and target streams from corresponding files.
		"""
		print("Reading the Source Dataset")
		self.source = Stream(sourceFile, Properties.INITIAL_DATA_SIZE)
		print("Reading the Target Dataset")
		self.target = Stream(targetFile, Properties.INITIAL_DATA_SIZE)
		print("Finished Reading the Target Dataset")

		Properties.MAXVAR = self.source.initialData.shape[0]

	"""
	Write value (accuracy or confidence) to a file with DatasetName as an identifier.
	"""
	def __saveResult(self, acc, datasetName):
		with open(datasetName + '_' + Properties.OUTFILENAME, 'a') as f:
			f.write(str(acc) + "\n")
		f.close()

	"""
	The main method handling multistream classification using KLIEP.
	"""
	def startFusionRegression(self, SrcDataName, TrgDataName, probFromSource):
		#save the timestamp
		globalStartTime = time.time()
		Properties.logger.info('Global Start Time: ' + datetime.datetime.fromtimestamp(globalStartTime).strftime('%Y-%m-%d %H:%M:%S'))
		#open files for saving accuracy and confidence
		fAcc = open(SrcDataName + '_' + TrgDataName + '_' + Properties.OUTFILENAME, 'w')
		fConf = open(SrcDataName + '_' + TrgDataName + '_confidence' + '_' + Properties.OUTFILENAME, 'w')

		#variable to track forceupdate period
		idxLastUpdate = 0


		#Get data buffer
		self.SDataBufferArr = self.source.initialData
		self.SDataLabels = self.source.initialDataLabels

		self.TDataBufferArr = self.target.initialData


		self.jda = Jda(Properties.jdaParK, Properties.jdaParLambda, Properties.jdaParGamma, Properties.jdaParIt, self.jdaEng)
		Zs, Zt, A, self.ClsModel = self.jda.run_JDA(self.SDataBufferArr, self.TDataBufferArr, self.SDataLabels)
		sDataSum = np.sum(Zs, axis=1)
		tDataSum = np.sum(Zt, axis=1)
		sDataMean = sDataSum / self.initialWindowSize
		tDataMean = tDataSum / self.initialWindowSize
		initMeanDisc = np.sqrt(np.sum(np.power(sDataMean - tDataMean,2)))


		Properties.logger.info('Estimating initial JDA')

		self.SDataBufferArr = self.SDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]
		self.SDataLabels = self.SDataLabels[-Properties.MAX_WINDOW_SIZE:]

		self.TDataBufferArr = self.TDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]


		Properties.logger.info('Initializing classifier with the first model')

		sDataIndex = 0
		tDataIndex = 0
		totAcc = 0.0


		#while tDataIndex < 100:
		#	newTargetDataArr = self.target.initialData[:, tDataIndex][np.newaxis].T
			# get Target Accuracy on the new instance
		#	resTarget = self.regModel.test(np.reshape(newTargetDataArr, (1, -1)))
		#	totError += abs(resTarget - self.target.initialDataLabels[tDataIndex])
		#	avgError = float(totError) / (tDataIndex + 1)
			#enoughInstToUpdate is used to see if there are enough instances in the windows to
			#estimate the weights
		#	tDataIndex += 1
		#	print(totError)
		#print(avgError)

		Properties.logger.info('Starting MultiStream Classification')
		while self.target.data.shape[1] > tDataIndex:
			"""
			if source stream is not empty, do proper sampling. Otherwise, just take
			the new instance from the target isntance.
			"""
			if self.source.data.shape[1] > sDataIndex:
				fromSource = random.uniform(0,1)<probFromSource
			else:
				print("\nsource stream sampling not possible")
				fromSource = False

			if fromSource:
				print('.', end="")
				#remove the first instance, and add the new instance in the buffers
				newSrcDataArr = self.source.data[:, sDataIndex][np.newaxis].T
				sDataSum += A.transpose() * newSrcDataArr
				sDataMean = sDataSum / (self.initialWindowSize + sDataIndex + 1)
				self.SDataBufferArr = self.SDataBufferArr[:, 1:]
				self.SDataLabels = self.SDataLabels[1:]
				#add new instance to the buffers
				self.SDataBufferArr = np.append(self.SDataBufferArr, newSrcDataArr, axis=1)
				self.SDataLabels.append(self.source.dataLabels[sDataIndex])
				sDataIndex += 1

			else:
				# Target Stream
				print('#', end="") # '#' indicates new point from target
				newTargetDataArr = self.target.data[:, tDataIndex][np.newaxis].T
				# get Target Accuracy on the new instance
				tDataSum += A.transpose() * newTargetDataArr
				tDataMean = tDataSum / (self.initialWindowSize + tDataIndex + 1)
				resTarget = self.ClsModel.test(np.reshape(A.transpose() * newTargetDataArr, (1,-1)))
				totAcc += (resTarget==self.target.dataLabels[tDataIndex])
				avgAcc = float(totAcc)/(tDataIndex+1)
				# save log info
				if (tDataIndex%100)==0:
					Properties.logger.info('\nTotal test instance: '+ str(tDataIndex+1) + ', Total Accuracy: ' + str(totAcc) + ', Average Accuracy: ' + str(avgAcc))
				fAcc.write(str(avgAcc)+ "\n")


				#remove the first instance from buffers
				self.TDataBufferArr = self.TDataBufferArr[:, 1:]
				self.TDataBufferArr = np.append(self.TDataBufferArr, newTargetDataArr, axis=1)


				tDataIndex += 1

			#print("sDataIndex: ", str(sDataIndex), ", tDataIndex: ", str(tDataIndex))
			enoughInstToUpdate = self.SDataBufferArr.shape[1]>=Properties.MAX_WINDOW_SIZE and self.TDataBufferArr.shape[1]>=Properties.MAX_WINDOW_SIZE
			if enoughInstToUpdate:
				print("Enough points in source and target sliding windows. Attempting to detect any change of distribution.")
				changeScore = math.log(np.sqrt(np.sum(np.power(sDataMean - tDataMean,2))) / initMeanDisc)
				if changeScore > Properties.jdaParThreshold:
					changeDetected = True
				else:
					changeDetected = False
				print("Change Score: ", changeScore)

			#instances from more than one class are needed for svm training
			if changeDetected or (self.enableForceUpdate and (tDataIndex + sDataIndex - idxLastUpdate == self.forceUpdatePeriod)):
				fConf.write("\n")
				Properties.logger.info(
					'\n-------------------------- Change of Distribution ------------------------------------')
				Properties.logger.info('Change of distribution found')
				Properties.logger.info(
					'sDataIndex=' + str(sDataIndex) + '\ttDataIndex=' + str(tDataIndex))
				Properties.logger.info('Training a model due to change detection')
				Zs, Zt, A, self.ClsModel = self.jda.run_JDA(self.SDataBufferArr, self.TDataBufferArr, self.SDataLabels)
				sDataSum = np.sum(Zs, axis=1)
				tDataSum = np.sum(Zt, axis=1)
				sDataMean = sDataSum / self.initialWindowSize
				tDataMean = tDataSum / self.initialWindowSize
				initMeanDisc = np.sqrt(np.sum(np.power(sDataMean - tDataMean, 2)))
				Properties.logger.info(self.ClsModel.getModelSummary())
				#update the idx
				idxLastUpdate = tDataIndex + sDataIndex
		#save the timestamp
		fConf.close()
		fAcc.close()
		globalEndTime = time.time()
		Properties.logger.info(
			'\nGlobal Start Time: ' + datetime.datetime.fromtimestamp(globalEndTime).strftime('%Y-%m-%d %H:%M:%S'))
		Properties.logger.info('Total Time Spent: ' + str(globalEndTime-globalStartTime) + ' seconds')
		Properties.logger.info('Done !!')
import logging, subprocess
import math
import threading, random

class Properties(object):
	IDENTIFIER = ''
	OUTFILENAME = ''
	TEMPDIR = ''
	LOGFILE = ''
	MAXVAR = 0
	BASEDIR = ''
	SRCAPPEND = ''
	TRGAPPEND = ''
	logger = None
	SENSITIVITY = 0.0
	MAX_WINDOW_SIZE = 0
	INITIAL_DATA_SIZE = 0
	enableForceUpdate = 0
	forceUpdatePeriod = 0
	jdaParK = 0
	jdaParLambda = 0.0
	jdaParGamma = 0.0
	jdaParIt = 0
	jdaParThreshold = 0.0
	def __init__(self, propfilename, SrcDataName, TrgDataName):
		dict = {}
		with open(propfilename) as f:
			for line in f:
				(key,val) = line.split('=')
				dict[key.strip()] = val.strip()

		self.__class__.jdaParK = int(dict['jdaParK'])
		self.__class__.jdaParLambda = float(dict['jdaParLambda'])
		self.__class__.jdaParGamma = float(dict['jdaParGamma'])
		self.__class__.jdaParIt = int(dict['jdaParIt'])
		self.__class__.jdaParThreshold = -math.log(float(dict['sensitivity']))


		self.__class__.MAXVAR = 0

		self.__class__.BASEDIR = dict['baseDir']
		self.__class__.SRCAPPEND = dict['srcfileAppend']
		self.__class__.TRGAPPEND = dict['trgfileAppend']

		self.__class__.SENSITIVITY = float(dict['sensitivity'])
		self.__class__.MAX_WINDOW_SIZE = int(dict['maxWindowSize'])
		self.__class__.INITIAL_DATA_SIZE = int(dict['initialDataSize'])

		self.__class__.enableForceUpdate = int(dict['enableForceUpdate'])
		self.__class__.forceUpdatePeriod = int(dict['forceUpdatePeriod'])

		self.__class__.IDENTIFIER = SrcDataName + '_' + TrgDataName + '_' + str(self.__class__.INITIAL_DATA_SIZE) \
									+ '_' + str(self.__class__.MAX_WINDOW_SIZE)
		self.__class__.OUTFILENAME = self.__class__.IDENTIFIER + '_' + dict['output_file_name']
		self.__class__.TEMPDIR = dict['tempDir']
		self.__class__.LOGFILE = self.__class__.IDENTIFIER + '_' + dict['logfile']

		if self.__class__.logger: self.__class__.logger = None
		self.__class__.logger = self.__setupLogger()

			#self.__class__.PY4JPORT = random.randint(25333, 30000)
			#t = threading.Thread(target=self.__startCPDJava)
			#t.daemon = True
			#t.start()

	def __startCPDJava(self):
		subprocess.call(['java', '-jar', 'change_point.jar', str(self.__class__.GAMMA), str(self.__class__.SENSITIVITY), str(self.__class__.MAX_WINDOW_SIZE), str(self.__class__.CUSHION), str(self.__class__.CONFCUTOFF), str(self.__class__.PY4JPORT)])

	def __setupLogger(self):
		logger = logging.getLogger(__name__)
		logger.setLevel(logging.INFO)

		sh = logging.StreamHandler()
		sh.setLevel(logging.INFO)
		logger.addHandler(sh)
		handler = logging.FileHandler(self.__class__.LOGFILE)
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		return logger

	def summary(self):
		line = 'Parameter values are as follows:'
		line += '\njdaParK = ' + str(self.jdaParK)
		line += '\njdaParLambda = ' + str(self.jdaParLambda)
		line += '\njdaParGamma = ' + str(self.jdaParGamma)
		line += '\njdaParIt = ' + str(self.jdaParIt)
		line += '\njdaParThreshold = ' + str(self.jdaParThreshold)
		line += '\ninitialWindowSize = ' + str(self.INITIAL_DATA_SIZE)
		line += '\nmaxWindowSize = ' + str(self.MAX_WINDOW_SIZE)
		line += '\nenableForceUpdate = ' + str(self.enableForceUpdate)
		line += '\nforceUpdatePeriod = ' + str(self.forceUpdatePeriod)
		line += '\nMaximum Num Variables = ' + str(self.MAXVAR)
		line += '\nOutput File = ' + str(self.OUTFILENAME)

		return line
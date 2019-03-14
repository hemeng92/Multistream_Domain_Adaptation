import sys
from manager import Manager
from properties import Properties

"""
Parameters
datasetName: main part of dataset file name, e.g., powersupply for powersupply_source_stream.csv, powersupply_target_stream.csv
baseline: 1=startMscKLIEP, 2=start, 3=start2, 4=start_skmm, 5=start_mkmm, 6=start_srconly, 7=start_trgonly
"""
def main(SrcDataName, TrgDataName, probFromSource):
	props = Properties('config.properties', SrcDataName, TrgDataName)
	srcfile = Properties.BASEDIR + SrcDataName + Properties.SRCAPPEND
	trgfile = Properties.BASEDIR + TrgDataName + Properties.TRGAPPEND
	mgr = Manager(srcfile, trgfile)

	Properties.logger.info(props.summary())
	Properties.logger.info('Start Stream Simulation')

	mgr.startFusionRegression(SrcDataName, TrgDataName, probFromSource)


from multistream import main



#print "Running syn data for global abrupt drift for regression"
#main('RegSynGlobalAbruptDrift', 0.1)

#print "Running syn data for global gradual drift for regression"
#main('RegSynGlobalGradualDrift', 0.1)

#print "Running syn data for local abrupt drift for regression"
#main('RegSynLocalAbruptDrift', 0.1)

#print "Running real world data for airfoil for regression"
#main('airfoil', 0.1)

#print "Running real world data for CASP for regression"
#main('CASP', 0.1)

#print "Running real world for housing for regression"
#main('household', 0.1)

#print("Running SynRegAbrupt for regression")
#main('SynRegAbrupt', 0.1)

print("Running real world for stream domain adaptation")
main('AU_rotate', 'EU_rotate', 0.2)


#print "Running real world for CASP for regression"
#main('flight_test', 0.5)

#print "Running test for regression"
#main('test', 0.5)

#print "Running real world for year prediction for regression"
#main('YearPredictionMSD', 0.1)

print "Done"

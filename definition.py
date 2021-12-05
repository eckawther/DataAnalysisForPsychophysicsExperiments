
# python modules
import sys
import os
# other modules
import matplotlib.pyplot as plt
import pandas
import numpy
import trajectory_base
import csv
import statistics
import sklearn.metrics
import math

###################################################### load data

def load_data_from_file(file_name, include_training=False):
    """ Load data from a production/reproduction experiment.

    Version for the experiments with psychopy. Loads csv files.

    Parameters
    ----------
    file_name : str
        Name of csv file to load.
    include_training : bool
        Include training trials? Default False.

    Returns
    -------
    stimuli : list
    responses : list 
    """
    if not file_name.endswith('.csv'):
        sys.exit('Provide csv file!')

    #print('loading ', file_name)
    df = pandas.read_csv(file_name)
    results = trajectory_base.paramsTrajectory(parameters=df.to_dict('records'))

    stimuli = numpy.array(results.getParameter('stimulus_duration'))
    responses = numpy.array(results.getParameter('key_resp_stop.rt'))

    # -- remove training data
    if not include_training:
        stimulus_ranges = numpy.array(results.getParameter('stimulus_range'))
        training_trials = numpy.where(stimulus_ranges == 'Training')[0]

        if len(training_trials):
            stimuli = numpy.delete(stimuli, training_trials)
            responses = numpy.delete(responses, training_trials)
            #print(""Removed"", len(training_trials), ""training trials."")

    return stimuli, responses
###################################################### main

def compute_parameters (stimuli, responses):
               
    #helps organize my list
    indexes = []
    for s in numpy.unique(stimuli):
        index=numpy.where(stimuli==s)[0]
        indexes.append(index)
            
    #stdev
    stdeviation = []
    

    #print(""start"")
    for i,j in enumerate(indexes):          
        res = responses[j]
        stdeviation.append(numpy.std(res,ddof=1))
        
    stdeviation=numpy.mean(stdeviation)

    #print(""the stdev for "", file, stdeviation)
            
    #BIAS
    BIAS = numpy.mean(responses - stimuli)
    #print (""the BIAS for"", file, ""is"", BIAS)
            
    #Square-root of the mean squared
    BIAS_squared= math.sqrt(numpy.mean((responses - stimuli)**2))
    #print (""the BIAS squared for"", file, ""is"", BIAS_squared)
            
    #CV
    CV = stdeviation / numpy.mean(responses)
    #print(""The CV for "", file, ""is "", CV)
            
    #slope
    from scipy.stats import linregress
    result = linregress(stimuli,responses)
    #print(""The slope for "", file, ""is "" , result.slope)
           
    #RMSE mine
    MSE = BIAS_squared + stdeviation ** 2
    RMSE= numpy.sqrt(MSE)
    #print (""The RMSE is"", RMSE)
    
    #standard error
    #SDERROR = stdeviation / math.sqrt(len(responses))       
            
    return stdeviation, BIAS, BIAS_squared, CV, result.slope, RMSE

###################################################### 

    
    
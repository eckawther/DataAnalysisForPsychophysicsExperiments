""" For plotting psychopy data.
"""
# python modules
import sys

# other modules
import matplotlib.pyplot as plt
import pandas
import numpy
import trajectory_base
import math

###################################################### load data

def load_data_from_file(file_name,include_training=False):
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

    print('loading ', file_name)
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
            print("Removed", len(training_trials), "training trials.")
 
    return stimuli, responses


###################################################### main

file_name = sys.argv[1],sys.argv[2]                              # get name of file to load from command line parameters
stimuli1, responses1 = load_data_from_file(file_name[0])  # load the data
stimuli2, responses2 = load_data_from_file(file_name[1])

for i,s1 in enumerate(stimuli1): 
    print(i,s1)
  
for i,s2 in enumerate(stimuli2): 
    print(i,s2) 


fig, axes=plt.subplots(1,2)
axes[0].plot(stimuli1,responses1,'.',markersize=2, color='m')
axes[1].plot(stimuli2,responses2,'.',markersize=2, color='b')

for ax in axes:
    ax.set_xlim(0,1.5)
    ax.set_ylim(0,1.5)
    ax.set_xlabel('Stimulus (s)')
    ax.set_ylabel('Reproduction (s)')
    ax.plot([0,1.5],[0,1.5],'--',color=[0.75,0.75,0.75])


stimuli1=numpy.array(stimuli1)
stimuli2=numpy.array(stimuli2)
responses1=numpy.array(responses1)
responses2=numpy.array(responses2)

for s1  in numpy.unique(stimuli1):
    ind1=numpy.where(stimuli1==s1)[0]
    m1=numpy.mean(responses1[ind1])
    print(s1,m1)
    
    axes[0].plot(s1,m1,'o',markersize=5, color='c')
    
    
for s2  in numpy.unique(stimuli2):
    ind2=numpy.where(stimuli2==s2)[0]
    m2=numpy.mean(responses2[ind2])
    print(s2,m2)
    
    axes[1].plot(s2,m2,'o',markersize=5, color='c')


from scipy.stats import linregress

result1=linregress(stimuli1,responses1)
axes[0].plot(stimuli1, result1.intercept + result1.slope*stimuli1, color='c')


result2=linregress(stimuli2,responses2)
axes[1].plot(stimuli2, result2.intercept + result2.slope*stimuli2, color='c')

plt.show()

#############################################################

#Standard deviation



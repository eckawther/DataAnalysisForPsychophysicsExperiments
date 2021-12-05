# python modules
import sys
import os, glob
# other modules
import matplotlib.pyplot as plt
import pandas
import numpy
import trajectory_base
import definition
import csv
import statistics
import sklearn.metrics
import math
import seaborn as sns

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
            #print("Removed", len(training_trials), "training trials.")

    return stimuli, responses
###################################################### main

if __name__ == '__main__':

    #returns the full directory path    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    # T_data_path = os.path.join(dir_path, "data_reproduction/pruned/T/")
    # D_data_path = os.path.join(dir_path, "data_reproduction/pruned/D/")
    # Z_data_path = os.path.join(dir_path, "data_reproduction/pruned/Z/")
    #
    # data_path_list = [T_data_path, D_data_path, Z_data_path]
    plot_path = os.path.join(dir_path, "plot")
    h_rowlist = ["stdeviation", "BIAS", "BIAS_squared", "CV", "slope", "RMSE"]
    #
    # for folder_path in data_path_list:
    #
    #     for i,file in enumerate(sorted(os.listdir(folder_path))):
    #         #print(file,i)
    #         if file.endswith('.csv'):
    #             file_path = os.path.join(folder_path, file)
    #             stimuli, responses = load_data_from_file(file_path)
    #             stdeviation, BIAS, BIAS_squared, CV, slope, RMSE = definition.compute_parameters (stimuli, responses)
    #
    #             plot_file_path = os.path.join(plot_path, file[0], file.split("_")[0] + ".csv")
    #             #print(file,i)
    #             if ("r1" in file):
    #                 h_rowlist = ["stdeviation", "BIAS", "BIAS_squared", "CV", "slope", "RMSE"]
    #                 d_rowlist = [stdeviation, BIAS, BIAS_squared, CV, slope, RMSE]
    #                 with open(plot_file_path, 'w') as plot_file:
    #                     writer = csv.writer(plot_file)
    #                     writer.writerow(h_rowlist)
    #                     writer.writerow(d_rowlist)
    #             elif ("r2" in file) or ("r3" in file):
    #                 rowlist = [stdeviation, BIAS, BIAS_squared, CV, slope, RMSE]
    #                 with open( plot_file_path, 'a') as plot_file:
    #                     writer = csv.writer(plot_file)
    #                     writer.writerow(rowlist)


    T_plot_path = os.path.join(dir_path, "plot/T/")
    D_plot_path = os.path.join(dir_path, "plot/D/")
    Z_plot_path = os.path.join(dir_path, "plot/Z/")
    
    plot_path_list = [T_plot_path, D_plot_path, Z_plot_path]
    plot_figures_path = os.path.join(plot_path, "figures/")

    plot_data = []    
    error_data = [] 
    
    for folder_path in plot_path_list:
        mean_list = []
        for file in sorted(os.listdir(folder_path)):
            fname = file.split(".")[0]
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as plot_file:
                    file_reader = csv.reader(plot_file,delimiter=',')
                    data = list(file_reader)
                    header = data[0]
                    r1_data = list(map(float,data[1]))
                    r2_data = list(map(float,data[2]))
                    r3_data = list(map(float,data[3]))
                    data_list = [r1_data,r2_data,r3_data]
                    mean_list.append(data_list)

        r1_mean_list = []
        r2_mean_list = []
        r3_mean_list = []
        
        for i in range(len(mean_list)):
            r1_mean_list.append(mean_list[i][0])
            r2_mean_list.append(mean_list[i][1])
            r3_mean_list.append(mean_list[i][2])

        plot_data.append([numpy.mean(r1_mean_list,axis=0),numpy.mean(r2_mean_list,axis=0),numpy.mean(r3_mean_list,axis=0)])
        error_data.append([numpy.std(r1_mean_list,axis=0)/len(r1_mean_list),
                           numpy.std(r2_mean_list,axis=0)/len(r2_mean_list),
                           numpy.std(r3_mean_list,axis=0)/len(r3_mean_list)])
    
    plot_data_path = os.path.join(plot_path,"plot_data.csv")
    with open(plot_data_path, 'w') as plot_data_file:
                        writer = csv.writer(plot_data_file,delimiter =',')
                        for i,r_country in enumerate(plot_data):
                            writer.writerow(h_rowlist)
                            for j, r_ranges in enumerate(plot_data[i]):
                                writer.writerow(plot_data[i][j])
                            writer.writerow([])    
    
    for i, h in enumerate(h_rowlist):

        data = []
        err_data = []
        
        for l in range(3):
            data.append([plot_data[0][l][i], plot_data[1][l][i], plot_data[2][l][i]])
            err_data.append([error_data[0][l][i], error_data[1][l][i], error_data[2][l][i]])
        data = numpy.array(data)
        err_data = numpy.array(err_data)
        
        X = numpy.arange(3)
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8 ,0.8])
        w = 0.4
        Nat_labels=["Tunisia", "Germany", "Cyprus"]
        #x = ['r1','r2','r3']
        plt.xlabel("sessions")
        #y = [r1_data[i],r2_data[i],r3_data[i]]
        plt.ylabel(h)
        #ax.bar(x, y)
        plt.title("Time estimation between different nationalities and different time intervals", loc='center')
        #plt.titel.set_label_position('top')
        #plt.errorbar(x, y, linestyle='None', marker='^', color='b', capsize=3)
        ax.bar(X + 0.00, data[:, 0], yerr=err_data[:, 0], color = 'r', width = 0.25)
        ax.bar(X + 0.25, data[:, 1], yerr=err_data[:, 1], color = 'g', width = 0.25)
        ax.bar(X + 0.50, data[:, 2], yerr=err_data[:, 2], color = 'b', width = 0.25)
        plt.xticks(X+w/2,('small', 'medium', 'Large'))
        ax.legend(Nat_labels)#loc="lower center", bbox_to_anchor=(0.25, 1.15), ncol=2)
        
        sns.set(style="whitegrid")
        
        #for i in range(len(data)):
                #plt.scatter (X, data[i], color = 'black', marker=".")
        
        #for i in range(len(data)):
            #for j in range(len(data[i])):
                #ax.scatter(data[i][j] + numpy.random.random(data[i][j].size) * w - w / 2, data[i][j], color= 'black')
                


                   
        #show and save figure
        plt.show()
        figure_path = os.path.join(plot_figures_path, h + ".png")
        plt.savefig(figure_path)
               
        
    """print(plot_data[0])
    print(plot_data[1])
    print(plot_data[2])"""

    for f,folder_path in enumerate(plot_path_list):
        for file in sorted(os.listdir(folder_path)):
            fname = file.split(".")[0]
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as plot_file:
                    file_reader = csv.reader(plot_file,delimiter=',')
                    data = list(file_reader)
                    header = data[0]
                    r1_data = list(map(float,data[1]))
                    r2_data = list(map(float,data[2]))
                    r3_data = list(map(float,data[3]))
                    
                    p1 = abs(r1_data[0] - plot_data[f][0][0]) / plot_data[f][0][0]
                    p2 = abs(r2_data[0] - plot_data[f][1][0]) / plot_data[f][1][0]
                    p3 = abs(r3_data[0] - plot_data[f][2][0]) / plot_data[f][2][0]
                        
                        
                    if (p1 > 0.75):
                        print(fname, " is outlier in r1")
                    if (p2 > 0.75):
                        print(fname, " is outlier in r2")
                    if (p3 > 0.75):
                        print(fname, " is outlier in r3")
    pass






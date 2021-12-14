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

###################################################### main

if __name__ == '__main__':

    #returns the full directory path    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    #returns the plot folder path 
    plot_path = os.path.join(dir_path, "plot")
    #saves the first row in csv files for analysed data
    h_rowlist = ["stdeviation", "BIAS", "BIAS_squared", "CV", "slope", "RMSE"]

    #returns the plot paths for different nationalities 
    T_plot_path = os.path.join(dir_path, "plot/T/")
    D_plot_path = os.path.join(dir_path, "plot/D/")
    Z_plot_path = os.path.join(dir_path, "plot/Z/")
    
    #gathers all plot paths in one list
    plot_path_list = [T_plot_path, D_plot_path, Z_plot_path]

    #returns path for saving figures
    plot_figures_path = os.path.join(plot_path, "figures/")

    #initializing plot and error data lists
    plot_data = []    
    error_data = [] 
    


    #main for loup for all folders (for different nationalities)
    for folder_path in plot_path_list:
        mean_list = []
        #calculates the mean for a specific parameter (stdve,bias...) for a specific range for a specific country 
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

        #here is where the data to be plotted is saved 
        plot_data.append([numpy.mean(r1_mean_list,axis=0),numpy.mean(r2_mean_list,axis=0),numpy.mean(r3_mean_list,axis=0)])
        #error bars
        error_data.append([numpy.std(r1_mean_list,axis=0)/len(r1_mean_list),
                           numpy.std(r2_mean_list,axis=0)/len(r2_mean_list),
                           numpy.std(r3_mean_list,axis=0)/len(r3_mean_list)])
    


    #here is the plotting part!
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
        plt.xlabel("sessions")
        plt.ylabel(h)
        plt.title("Time estimation between different nationalities and different time intervals", loc='center')
        ax.bar(X + 0.00, data[:, 0], yerr=err_data[:, 0], color = 'r', width = 0.25)
        ax.bar(X + 0.25, data[:, 1], yerr=err_data[:, 1], color = 'g', width = 0.25)
        ax.bar(X + 0.50, data[:, 2], yerr=err_data[:, 2], color = 'b', width = 0.25)
        plt.xticks(X+w/2,('small', 'medium', 'Large'))
        ax.legend(Nat_labels) #loc="lower center", bbox_to_anchor=(0.25, 1.15), ncol=2)
        
        sns.set(style="whitegrid")
                   
        #show and save figure
        plt.show()
        figure_path = os.path.join(plot_figures_path, h + ".png")
        plt.savefig(figure_path)


    #this for loop can identify outliers in the data.
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

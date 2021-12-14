# python modules
import sys, os

# other modules
import numpy
import pandas
import csv
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels


###################################################### functions

def anova_table(aov):
    """ Add effect siz measure to ANOVA table.

    Statsmodel's ANOVA table does not provide any effect size measures to tell
    if the statistical significance is meaningful. Here eta-squared and omega-squared are calculated.
    Omega-squared is considered a better measure of effect size since it is unbiased
    in it's calculation by accounting for the degrees of freedom in the model.
    """
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]

    return aov


###################################################### load data
# needs to be pandas DataFrame

dir_path = os.path.dirname(os.path.abspath(__file__))
T_data_path = os.path.join(dir_path, "plot/T/")
D_data_path = os.path.join(dir_path, "plot/D/")
Z_data_path = os.path.join(dir_path, "plot/Z/")
data_path_list = [T_data_path, D_data_path, Z_data_path] 
crosscul_file_path = os.path.join(dir_path, "crosscultural.xlsx")

crosscul_df = pandas.read_excel(crosscul_file_path)

id_col     = crosscul_df['ID'].tolist()
gamer_col  = crosscul_df['Gamer'].to_numpy()
music_col  = crosscul_df['Musical_instruments'].to_numpy()
gender_col = crosscul_df['Gender'].tolist()

#psycho_data_all = pandas.DataFrame([], columns=['ID', 'Country', 'Slope', 'CV', 'stdeviation' , 'BIAS', 'BIAS_squared', 'RMSE', 'Range',])
df_data = []
#on a 3 mini-tableaux, chacun contenant T-D-Z pour un seul range 
for i in range(3):
    #3 pays dans chacun des mini-tableaux
    for folder_path in data_path_list:
        #on a plusieurs participants pour chaque pays
        for file in sorted(os.listdir(folder_path)):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as data_file:
                    file_reader = csv.reader(data_file,delimiter=',')
                    #data contient le fichier CSV, par exemple T01.csv
                    data = list(file_reader)
                    header = data[0]
                    #r_data contient juste la ligne contenant le range etudié
                    r_data = list(map(float,data[i+1]))
                    if ('T' in file):
                        country = 'Tunisia'
                    elif ('D' in file):
                        country = 'Germany'
                    elif ('Z' in file):
                        country = 'Cyprus'
                    df_data.append([file.split(".")[0], country, r_data[0], 
                                                                 r_data[1], 
                                                                 r_data[2], 
                                                                 r_data[3], 
                                                                 r_data[4], 
                                                                 r_data[5] , i+1, gamer_col[id_col.index(file.split(".")[0])],
                                                                                  music_col[id_col.index(file.split(".")[0])],
                                                                                  gender_col[id_col.index(file.split(".")[0])]])
                    #psycho_data_all.append([df])


psycho_data_all = pandas.DataFrame(df_data, columns=['ID', 'Country', 'stdeviation' , 'BIAS', 'BIAS_squared', 'CV', 'Slope', 'RMSE', 'Range',
                                                     'Gamer', 'Musical_instruments', 'Gender'])
print(psycho_data_all)             



###################################################### display summary data and make ANOVAs for each stimulus

parameters = {'stdeviation' : 'stdeviation',
              'BIAS' : 'BIAS',
              'BIAS_squared' : 'BIAS_squared',
              'CV' : 'CV',
              'Slope':'Slope',
              'RMSE' : 'RMSE' }
for param_num, param in enumerate(parameters):
    #print(parameters[param])
    #print('_'*len(parameters[param]))

    # -- regular two-way ANOVA
    model = ols(param + ' ~ C(Country) + C(Range) + C(Country):C(Range)', data=psycho_data_all).fit()
    aov2w_table = sm.stats.anova_lm(model, typ=2)
    #print(anova_table(aov2w_table))

# =============================================================================
#     #Tukeyhsd
# =============================================================================
    multicomp_results = statsmodels.stats.multicomp.pairwise_tukeyhsd(psycho_data_all[param], psycho_data_all['Country'])
    #print(multicomp_results)
    multicomp_results.plot_simultaneous(comparison_name='Tunisia')
   
    #print()

# =============================================================================
# CULTURAL DIFFERENCE
# GAMER
# =============================================================================
    
    # -- regular ONE-way ANOVA
    model = ols(param + ' ~ C(Gamer)', data=psycho_data_all).fit()
    aov1w_table_game = sm.stats.anova_lm(model, typ=1)
    print(anova_table(aov1w_table_game))

    # -- regular two-way ANOVA
    model = ols(param + ' ~ C(Gamer) + C(Range) + C(Gamer):C(Range)', data=psycho_data_all).fit()
    aov2w_table_game = sm.stats.anova_lm(model, typ=2)
    print(anova_table(aov2w_table_game))  
    
# =============================================================================
# CULTURAL DIFFERENCE
# MUSIC
# =============================================================================
    
    ''' #-- regular ONE-way ANOVA
    model = ols(param + ' ~ C(Musical_instruments)', data=psycho_data_all).fit()
    aov1w_table_music = sm.stats.anova_lm(model, typ=1)
    print(aov1w_table_music)

    # -- regular two-way ANOVA
    model = ols(param + ' ~ C(Musical_instruments) + C(Range) + C(Musical_instruments):C(Range)', data=psycho_data_all).fit()
    aov2w_table_music = sm.stats.anova_lm(model, typ=2)
    print(aov2w_table_music)'''
    
# =============================================================================
# CULTURAL DIFFERENCE
# Gender
# =============================================================================
    
    '''# -- regular ONE-way ANOVA
    model = ols(param + ' ~ C(Gender)', data=psycho_data_all).fit()
    aov1w_table_gender = sm.stats.anova_lm(model, typ=1)
    print(anova_table(aov1w_table_gender))

    # -- regular two-way ANOVA
    model = ols(param + ' ~ C(Gender) + C(Range) + C(Gender):C(Range)', data=psycho_data_all).fit()
    aov2w_table_gender = sm.stats.anova_lm(model, typ=2)
    print(anova_table(aov2w_table_gender))'''    
    
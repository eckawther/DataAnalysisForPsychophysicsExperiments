__author__ = "KT"
# python modules
import os

# other modules
import numpy
import scipy.stats

import matplotlib.pyplot as pl
from itertools import count


###################################################### FUNCTIONS


###################################################### CLASSES

class paramsTrajectory(object):
    """ Class for handling the parameters of an experiment.
    """

    def __init__(self, times=None, parameters=None, meta=None):

        # -- initialize some variables
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta
        self.t_start = 0.0
        self.mazeType = ''
        self.timeUnit = 's'

        # -- get real values if parsable from entries in meta
        if 'dt' in self.meta:
            self.dt = self.meta['dt']                  # dt [s]
        if 't_start' in self.meta:
            self.t_start = self.meta['t_start']        # t_start [s]
        if 'time' in self.meta:
            self.time = self.meta['time']              # time and date, when recording was started
                                                       # just to provide some time stamp
        if 'mazetype' in self.meta:
            self.mazeType = self.meta['mazetype']

        if times is None:
            self.times = []
        else:
            self.times = times
        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters


    def __getitem__(self, item):
        """ Get sliced version of object.

        Paramaters
        ----------
        trial : int
            Trial to split at.

        Returns
        -------
        param_traj0, param_traj1
        """
        param_traj = paramsTrajectory(self.times[item], self.parameters[item], self.meta)
        for attr in ['mazeType', 'timeUnit']:
            setattr(param_traj, attr, getattr(self, attr))

        return param_traj


    def getParameter(self, key, default=0):
        """Get values for parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.
        default : value
            Default value if key is not in dictionary.

        Returns
        -------
        param_array : list
            List of values for the parameter from the parameters dictionary."""

        param_array = []
        for p in self.parameters:
            param_array.append(p.get(key, default))

        return param_array


    def setParameter(self, key, value):
        """Set values for parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.
        value : value
            Value to be set to.
        """
        for p in self.parameters:
            p[key] = value


    def roundParameter(self, key, r=1.):
        """Round values for parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.
        round : float
            Value to be set to.
        """
        for p in self.parameters:
            p[key] = numpy.round(p[key]/r)*r


    def plotParameter(self, key, fig=None, ax=None, smoothed=False):
        """Plot the values of a parameter.

        Parameters
        ----------
        key : str
            Key name of parameter in dictionary.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional
        smoothed : bool, optional
            Smooth the development of trials. Defaults to False.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib axes
        """

        if not fig:
            fig = pl.figure(figsize=(8, 8))
        if not ax:
            pos = [0.125, 0.6, 0.5, 0.35]
            ax = fig.add_axes(pos)
            pos_hist = list(pos)
            pos_hist[0] += pos_hist[2] + .1
            pos_hist[2] = .25
            ax_hist = fig.add_axes(pos_hist)
            pos = list(pos)
            pos[2] = .75
            pos[1] -= pos[3] + .1
            ax2 = fig.add_axes(pos)

        param_array = numpy.array(self.getParameter(key))
        ax.plot(param_array, 'o-')
        print('avg, std:', param_array.mean(), param_array.std())
        print('chi-square:', scipy.stats.chisquare(numpy.histogram(param_array, numpy.unique(param_array))[0]))
        ax_hist.hist(param_array, numpy.unique(param_array).size)
        if smoothed:
            smoothed_param_array = smooth(param_array, smoothed, 'flat')
            ax.plot(smoothed_param_array)
            ax_hist.hist(smoothed_param_array, 10)
            print('smoothed avg, std:', smoothed_param_array.mean(), smoothed_param_array.std())

        if self.times.size:
            ax2.plot(self.times, param_array, '.-')

        # huebsch machen
        for a in [ax, ax_hist, ax2]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel('#')
        ax.set_ylabel(key)

        ax2.set_ylabel(key)
        ax2.set_xlabel('Times ('+self.timeUnit+')')

        return fig, ax


    def get_parameter_distribution(self, key):
        """ Get count/distribution of parameter.
        
        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        
        Returns
        -------
        
        """
        param_list = self.getParameter(key)

        # creating a dictionary of parameter counts
        param_dict = dict((x, [param_list.count(x),
                               numpy.float(param_list.count(x))/numpy.float(len(param_list))]) for x in param_list)


        return param_dict

    
    def plotParameterDistribution(self, key, fig=None, ax=None, labelsize=None,
                                  showFigs=False, saveFig=False, saveFolder=''):
        """ Plots distributions for parameter.

        One of the parameter distribution and one of the transition distribution across trials.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        fig : matplotlib figure
        ax : matplotlib axes
        labelsize : int
            Font size of the label.

        Returns
        -------
        fig : matplotlib figure
        ax_top : matplotlib axes
        ax_bottom : matplotlib axes
        """

        param_dict = self.get_parameter_distribution(key)
        
        self.param_list = param_dict.keys()
        self.param_dict = [x[0] for x in param_dict.values()]
    
        # sorting the parameter dictionary and storing its keys and values in variables
        param_sorted_dict = sorted([(param_key, param_value) for (param_key, param_value) in self.param_dict.items()])
        param_keys = []
        param_values = []
        for i in range(len(param_sorted_dict)):
            param_keys.append(param_sorted_dict[i][0])
            param_values.append(param_sorted_dict[i][1])
    
        # doubling all the parameters except the first and last one, to analyse parameter transitions
        doubled_params = [str(self.param_list[0])]
        for i in numpy.arange(1, len(self.param_list)-1):
            doubled_params.extend([str(self.param_list[i]), str(self.param_list[i])])
        doubled_params.append(str(self.param_list[len(self.param_list)-1]))
        param_transitions = [(', '.join(map(str, doubled_params[n:n+2]))).replace(",", r"$\rightarrow$")
                            for n in range(0, len(doubled_params), 2)]
        self.param_transitions_dict = dict((x, param_transitions.count(x)) for x in param_transitions)
    
        # plotting absolute and transition parameter distributions as histograms
        if not fig:
                fig = pl.figure(figsize=(8, 4))
        if ax:
            pos = ax.get_position()      # [left, bottom, width, height]
            ax_top = fig.add_axes([pos[0], pos[1]+(4*pos[3]/7), pos[2], 3*pos[3]/7])
            ax_bottom = fig.add_axes([pos[0], pos[1], pos[2], 3*pos[3]/7])
        else:
            pos = [.25, .21, .65, .625]    # [left, bottom, width, height]
            ax_top = fig.add_axes([pos[0], pos[1]+(4*pos[3]/7), pos[2], 3*pos[3]/7])
            ax_bottom = fig.add_axes([pos[0], pos[1], pos[2], 3*pos[3]/7])
    
        ax_top.bar(*zip(*zip(count(), param_values)))
        ax_top.set_title('Absolute and transition %s distribution' %key)
        ax_top.set_xticks((1*zip(*zip(count(0.4), param_keys)))[0])
        ax_top.set_xticklabels((1*zip(*zip(count(0.4), param_keys)))[1])
        if labelsize:
            for tick in ax_top.xaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
            for tick in ax_top.yaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
        ax_top.set_xlabel(key)
        ax_top.set_ylabel('Value count')

        ax_bottom.bar(*zip(*zip(count(), self.param_transitions_dict.values())))
        ax_bottom.set_xticks((1*zip(*zip(count(0.4), self.param_transitions_dict)))[0])
        ax_bottom.set_xticklabels((1*zip(*zip(count(0.4), self.param_transitions_dict)))[1])
        if labelsize:
            for tick in ax_bottom.xaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
            for tick in ax_bottom.yaxis.get_major_ticks():
                tick.label.set_fontsize(labelsize)
        ax_bottom.set_xlabel('%s transition' %key)
        ax_bottom.set_ylabel('Value count')

        if saveFig:
            if not os.path.exists(saveFolder):
                os.mkdir(saveFolder)
            print('Parameter Distribution plot saved to:', saveFolder+'Parameter_Distribution.png')
            fig.savefig(saveFolder+'Parameter_Distribution.png', format='png')
        if not showFigs:
                pl.ioff()
                pl.close(fig)
        else:
            pl.show()

        return fig, ax_top, ax_bottom


    def plotParam1vsParam2(self, key1, key2, fig=None, ax=None, refline=False):
        """Plot two parameters against each other.

        Parameters
        -----------
        key1 : str
            Name of one parameter.
        key2 : str
            Name of other parameter.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional
        refline : bool, optional
            Draw reference line.

        Returns
        -------
        (fig, ax), (fig1, ax1, ax1b) : two tuples
            Tuples with Figure and Axes handles.
        """

        if not fig:
            fig = pl.figure()
            fig1 = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)
            ax1 = fig1.add_subplot(111)

        param_array1 = numpy.array(self.getParameter(key1))
        param_array2 = numpy.array(self.getParameter(key2))
        mini = min(param_array1.min(), param_array2.min())
        maxi = max(param_array1.max(), param_array2.max())
        if refline:
            ax.plot([mini, maxi], [mini, maxi], '--', color=numpy.ones(3)*.5, linewidth=1)
        ax.plot(param_array1, param_array2, 'bo', markerfacecolor='none', markeredgecolor='b', alpha=.5)
        ax1.plot(param_array1, 'bo-', linewidth=1)
        ax1b = ax1.twinx()
        ax1b.plot(param_array2, 'go-', linewidth=1)

        # huebsch machen
        custom_plot.turnOffAxes(ax1b, ['top'])
        ax1.set_ylabel(key1)
        ax1b.set_ylabel(key2)

        for a in [ax]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_xlim(ax.get_xlim()*numpy.array([.95, 1.05]))
        ax.set_ylim(ax.get_ylim()*numpy.array([.95, 1.05]))

        return (fig, ax), (fig1, ax1, ax1b)


    def corrParam1vsParam2(self, key1, key2):
        """Determine Pearson r between two parameters.

        Parameters
        -----------
        key1 : str
            Name of one parameter.
        key2 : str
            Name of other parameter.

        Returns
        -------
        r, p : floats Pearson r and p
        """

        param_array1 = numpy.array(self.getParameter(key1))
        param_array2 = numpy.array(self.getParameter(key2))

        return scipy.stats.pearsonr(param_array1, param_array2)


    def parameterTransitionsDict(self, key, labelsize=None, showFigs=False, saveFig=False, saveFolder=''):
        self.plotParameterDistribution(key, labelsize=labelsize, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)
        return self.param_transitions_dict

    def parameterDict(self, key, labelsize=None, showFigs=False, saveFig=False, saveFolder=''):
        self.plotParameterDistribution(key, labelsize=labelsize, showFigs=showFigs, saveFig=saveFig, saveFolder=saveFolder)
        return self.param_dict


    def testParameter(self, key, p=.1, smoothed=10, display=True, fig=None, ax=None):
        """ Test parameter time array for local uniform distribution and make some plots.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        p : float
            Minimum for acceptance of the p-value of the chi-squared test.
        smoothed : bool, optional
            Smooth the development of trials. Defaults to 10.
        display : bool, optional
            Display the a plot of the test? Defaults to True.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional

        Returns
        -------
        chi_ok : bool
            True if all p-values of the chi-squared test are above given p.
        """

        param_array = numpy.array(self.getParameter(key))
        bins = numpy.unique(param_array)
        bins = numpy.concatenate([bins, [bins.max()+.1]])

        # chi-square testing
        chi_p = []
        for i in range(param_array.size-smoothed):
            chi_p.append(scipy.stats.chisquare(numpy.histogram(param_array[i:i+smoothed], bins)[0])[1])
            if i in [10, 50]:
                print(numpy.histogram(param_array[i:i + smoothed], bins))


        chi_p = numpy.array(chi_p)
        chi_ok = False
        if numpy.all(chi_p > p):
            chi_ok = True

        if not display:
            return chi_ok

        if not fig:
            fig = pl.figure(figsize=(8, 8))
        if not ax:
            pos = [0.125, 0.75, 0.5, 0.23]
            ax = fig.add_axes(pos)
            pos_hist = list(pos)
            pos_hist[0] += pos_hist[2] + .1
            pos_hist[2] = .25
            ax_hist = fig.add_axes(pos_hist)
            pos = list(pos)
            pos[2] = 0.775
            pos[1] -= pos[3] + .1
            ax_corr = fig.add_axes(pos)
            pos = list(pos)
            pos[1] -= pos[3] + .1
            ax2 = fig.add_axes(pos)

        ax.plot(param_array, 'o-')
        print('avg, std:', param_array.mean(), param_array.std())
        print('chi-square:', scipy.stats.chisquare(numpy.histogram(param_array, numpy.unique(param_array))[0]))
        # ax_hist.hist(param_array, numpy.unique(param_array))
        ax_hist.hist(param_array, numpy.unique(param_array).size)
        ax_corr.plot(numpy.correlate(param_array, param_array, 'full')[param_array.size-1:])

        smoothed_param_array = smooth(param_array, smoothed, 'flat')
        ax.plot(smoothed_param_array)
        ax_hist.hist(smoothed_param_array, 10)
        print('smoothed avg, std:', smoothed_param_array.mean(), smoothed_param_array.std())

        ax2.plot(chi_p, 'o-')
        ax2.plot(numpy.ones(chi_p.size)*p, ':', linewidth=2)


        # huebsch machen
        for a in [ax, ax_hist, ax_corr, ax2]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel('#')
        ax.set_ylabel(key)
        ax.set_xlim(0, param_array.size)

        ax_hist.set_xlabel(key)
        ax_corr.set_ylabel('auto correlation')

        ax2.set_xlabel('#')
        ax2.set_ylabel('p for uniform distribution')
        ax2.set_xlim(0, param_array.size)
        ax2.set_ylim(0, 1)

        return chi_ok


    def throw_out_trials(self, trial_indices, display=True):
        """ Throws out trials.

        Parameters
        -----------
        trial_indices : int or array_like
            Indices of the trials to be thrown out.
        """
        trial_indices = numpy.array(trial_indices, ndmin=1)
        if trial_indices.size:
            if display:
                print(str(self.__class__) + ': Throwing out', trial_indices.size, 'trials.')
            self.parameters = numpy.delete(self.parameters, trial_indices).tolist()
            self.times = numpy.delete(self.times, trial_indices)


class paramsTrajectoryList(object):

    def __init__(self):
        self.timeUnit = 's'
        self.parameter_trajectories = {}

    def plotParameter(self, key, fig=None, ax=None):

        if not fig:
            fig = pl.figure()
        if not ax:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

        param_array = []

        for p in self.parameter_trajectories.values():
            param_array.extend(p.getParameter(key))
        ax.plot(param_array, '.-')
        if hasattr(self, 'times'):
            ax2.plot(self.times, param_array, '.-')

        # huebsch machen
        custom_plot.huebschMachen(ax)
        ax.set_ylabel(key)
        ax.set_xlabel('#')
        
        custom_plot.huebschMachen(ax2)
        ax2.set_ylabel(key)
        ax2.set_xlabel('Time ('+self.timeUnit+')')

        return fig, ax

    def plotParameterDistribution(self, key, fig=None, ax=None, showFigs=True, saveFig=False, saveFolder=''):
        """ Plots distributions for parameter.

        One of the parameter distribution and one of the transition distribution across trials.

        Parameters
        ----------
        key : str
            Keyword of parameter in dictionary.
        fig : matplotlib Figure
        ax : matplotlib Axes

        Returns
        -------
        fig : matplotlib Figure
        ax_top : matplotlib Axes
        ax_bottom : matplotlib Axes
        """

        param_array = []
        for p in self.parameter_trajectories.values():
            param_array.extend(p.getParameter(key))


        # plotting absolute and transition parameter distributions as histograms
        if not fig:
            fig = pl.figure(figsize=(20, 7))
        if not ax:
            ax_top = fig.add_subplot(211)
            ax_bottom = fig.add_subplot(212)

        bins = numpy.unique(param_array)
        if bins.size < 10:
            dbin = numpy.mean(numpy.diff(bins))
            bins = numpy.insert(bins, [0, bins.size], [bins[0]-dbin, bins[-1]+dbin])
        else:
            bins = 10
        print(bins)
        ax_top.hist(param_array, bins=bins, align='left')

        # huebsch machen
        custom_plot.huebschMachen(ax_top)
        for ax in [ax_top, ax_bottom]:
            for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
        ax_top.set_title('Absolute and transition %s distribution' %key)
        ax_top.set_xlabel(key)
        ax_top.set_ylabel('Value count')

        # transitions
        # doubling the all parameters except the first and last one, to analyse parameter transitions
        doubled_params = [str(param_array[0])]
        for i in numpy.arange(1, len(param_array)-1):
            doubled_params.extend([str(param_array[i]), str(param_array[i])])
        doubled_params.append(str(param_array[len(param_array)-1]))
        param_transitions = [(', '.join(map(str, doubled_params[n:n+2]))).replace(",", " ->")
                            for n in range(0, len(doubled_params), 2)]
        self.param_transitions_dict = dict((x, param_transitions.count(x)) for x in param_transitions)

        ax_bottom.bar(*zip(*zip(count(), self.param_transitions_dict.values())))

        # huebsch machen
        custom_plot.huebschMachen(ax_bottom)
        ax_bottom.set_xticks((1*zip(*zip(count(0.4), self.param_transitions_dict)))[0])
        ax_bottom.set_xticklabels((1*zip(*zip(count(0.4), self.param_transitions_dict)))[1])
        ax_bottom.set_xlabel('%s transition' %key)
        ax_bottom.set_ylabel('Value count')

        if saveFig:
            print('Parameter Distribution plot saved to:', saveFolder+'Parameter_Distribution.png')
            fig.savefig(saveFolder+'Parameter_Distribution.png', format='png')
        if not showFigs:
                pl.ioff()
                pl.close(fig)
        else:
            pl.show()

        return fig, ax_top, ax_bottom

    def plotParam1vsParam2(self, key1, key2, fig=None, ax=None, refline=False):
        """Plot two parameters against each other.

        Parameters
        -----------
        key1 : str
            Name of one parameter.
        key2 : str
            Name of other parameter.
        fig : matplotlib figure, optional
        ax : matplotlib axes, optional
        refline : bool, optional
            Draw reference line.

        Returns
        -------
        (fig, ax), (fig1, ax1, ax1b) : two tuples
            Tuples with Figure and Axes handles.
        """

        if not fig:
            fig = pl.figure()
            fig1 = pl.figure()
        if not ax:
            ax = fig.add_subplot(111)
            ax1 = fig1.add_subplot(111)

        param_array1 = []
        param_array2 = []
        for p in self.parameter_trajectories.values():
            param_array1.extend(p.getParameter(key1))
            param_array2.extend(p.getParameter(key2))
        param_array1 = numpy.array(param_array1)
        param_array2 = numpy.array(param_array2)

        mini = min(param_array1.min(), param_array2.min())
        maxi = max(param_array1.max(), param_array2.max())
        if refline:
            ax.plot([mini, maxi], [mini, maxi], '--', color=numpy.ones(3)*.5, linewidth=1)
        ax.plot(param_array1, param_array2, 'bo', markerfacecolor='none', markeredgecolor='b', alpha=.5)
        ax1.plot(param_array1, 'bo-', linewidth=1)
        ax1b = ax1.twinx()
        ax1b.plot(param_array2, 'go-', linewidth=1)

        # huebsch machen
        custom_plot.turnOffAxes(ax1b, ['top'])
        ax1.set_ylabel(key1)
        ax1b.set_ylabel(key2)

        for a in [ax]:
            custom_plot.huebschMachen(a)
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_xlim(ax.get_xlim()*numpy.array([.95, 1.05]))
        ax.set_ylim(ax.get_ylim()*numpy.array([.95, 1.05]))

        return (fig, ax), (fig1, ax1, ax1b)

    def id_list(self):
        """ Return the list of all the ids in the parameter trajectory.
        """

        return numpy.array(self.parameter_trajectories.keys())

    def __getitem__(self, id):
        if id in self.id_list():
            return self.parameter_trajectories[id]
        else:
            raise Exception("id %d is not present in the paramsList. See id_list()" % id)

    def __setitem__(self, i, val):
        assert isinstance(val, paramsTrajectory), "An paramsTrajectoryList object can only contain paramsTrajectory objects"
        self.parameter_trajectories[i] = val

    def __iter__(self):
        return self.parameter_trajectories.itervalues()

    def __len__(self):
        return len(self.parameter_trajectories)

    def append(self, signal):
        """
        Add an paramsTrajectory object to the paramsTrajectoryList

        Parameters
        ----------
        signal : paramsTrajectory object
            The paramsTrajectory object to be appended.

        See also
        --------
        __setitem__
        """

        assert isinstance(signal, paramsTrajectory), "An paramsTrajectoryList object can only contain paramsTrajectory objects"
        self[self.parameter_trajectories.__len__()] = signal

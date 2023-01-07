#SORT
#=============================
#=============================
    

#=============================
def list_series(length, num):
#=============================
    
    """
    This function creates a series of empty lists of the same dimension.
    
    
    Inputs:
        length (int): length of each empty list
        num (int): number of lists
        
    returns:
        out_l (list of list): list of list
    
    """
    
    out_l = [0]*num

    for x,n in enumerate(range(num)):
        out_l[x] = [0]*length
    return(out_l)



#=============================
def h5_2dict(data):
#=============================
    """
    This function converts h5 files into a dictionary by looping through all keys. 
    
    
    Inputs:
        data (h5): h5 file
        
    returns:
        d (dict): dictionary
    
    """
    
    import h5py
    import numpy as np
    h5read = h5py.File(data, 'r')
    par_l = np.array(h5read)
    d = {}
    for i in par_l:
        d.update({i: np.array(h5read[i])})
        
    return(d)




#=========================================
def return_files(path, experiment, search):
#=========================================

    """
    Return list of files in defined path, and changes current working directory to path containing desired files
    
    Inputs:
    path (string): name of parent path
    experiment (string): name of experiment folder 
    search (string): words that files must contain
    
    Returns:
    data_list (list): list containing all files of interest
    
    """

    import os
    import glob
    os.chdir(path + experiment)
    data_list = sorted(glob.glob(search))
    return(data_list)


#=============================
def name_zero(pad, start, end, step): 
#=============================
    """
    Adds zero to front or back of a list of numbers - useful for saving filenames with numbers
    
    Inputs:
    pad (string): 'front' or 'back'
    start (int): number to start from
    end (int): number to end wtih
    step (int): stepsize
    
    Returns:
    listme (list): a list of strings with 0s appended
    
    """
    
    
    import os 
    import numpy as np
    
    if pad == 'front': 
        count = 0
        listme = list(range(start, end+1, step))
        for i in range(start, end+1, step):
            if i < 10:
                num = '0' + str(i) #add 0 to front if integer value less than 10
            elif i >9:
                num = str(i) #else do not add 0
            listme[count] = num
            count+=1
        return(listme)

    if pad == 'back': 
        count, count1 = 0,0
        looplist = np.arange(start, end + step, step)
        listme = list(range(0, looplist.shape[0]))
        lenlist = list(range(looplist.shape[0]))
        for i in looplist:
            lenlist[count1] = len(str(round(i, len(str(step)))))
            count1 +=1
        for i in looplist:
            if len(str(round(i,len(str(step))))) < np.max(lenlist):
                num = str(round(i,len(str(step)))) + '0'
            else:
                num = str(round(i,len(str(step))))
            listme[count] = num
            count+=1
        return(listme)

#=============================
def repeat_list(name, length): 
#=============================  
    """
    Creates list of the same string repeated n times
    
    Inputs:
    name (string): string to repeat
    length (int): length of list
    
    Returns:
    itlist (list): a list of repeated string
    
    """
    
    itlist = list(range(length))
    for i in range(len(itlist)):
        itlist[i] = name
    return(itlist)


#==============================
def save_name(name): 
#===============================
    """
    Creates name template for saving - requires standardised input format
    
    Inputs:
    name (string): full input string
    
    Returns:
    (string): name template
    
    """

    return(name[:name.find('run')+6])


#=======================================================================================
def comb_list(inp_list):
#=======================================================================================

    """
    This function takes a series of lists and combines them into one list. 
    
    Inputs:
        inp_list (list): input list
        
    Returns:
        out_list (list): output list

    """    

    #Find total length
    sumd=0
    for i in range(len(inp_list)):
        for e in inp_list[i]:
            sumd+=1
    

    out_list = list(range(sumd))
    count=0
    for i in range(len(inp_list)):
        for e in inp_list[i]:
            out_list[count] = e
            count+=1
        
    return(out_list)


#=======================================================================================
def cond_list(inp_list, cond_list, mode):
#=======================================================================================

    """
    This function takes an input list and iterates over a condition list by some rule, to label the input list by its condition. 
    
    Inputs:
        inp_list (list of lists): input list of lists
        cond_list (list): condition list, e.g. colours, plotting styles, labels etc
        mode (str): 'dataset' orders condition list by dataset, 'datapoint' orders the condition list by data point
    Returns:
        out_list (list): output list

    """    
    #check that cond_list is correct shape
    if mode == 'dataset' and len(cond_list) != len(inp_list):
        print('Number of colours does not match number of datasets')
        return()
    
    if mode == 'datapoint' and len(cond_list) != len(comb_list(inp_list))/len(inp_list):
        print('Number of colours does not match number of datapoints')
        return()

    #Find total length
    sumd=0
    for i in range(len(inp_list)):
        for e in inp_list[i]:
            sumd+=1
    

    out_list = list(range(sumd))
    count=0
    for i in range(len(inp_list)):
        for e in range(len(inp_list[i])):
            if mode == 'dataset':
                out_list[count] = cond_list[i]
                
            elif mode == 'datapoint':
                out_list[count] = cond_list[e]
            count+=1
        
    return(out_list)


#=======================================================================================
def load_list(inp_list):
#=======================================================================================
    """
    This function takes an input a list of file names and loads them into a list
    
    Inputs:
        inp_list (list of strings): input list of files names

    Returns:
        out_list (list of np arrays): output list

    """    
    import numpy as np
    
    out_list = list(range(len(inp_list)))
    for i in range(len(inp_list)):
        out_list[i] = np.load(inp_list[i])
    return(out_list)


#PROCESS
#=============================
#==============================

#===============================
def par_save_name(name, par):
#===============================

    
    """
    This function saves name with a parameter, placing it before run.
    """
    
    pref = name[:name.find('run')]
    run = name[name.find('run'):name.find('run')+6]
    return(pref + par + run)


#================================================
def select_region(trace, coord, region):
#================================================
    
    """
    This function slices data to include only those within a specific brain region.

    Inputs:
        trace (np array): cells x timepoints, raw or normalised fluorescence values
        coord (np array): cells x XYZ coordinates and labels
        region (str): 'all', 'Diencephalon', 'Midbrain', 'Hindbrain' or 'Telencephalon'
    
    Returns:
        sub_trace (np array): cells x timepoints, raw or normalised fluorescence values for subregion
        sub_coord (np array): cells x XYZ coordinates for subregion
    
    """
    
    import numpy as np

    if coord.shape[0] != trace.shape[0]:
        print('Trace and coordinate data not same shape')
        return()


    if region == 'all':
        locs = np.where(coord[:,4] != 'nan')

    else: 
        locs = np.where(coord[:,4] == region)

    sub_coord = coord[locs]

    sub_trace = trace[locs]


    return(sub_trace,sub_coord)

#============================================
def save_shared_files(path, son_path, mode):
#============================================
    """
    Saves shared modules across different repositories
    
    Inputs:
    path (string): name of parent path - should be Fcode
    son_path (string): name of code folder containing module which you have just edited
    mode (string): define which module to save: 'admin', 'criticality', or 'trace'
    
    """

    import os
    from shutil import copyfile
    
    def loop_dir(file_list, path_list):
        """
        Loop between directories and save file in all but current 

        """

        #Loop through files and directories
        for x in file_list:
            for e,i in enumerate(path_list):
                if path_list[e] not in os. getcwd() and not path_list[e].startswith('.'): #skip current directory
                    copyfile(x, path + i + os.sep + x) #copy and overwrite files in directory

    if mode == 'admin':
        file_list = return_files(path , son_path, 'admin_functions.py' ) #search for admin file in current directory
        path_list = os.listdir(path) #get names of all directories


    if mode == 'criticality':
        file_list = [return_files(path , son_path, 'criticality.py')[0], return_files(path , son_path, 'IS.py')[0], return_files(path, son_path, 'trace_analyse.py')[0]]  #search for admin file in current directory
        path_list = ['criticality', 'avalanche_model', 'mutant_analysis'] #CHANGE AS NEEDED!
        
    if mode == 'trace':
        file_list = return_files(path , son_path, 'trace_analyse.py' ) #search for trace_analyse file in current directory
        path_list = ['criticality', 'avalanche_model', 'mutant_analysis'] #CHANGE AS NEEDED!
        
        
    loop_dir(file_list, path_list) 


#=====================================================================
def parallel_func(cores, savepath, iter_list, func, param_list, name, variables, mode): 
#=====================================================================
    """
    This function allows parallel pooling of processes using functions
    
    Inputs:
    cores = number of cores 
    savepath = path for saving
    iter_list = list with parameter inputs that you will parallel process (inputs must be at start of function)
    func = function name
    param_list = list containing remaining function parameters 
    name = filename for saving, should be unique if mode = save_group
    variables = list containing name endings for each variable, if function returns multiple
    mode = output type:
        save_single - saves each variable of function output individually
        save_group - saves all batched function outputs in a list
        NA - returns all batched function outputs in a list, without saving
    """
    
    from multiprocessing import Pool
    import numpy as np
    pool = Pool(cores) #number of cores
    count = 0

    batch_list = list(range((np.int(len(iter_list)/cores)))) #define number of batches

    for i in range(len(batch_list)): #process each batch
        cores_inputs = list(range(cores)) #define input for each core
        for e in range(len(cores_inputs)):  
            sub_iter_list = iter_list[count:count+1] #Find current iter value - add to subset iter_list
            sub_iter_list.extend(param_list) #Append current iter value onto remaining parameter
            cores_inputs[e] = sub_iter_list 
            count+=1
        batch_list[i] = pool.starmap(func, cores_inputs) #pool process on each core

        if mode == 'save_single':
            for t in range(cores):  #loop through each core in current loop
                for f in range(len(batch_list[i][t])):
                    save_var = batch_list[i][t][f] #function output for current core in current batch
                    save_name = name + '-' + str(cores_inputs[t][0]) + '-' + variables #save name based on iterable parameter
                    np.save(savepath + save_name, save_var)
        
    if mode != 'save_single':
        #Append all calculated value together
        if isinstance(batch_list[0][0], int) or isinstance(batch_list[0][0], float) :
            return_me = np.hstack(np.array(batch_list))
        else:
            return_list = list(range(len(batch_list[0][0])))
            new_array = np.vstack(np.array(batch_list))
            return_me = [new_array[:,i] for i in range(new_array.shape[1])]

        if mode == 'save_group':
            save_name = name
            np.save(savepath + save_name, return_me)

        else:
            return(return_me)
      
    
#=====================================================================
def parallel_class(cores, savepath, iter_list, func, param_list, name, variables, mode): 
#=====================================================================
    """This function allows parallel pooling of processes using classes
    
    Inputs:
    cores = number of cores 
    savepath = path for saving
    iter_list = list with parameter inputs that you will parallel process (inputs must be at start of function)
    func = function name
    param_list = list containing remaining function parameters 
    name = filename for saving, should be unique if mode = save_group
    variables = list containing name endings for each variable, if function returns multiple
    mode = output type:
        save_single - saves each variable of function output individually
        save_group - saves all batched function outputs in a list
        NA - returns all batched function outputs in a list, without saving
    """
    
    from multiprocessing import Pool
    import numpy as np
    pool = Pool(cores) #number of cores
    count = 0

    batch_list = list(range((np.int(len(iter_list)/cores)))) #define number of batches
    for i in range(len(batch_list)): #process each batch
        cores_inputs = list(range(cores)) #define input for each core
        for e in range(len(cores_inputs)):  
            sub_iter_list = iter_list[count:count+1] #Find current iter value - add to subset iter_list
            sub_iter_list.extend(param_list) #Append current iter value onto remaining parameter
            cores_inputs[e] = sub_iter_list 
            count+=1
        batch_list[i] = pool.starmap(func, cores_inputs) #pool process on each core
        
        

        
        if mode == 'save_single':
            import scipy.sparse
            for t in range(cores):  #loop through each core in current loop
                for s in range(len(variables)):
                    save_var = batch_list[i][t].__dict__[variables[s]] #function output for current core in current batch
                    save_name = name + '-' + str(cores_inputs[t][0]) + '-' + variables[s] #save name based on iterable parameter
                    sparse_matrix = scipy.sparse.csc_matrix(save_var)
                    scipy.sparse.save_npz(savepath + save_name, sparse_matrix)
                    #np.save(savepath + save_name, save_var)

    if mode != 'save_single':
    
        #Append all calculated values together
        if len(variables) == 1:
            if isinstance(batch_list[0][0].__dict__[variables[0]], int) or isinstance(batch_list[0][0].__dict__[variables[0]], float):
                count=0
                return_me = list(range(len(iter_list)))
                for first in range(len(batch_list)):
                    for second in range(len(batch_list[0])):
                        return_me[count] = batch_list[first][second].__dict__[variables[0]]
                        count+=1
            else:            
                count=0
                return_me = list_of_list(len(variables),len(iter_list))
                for first in range(len(batch_list)):
                    for second in range(len(batch_list[0])):
                        for third in range(len(variables)):
                            return_me[third][count] = batch_list[first][second].__dict__[variables[third]]
                        count+=1       

        if len(variables) > 1:
            count=0
            return_me = list_of_list(len(variables),len(iter_list))
            for first in range(len(batch_list)):
                for second in range(len(batch_list[0])):
                    for third in range(len(variables)):
                        return_me[third][count] = batch_list[first][second].__dict__[variables[third]]
                    count+=1
        
        if mode == 'save_group':
            save_name = name
            np.save(savepath + save_name, return_me)
        
        else:
            return(return_me)
    
        
#=======================================================================================        
def timeprint(per, r, numrows, name):
#=======================================================================================
    """ 
    Print current time step every percentile
    
    Inputs:
    per = how often you want to print (as percentiles)
    r = current iterator value
    numrows = total number of steps
    name = name to output
    """
    if r % round((per*numrows/100)) == 0: 
            print("Doing number " + str(r) + " of " + str(numrows) + " for " + name)
            
            
#MATHS
#=============================
#=============================

#==============================================
def autocorr(data, length):
#==============================================
    """
    This function calculates the autocorrelation of a timeseries against itself over successive delays. 
    
    Inputs:
        data (np array): 1d vector timeseries
        length (int): how many delays to calculate over
    
    Returns:
        1d vector of correlation values of data_t against data_ti
    
    """
    import numpy as np
        
    return np.array([1]+[np.corrcoef(data[:-i], data[i:])[0,1]  \
        for i in range(1, length)])
    


#=======================================================================================
def window(size, times): #make window of given size that is divisible by time series
#=======================================================================================
    """
    Returns the window size that is the closest divisor of a timeseries to given input
    
    Inputs:
    size (int): ideal window size
    times(int): overall trace length
    
    Returns: 
    size (int): window size that is divisible by trace (rounds up)
    n_windows (int): number of windows that split up trace
    """
    for i in range(times):
        if times % size ==0:
            break
        else:
            size+=1
    n_windows = int(times/size)
    return(size, n_windows)

#=======================================================================================
def mean_std(label, data):
#=======================================================================================
    """
    Prints the mean and standard deviation.
    
    Inputs:
    label (str): dataset label
    data (np array/list/dataframe): row of data

    """
    import numpy as np
    from scipy import stats
    mean = np.mean(data)
    sem = stats.sem(data)
    print(label + " mean = " + str(mean) + '  , std = ' + str(sem))

#=======================================================================================
def stats_2samp(data1, data2, alpha, n_comp, mode):
#=======================================================================================
    """
    Performs significance test on 2 sample data. 
    
    Inputs:
    data1 (np array/list/dataframe): row of dataset 1
    data2 (np array/list/dataframe): row of dataset 2
    alpha (float): significant level
    n_comp (int): number of comparisons for bonferroni correction
    mode (str): 'ind' for independent samples, 'rel' for related samples

    Outputs:
     (float): test statistic
    p (float): p value
    """

    from scipy import stats
    
    def print_sig(t,p,a):
        if p > a:
            print('Samples are the same')
        else:
            print('Samples are significantly different')
    
    corrected_alpha = alpha/n_comp
    if len(data1) >7 and len(data2) > 7:
        p1, p2 = stats.normaltest(data1)[1], stats.normaltest(data2)[1]
        if p1 or p2 < alpha:
            normal = 'no'
        else:
            normal = 'yes'
    
    else: #if you have less than 8 samples, use non-parametric test
        normal = 'no'
        
    
    if normal == 'no':
        print('At least one sample is non-Gaussian - performing non-parametric test')
        
        if mode == 'ind':
            U, p = stats.mannwhitneyu(data1, data2)
            print_sig(U,p,corrected_alpha)
            print('U = ' + str(U) +  '   p = ' + str(p))
            return(U,p)
            
        elif mode == 'rel':
            w, p = stats.wilcoxon(data1, data2)
            print_sig(w,p,corrected_alpha)
            print('w = ' + str(w) +  '   p = ' + str(p))
            return(w,p)
        
            
    elif normal == 'yes':
        print('Both samples are Gaussian - performing parametric test')
    
        if mode == 'ind':
            t, p = stats.ttest_ind(data1, data2)
            print_sig(t,p,corrected_alpha)
            
        elif mode == 'rel':
            t, p = stats.ttest_rel(data1, data2)
            print_sig(t,p,corrected_alpha)
            
        print('t = ' + str(t) +  '   p = ' + str(p))
        return(t,p)


#=======================================================================
def mean_distribution(distlist): #Generate mean distribution 
#=======================================================================
    import numpy as np
    comb_vec = []
    for i in range(len(distlist)):
        comb_vec = np.append(comb_vec, distlist[i])
    av = np.unique(comb_vec, return_counts=True)[0]
    freq = (np.unique(comb_vec, return_counts=True)[1]).astype(int)//len(distlist)
    mean_vec = []
    for e in range(freq.shape[0]):
        mean_vec = np.append(mean_vec, np.full(freq[e],av[e]))
    return(mean_vec)
        
        
#PLOT
#=============================
#=============================

#=======================================================================================
def multi_plot(data_list, col_list, plot_type, size, rows, cols): 
#=======================================================================================
    """
    Matplotlib confuses me - this function allows me to build a subplot frame without having to remember how to use matplotlib. 
    
    Inputs:
    data_list(list): list of data to plot, must match the method type
    plot_type (str): must be a method available to plot
    size (tuple): fig size
    rows (int): number of rows
    cols (int): number of columns
    col_list (list): list of colors for plotting

    """
    from matplotlib import pyplot as plt
    
    plt.figure(figsize = size)
    
    for i in range(len(data_list)):
        plt.subplot(rows, cols, i + 1)
        plot = getattr(plt, plot_type)(data_list[i], color = col_list[i]) 
    plt.show()
        
        
#=======================================================================================     
def bar_scatter_plot(dic, data_name, fig_size, bar_size, dot_size, colours):
#=======================================================================================
    """
    Plot a bar and scatter plot with mean and individual data points. 
    
    Inputs:
        dic (dict): dictionary of data points
        data_name (str): data name in dictionary
        fig_size (tuple): figure size
        bar_size (float): size of mean bar
        dot_size (float): size of dot
        colours (list): colors of data points

    """
    from matplotlib import pyplot as plt
    import seaborn as sns
    from matplotlib.collections import PathCollection
    from matplotlib import cm
    sns.set(style="white")
    
    

    fig, ax = plt.subplots(figsize = fig_size)
    ax = sns.pointplot(x="condition", y=data_name, data = dic, hue = 'condition', palette = colours, join=True, ci=0, scale=bar_size, markers = '_')
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    ax = sns.stripplot(x="condition", y=data_name, data = dic,hue = 'condition', palette = colours, size = dot_size, jitter = True ,alpha = 1)

    plt.yticks(size = 20)
    points = ax.collections
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend_.remove()
    plt.show()
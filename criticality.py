import admin_functions as adfn
import IS as isfn


#=======================================================================
def neighbour(coord, n_neigh): 
#=======================================================================
    """
    This function calculates the nearest n neighbours for each neuron.
    
    Inputs:
        coord (np array): cells x XYZ coordinates 
        n_neigh (int): number of closest neigbours to find
    
    Returns:
        nnb (np array): cells x cells, with 0s meaning not neighbours and 1s meaning neighbours
    """
    
    
    import numpy as np
    import os
    
    #Loop through all fish
    #----------------------
        
        # Set up nearest neighbour graph
        #---------------------------------------------------------------------------
        
        # Initialise full distance matrix and nearest neighbour graph (binary) matrix
        #nearest neigh binary matrix of celln by celln storing 
        #distance of each cell to every other cell
        #---------------------------------------------------------------------------
    nnb  = np.zeros((coord.shape[0],coord.shape[0]))  
        
    for r in range(coord.shape[0]):
        distance = np.zeros(coord.shape[0])
        #if r % round((10*coord.shape[0]/100)) == 0: 
            #print("Doing row " + str(r) + " of " + str(coord.shape[0]))
        
        for x in range(coord.shape[0]):
            if x == r: 
                distance[x] = 100000
            else:
                distance[x] = np.linalg.norm(coord[r]-coord[x]) 
                
        index = np.argsort(distance)[:n_neigh]
        nnb[r,index] = 1 #binary value defining whether in range or not 
    return(nnb)

    

#=======================================================================
def avalanche(nnb, bind): 
#=======================================================================
    """
    This function calculates the spatiotemporal propagation of spike events - neural avalanches.
    
    Inputs:
        nnb (np array): cells x cells, with 0s meaning not neighbours and 1s meaning neighbours
        bind (np array): cells x time, binarised state vector
    
    Returns:
        av (np array): 2d vector of avalanche sizes and avalanche durations
        pkg (np array): cells x time, with each timepoint marking distinct avalanche events, i.e. each entry represents no avalanche (0) or a specific avalanche event (any integer)
    """
    
    
    import numpy as np
    import os
    import itertools


#Calculate avalanche size + duration
#-----------------------------------
    binarray, oldav, firstav, realav, timemachine, convertav, fill, time = [],[],[],[],[],[],[],[]
    
    #LOOP THROUGH EACH FISH
    #---------------------------------
    #---------------------------------
    binarray, nnbarray, pkg = bind,nnb, np.zeros(bind.shape)
    i, marker, avcount = 0,0,0
        
    #LOOP THROUGH EACH TIME POINT
    #------------------------------
    #------------------------------
    for t in range(binarray.shape[1]-1): #loop through all time points
        i = i+1
        cid = np.where(binarray[:,t] > 0)[0]  #cid = cells active at current time point
    
            
        #LOOP THROUGH EACH ACTIVE CELL
        #-------------------------------
        #-------------------------------
        for c in cid:            #loop through all active cells at this time point

            if pkg[c,t] == 0:    #only find non-marked cells
                if len(np.intersect1d(np.where(nnbarray[c,:] > 0)[0], cid) > 2): #if >2 neighbours active
                    marker = marker + 1  
                    pkg[c,t] = marker  #mark active non-marked cell with new marker value
                       

            #LOCATE ALL NEIGHBOURS
            #----------------------------
            #----------------------------
            neighbour = np.where(nnbarray[c,:] > 0)[0]  #return indeces of current cell neighbours
            neighbouron  = np.intersect1d(cid,neighbour) #indeces of active cells in t, and also neighbours of c
            where0 = np.where(pkg[neighbouron,t] == 0)[0] #neighbours not already part of an avalanche
                
            #CONVERT NEIGHBOURS WHO ARE ALREADY PART OF AN AVALANCHE
            #-------------------------------------------------------
            #-------------------------------------------------------

            if len(where0) < len(neighbouron): #if any cells are already part of another avalanche
                oldav = np.unique(pkg[neighbouron, t]) #all avalanche values from neighbours
                firstav = np.min(oldav[np.where(oldav > 0)])   #minimum avalanche value that is not 0
                    
                    #define which cells we want to combine
                realav =  oldav[np.where(oldav > 0)] #all avalanche values that are not 0
                uniteav = np.where(pkg[:,t]==realav[:,None])[1] #indeces of all cells that need to be connected
                pkg[uniteav,t] = firstav #convert all current cell neighbours and their active neighbours 
                pkg[c,t] = firstav #also convert current cell
                    
                #GO BACK IN TIME AND CONVERT
                #----------------------------
                #----------------------------
                stop = 30 #value to stop counting back to for minimising compute time
                convertav = realav[1:] #avalanche numbers needing to be converted
                if t < stop:
                    time = t-1
                
                if t > stop - 1:
                    time = stop
                        
                for e in range(convertav.shape[0]):
                    for timemachine in range(1, time): #loop through max possible time of previous avalanche
                        fill = np.where(pkg[:,t-timemachine] == convertav[e])[0]
                        if fill.shape[0] > 0:
                            pkg[fill, t-timemachine] = firstav 
                                    
            
            #CONVERT NEIGHBOURS WHO ARE NOT PART OF AN AVALANCHE
            #-------------------------------------------------------
            #-------------------------------------------------------
            if len(where0) == len(neighbouron): #if all cells are not part of an avalanche
                pkg[neighbouron[where0],t] = pkg[c,t]  

            
        #SEE IF AVALANCHE CAN PROPAGATE TO NEXT TIME FRAME
        #-------------------------------------------------------
        #-------------------------------------------------------
        n_av = np.unique(pkg[:,t])  #returns the marker values for each avalanche at this time point
    
        for n in n_av: #loop through each avalanche in this time point
            if n > 0:
                cgroup = np.where(pkg[:,t] == n)[0] #cells that are in same avalanche at t
                cid2 = np.where(binarray[:,t+1] > 0) #cells in next time point that are active
                intersect = np.intersect1d(cgroup, cid2) #check if any of the same cells are active in next time point
                wherealso0 = np.where(pkg[intersect,t+1] == 0)[0] #here we find all cells that are active in both time frames, and that are not already part of another avalanche - and mark them as current avalanche
                pkg[intersect[wherealso0], t+1] = pkg[cgroup[0],t] #carry over value to next frame for those cells
      
    allmark = np.unique(pkg)[1:] #all unique marker values

    #CALCULATE AVALANCHE SIZE
    #-------------------------------------------------------
    #-------------------------------------------------------
    avsize = np.unique(pkg, return_counts = True)[1][1:] #return counts for each unique avalanche
    frameslist = np.zeros(avsize.shape[0]) #create empty frames list of same length

    #CALCULATE AVALANCHE DURATION
    #-------------------------------------------------------
    #-------------------------------------------------------
    avpertimelist = list(range(pkg.shape[1])) #empty list of length time frames

    for e in range(pkg.shape[1]): #loop through each time point in pkg
            avpertime = np.unique(pkg[:,e]) #unique marker value in each time point
            avpertimelist[e] = avpertime #fill list of unique values in each time point
                          
    #link entire recording together
    #-----------------------------------------------------------
    linktime = list(itertools.chain(*avpertimelist)) #vector of all unique marker values in each time bin linked together
    framesvec = np.unique(linktime, return_counts = True)[1][1:] #vector of number of frames for each consecutive avalanche

    #COMBINE AV SIZE AND DURATION INTO ONE ARRAY
    #-------------------------------------------------------
    #-------------------------------------------------------
    avsizecut = avsize[avsize >= 3]  #only select avalanches above 2
    avframescut = framesvec[avsize >=3]
    av = np.vstack((avsizecut, avframescut))      
    return(av, pkg)



#=======================================================================
def power_exponent(data, npart):
#=======================================================================
    """
    Calculates the power law exponent using max likelihood. 
    
    Inputs:
        data (np array): 2d vector of avalanche sizes and avalanche durations
        npart (int): number of samples to draw
        
    Returns:
        size_exp (float): exponent for avalanche size 
        dur_exp (float): exponent for avalanche duration
        
    
    """


    import numpy as np
    #Size
    sizes=data[0,:]
    M=len(sizes)
    a=min(sizes) #define xmin
    b=max(sizes) #define xmax
    size_exp=isfn.IS(npart, sizes, M, a, b)[0]
    
    
    #Dur
    sizes=data[1,:]
    a=2 #define xmin
    b=max(sizes) #define xmax
    M=len(sizes[np.where(sizes>a-1)])
    dur_exp=isfn.IS(npart, sizes, M, a, b)[0]
        
    return(size_exp, dur_exp)
    
#=======================================================================
def LLR(data, npart):
#=======================================================================
    """
    Calculates the loglikelihood ratio for power law vs lognormal. 
    
    Inputs:
        data (np array): 2d vector of avalanche sizes and avalanche durations
        npart (int): number of samples to draw
        
    Returns:
        size_llr (float): log likelihood ratio for size 
        dur_llr (float): log likelihood ratio for dur
        
    
    """


    import numpy as np
    #Size
    sizes=data[0,:]
    M=len(sizes)
    a=min(sizes) #define xmin
    b=max(sizes) #define xmax
    size_ln=isfn.IS_LN(npart, sizes, M, a, b)
    size_po=isfn.IS(npart, sizes, M, a, b)
    
    size_llr = size_po[1] - size_ln[2]
    
    #Dur
    sizes=data[1,:]
    a=2 #define xmin
    b=max(sizes) #define xmax
    M=len(sizes[np.where(sizes>a-1)])
    dur_ln=isfn.IS_LN(npart, sizes, M, a, b)
    dur_po=isfn.IS(npart, sizes, M, a, b)
    dur_llr = dur_po[1] - dur_ln[2]
    
    
    return(size_llr, dur_llr)


#=======================================================================
def DCC(av):
#=======================================================================
    """
    Calculates the deviation from criticality coefficient, a measure of exponent relation. DCC is calculated by predicting B from critical relationship between exponents, and measuring B from avalanche size vs duration. DCC is then the difference between predicted and fitted B.
    
    Inputs:
        av (np array): 2d vector of avalanche sizes and avalanche durations
        
    Returns:
        dcc (float): DCC value

    """
    from matplotlib import pyplot as plt
    import numpy as np
    av_size = av[0]
    av_dur = av[1]
    size_e, dur_e = power_exponent(av, 2000)
    fig, axarr = plt.subplots(figsize = (7,5))
    av_size = av_size
    av_dur = (1/2.73)*av_dur

    size_vec, dur_vec = [],[]
    for e in np.unique(av_dur):
        size_vec = np.append(size_vec, np.mean(av_size[np.where(av_dur == e)])) 
        dur_vec = np.append(dur_vec, e)

    xaxis = np.unique(dur_vec)
    yaxis = size_vec
    axarr.plot(xaxis[:len(xaxis)-1], yaxis[:len(xaxis)-1], '-', linewidth = 1.5, alpha = 1)
    fit_beta,c = np.polyfit(np.log10(xaxis[:len(xaxis)-1]), np.log10(yaxis[:len(xaxis)-1]), 1)
    plt.close(fig)
    
    pred_beta = (dur_e - 1)/(size_e - 1)
    dcc = np.abs(fit_beta - pred_beta)
    return(dcc)





#=======================================================================
def branch(pkg, av): 
#=======================================================================
    """
    Calculate branching ratio, by iterating through each avalanche event and finding mean descendants/ancestor at each time step. 
    
    Inputs:
        
        av (np array): 2d vector of avalanche sizes and avalanche durations
        pkg (np array): cells x time, with each timepoint marking distinct avalanche events, i.e. each entry represents no avalanche (0) or a specific avalanche event (any integer)
    
    Returns:
        branchmean (float): mean branching ratio
    
    """


    
    import numpy as np
    import os
    branchmean = []
    brancharr = np.zeros((np.int(np.max(pkg)), np.max(av[1])))
    i = 0
        
    for t in range(pkg.shape[1]): #loop through all time points
        if t == pkg.shape[1]-1:
            break
        n1 = np.unique(pkg[:,t])  #unique marker values at each time point
        n2 = np.unique(pkg[:,t+1]) 
        nx = np.intersect1d(n1, n2) #marker values that continue to next time frame
    
        #if i% round(10*pkg.shape[1]/100) == 0: print('doing time step ' + str(i) + ' of ' + str(pkg.shape[1]))
        i = i+1

        for mark in nx[1:]: #loop through each marker value at this time point (only if marker active in next time point)
            if mark == brancharr.shape[0]:
                continue
            mark = np.int(mark)
            ancestor = np.unique(pkg[:,t], return_counts = True)[1][np.where(np.unique(pkg[:,t], return_counts = True)[0] == mark)[0]][0] #number of cells in that avalanche for that marker value at time point t  
            descend = np.unique(pkg[:,t+1], return_counts = True)[1][np.where(np.unique(pkg[:,t+1], return_counts = True)[0] == mark)[0]][0] #same as above for next time point
            brancharr[mark, np.where(brancharr[mark] == 0)[0][0]] = (descend/ancestor)
    branchmean = np.mean(brancharr[np.where(brancharr > 0)])
    return(branchmean)


#=======================================================================
def corrdist(corr, dist, n_bins, mini, maxi):
#=======================================================================
    """
    This function calculates the correlation function of a matrix - this is the mean correlation as a function of distance across pairs of neurons. It does this by binning the data by distance and calculating the mean distance per bin. 
    
    Inputs:
        corr (np array): cells x cells, correlation matrix
        dist (np array): cells v cells, distance matrix
        n_bins (int): number of bins to use
        mini (int): first bin
        maxi (int): last bin
    
    Returns:
        output (np array): 2d vector of mean values for distance and correlation across each bin
    """
    
    import numpy as np
    if corr.shape[0] != dist.shape[0]:
        print('Correlation and Distance matrices have unequal cell numbers')
        return()
    
    #Define the bins
    bins = np.linspace(mini, maxi, n_bins) #Majority unused - may need to sort out spacing? 
    
    # Take upper triangular of matrix and flatten into vector
    corr = np.triu(corr, k=0) 
    dist = np.triu(dist, k=0)
    corr_v = corr.flatten()
    dist_v = dist.flatten()

    # Convert all negative correlations to 0
    corr_v = [0 if o < 0 else o for o in corr_v]
    corr_v = np.array(corr_v)
    dist_v[np.where(corr_v == 0)] = 0 #Convert all negative correlations to 0s in distance matrix

    # Order by distances
    unq = np.unique(dist_v)
    dist_vs = np.sort(dist_v)
    corr_vs = np.array([x for _,x in sorted(zip(dist_v,corr_v))])

    # Remove all 0 distance values = negative correlations and self-correlation
    dist_ = dist_vs[len(np.where(dist_vs == 0)[0]):]
    corr_ = corr_vs[len(np.where(dist_vs == 0)[0]):]
    
    #Bin distances
    bin_ind = np.digitize(dist_, bins)
    
    #Loop through each bin and calculate mean correlation
    output = np.zeros(n_bins), np.zeros(n_bins) 
    for i in range(len(bins)):
        output[0][i] =  bins[i]  #Distance bin

        output[1][i] = np.mean(corr_[bin_ind == i])   #Mean correlation
    
    return(output)

#=======================================================================
def mean_av(data_l, bins, choose):
#=======================================================================
    """
    This function takes a list of avalanche files and finds the average histogram for the distribution across all files. 
    
    Inputs:
        data_l (list of str): list of files to group together
        bins (int): number of bins
        choose (str): 'size' or 'dur'

        
    Returns:
        yaxis (list): list of each yaxis bin - probability
        xaxis (list): list of each xaxis bin - avalanches

    """
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt

    
    if choose == 'size':
        num = 0
    if choose == 'dur':
        num = 1
    #Load all av data in a list
    dist_l = [np.load(data_l[i], allow_pickle=True).item()['av'][num] for i in range(len(data_l))]
    av_l = []
    #Append all together
    for i in range(len(dist_l)): av_l = np.append(av_l, dist_l[i]) 
        
    hist_l = list(range(len(dist_l)))
    #Find max and min for binning
    mini, maxi = np.min(av_l), np.max(av_l)
    fig, axarr = plt.subplots(figsize = (7,5))

    
    #Put each into histogram with same binning
    for i in range(len(dist_l)):    
        hist_l[i] = axarr.hist(dist_l[i], bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 3, cumulative=-1, color = 'k')[0]
    yaxis = axarr.hist(dist_l[0], bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 3, cumulative=-1, color = 'k')[1][:bins]
    plt.close(fig) 
    
    #Find mean across xbins
    xaxis = np.mean(hist_l, axis= 0)
    return(yaxis, xaxis)
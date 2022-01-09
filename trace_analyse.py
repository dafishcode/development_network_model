import criticality as crfn

#CHECK
#================================    
class trace_analyse: 
#================================    
    """
    Class to analyse trace datasets. 
    
    """
    
    #========================
    def __init__(self, name, trace, dff, bind, coord):
    #========================
        self.name = name # dataset name
        self.trace = trace # Raw traces
        self.dff = dff # Normalised fluorescence
        self.bind = bind # Binarised traces
        self.coord = coord # Cell coordinates
        print('Loaded ' + name)

    #====================================
    def criticality_stats(self, n_neigh, n_bins, mini, maxi):
    #====================================
        
        """
        This functions runs all criticality analysis on your data.
        
   
    Inputs:
        n_neigh (int): number of closest neigbours to find
        n_bins (int): number of bins to use for correlation function
        mini (int): first bin
        maxi (int): last bin
    
        """
        import numpy as np
        from sklearn.metrics.pairwise import euclidean_distances

        
        
        self.nnb = crfn.neighbour(self.coord, n_neigh) #Calculate nearest neighbours
        print('Nearest neighbours found')
        
        self.av, self.pkg = crfn.avalanche(self.nnb, self.bind) #Calculate avalanches
        print('Avalanches calculated')
        
        self.llr_s, self.llr_d = crfn.LLR(self.av, 2000) #Calculate loglikelihood ratio
        self.exp_s, self.exp_d = crfn.power_exponent(self.av, 2000) #Calculate power law exponents
        self.dcc = crfn.DCC(self.av) #Calculate exponent relation
        print('Avalanche statistics calculated')
        
        self.br = crfn.branch(self.pkg, self.av) #Calculate branching ratio
        print('Branching ratio calculated')
        
        
        dist = euclidean_distances(self.coord) #Calculate euclidean distance matrix between all cells
        corr = np.corrcoef(self.trace) #Calculate correlation matrix
        self.corrdis = crfn.corrdist(corr, dist, n_bins, mini, maxi)
        print('Correlation function calculated')
        
        return(self)
    
    
    #====================================
    def firing_stats(self, denominator, cutoff):
    #====================================
        
        """
        This functions calculates all firing statistics on data.
        
   
    Inputs:
        denominator (int): denominator to convert into rate
        cutoff (int): threshold for short vs long range correlations in microns

        """
        
        import numpy as np
    
        self.fr = firing_rate(self.bind, denominator) #Calculate firing rates
        print('Firing rate calculated')
    
        self.fa = firing_amp(self.dff, self.bind) #Calculate firing amplitude
        print('Firing amplitude calculated')
        
        self.fd = firing_dur(self.bind) #Calculate firing duration
        print('Firing duration calculated')
        
        self.s_corr, self.l_corr = short_long_corr(self.trace, self.coord, cutoff) #Calculate firing rates
        print('Correlation calculated')
    
        self.dim = linear_dimensionality(np.cov(self.trace)) #Calculate dimensionality
        print('Dimensionality calculated')
        
        return(self)
        
#================================================
def select_region(trace, dff, bind, coord, region):
#================================================
    
    """
    This function slices data to include only those within a specific brain region.

    Inputs:
        trace (np array): cells x timepoints, raw fluorescence values
        dff (np array): cells x timepoints, normalised fluorescence
        bind (np array): cells x time, binarised state vector
        coord (np array): cells x XYZ coordinates and labels
        region (str): 'all', 'Diencephalon', 'Midbrain', 'Hindbrain' or 'Telencephalon'
    
    Returns:
        sub_trace (np array): cells x timepoints, raw or normalised fluorescence values for subregion
        sub_bind (np array): cells x time, binarised state vector for subregion
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

    sub_coord = coord[locs][:,:3].astype(float)
    sub_trace, sub_dff, sub_bind = trace[locs], dff[locs], bind[locs]


    return(sub_trace, sub_dff, sub_bind, sub_coord)



#===============================
def firing_rate(bind, denominator):
#===============================
    """
    This function calculate the median firing rate over all neurons. 
    
    Inputs:
        bind (np array): cells x time, binarised state vector
        denominator (int): denominator to convert into rate
        
    Returns:
        fr (float): median firing rate over all neurons
    
    """
    import numpy as np
    
    fr = np.median(np.sum(bind, axis = 1)/denominator)
    
    return(fr)


#===============================
def firing_amp(dff, bind):
#===============================
    """
    This function calculate the median normalised firing amplitude over all neurons. 
    NB this functions treats each spike as independent. 
    
    Inputs:
        dff (np array): cells x timepoints, normalised fluorescence
        bind (np array): cells x time, binarised state vector
        
    Returns:
        fa (float): median firing amplitude over all neurons
    
    """
    import numpy as np
    
    fa = np.median(dff[bind == 1])
    
    return(fa)


#===============================
def firing_dur(bind):
#===============================
    """
    This function calculate the mean firing event duration across all neurons. 
    
    Inputs:
        bind (np array): cells x time, binarised state vector
        
    Returns:
        fd (float): mean firing event duration over all neurons
    
    """
    import numpy as np
    import more_itertools as mit
    
    n_trans = []
    for i in range(bind.shape[0]): #Loop through each neuron
        si = np.where(bind[i] == 1)[0] #Find spike index
        n_trans = np.append(n_trans,[len(list(group)) for group in mit.consecutive_groups(si)]) #Group continuous values together and find their length
    fd = np.mean(n_trans) 
    
    return(fd)


#===============================
def short_long_corr(trace, coord, cutoff):
#===============================
    """
    This function calculate the median pairwise correlation across all neurons above and below a given distance range. 
    This function ignores all self correlations and negative correlations. 
    
    Inputs:
        trace (np array): cells x timepoints, raw fluorescence values
        coord (np array): cells x XYZ coordinates and labels
        cutoff (int): threshold for short vs long range correlations in microns
        
    Returns:
        corr_s (float): median short range correlation over all neurons
        corr_l (float): median long range correlation over all neurons
    
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    #Short + Long range pairwise Correlation
    dist = euclidean_distances(coord) 
    corr = np.corrcoef(trace)

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

    corr_s = np.median(corr_[dist_ < cutoff])
    corr_l = np.median(corr_[dist_ > cutoff])
    return(corr_s, corr_l)

#===============================
def linear_dimensionality(data):
#===============================
    """
    This function calculate the dimensionality as a measure of the equal/unequal weighting across all eigenvalues.
    
    Inputs:
        data (np array): covariance matrix - make sure this is the correct way around! 
        
    
    Returns:
        dim (float): dimensionality
    
    """
    import numpy as np
    
    v = np.linalg.eigh(data)[0]
    dim = (np.sum(v)**2)/np.sum((v**2))
    
    return(dim)


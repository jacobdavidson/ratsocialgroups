import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx


groupselnames = ['A1','A2','B1','B2']
groupselnamesG = ['G1','G2','G3','G4']
uids = np.array(['GOB', 'OBG', 'OPG', 'ORG', 'RBG', 'ROG', 'RPG', 'GOP', 'OBP', 'OPB',
       'ORP', 'RBP', 'ROP', 'RPO', 'GPB', 'GRB', 'OGB', 'ORB', 'RGB', 'ROB',
       'RPB', 'BGP', 'BOP', 'GBP', 'GRP', 'OGP', 'RBO', 'RGP'])

# see https://seaborn.pydata.org/tutorial/color_palettes.html
groupcolors = np.array(sns.color_palette("tab10"))[[0,9,3,6,5,7,2,8]]
snscolors=sns.color_palette()
snscolors = np.concatenate((snscolors,snscolors,snscolors))

############################################################
##### DATA PROCESSING FUNCTIONS #####
############################################################
def dfdata_to_dfpca(dfinput,dfdegree=None):
    expnums = np.unique(dfinput['Experiment'])
    metrics = np.unique(dfinput['metric'])
    ## make a new data structure, easier to work work.  
    # do it the naiive way, looping through, but thats easier
    df = pd.DataFrame(columns = np.concatenate((['ID','Experiment'],metrics)))
    for uid in uids:
        for expnum in expnums:
            singledata = (dfinput[dfinput['Experiment']==expnum][['metric',uid]].transpose())
            singledata.columns=singledata.loc['metric']
            singledata['ID']=uid
            singledata['Experiment']=expnum
            # df = df.append(singledata.loc[uid])  # old way, now depreceiated in pandas
            df = pd.concat((df,pd.DataFrame(singledata.loc[uid]).T.copy()))
    df.reset_index(drop=True,inplace=True)   
    # convert to float/int
    for m in metrics:
        df[m] = df[m].astype(float)
    # normalize 'FQWhileF' and 'InHomeDay' by subtracting the average during the meas period, because it changed.
    for e in expnums:
        sel = df['Experiment']==e
        for m in ['FQWhileF','InHomeDay']:
            df.loc[sel,m] = df.loc[sel,m]-np.mean(df.loc[sel,m])
                
    # add in- and out-degree, which were calculated from the AA networks, if included
    if not np.all(dfdegree==None):
        df = df.merge(dfdegree,how='left',on=['ID','Experiment'])
        df['Num. contests'] = df['in-degree']+df['out-degree']
        divvalues = df['Num. contests'].values
        divvalues[divvalues==0] = np.nan
        df['Fraction won'] = df['in-degree']/divvalues
    return df

def df_to_dfnetwork(dfsel):
    dfnetwork = dfsel.copy()
    if len(dfsel)>7:
        print('Error: make sure to select Experiment, Group, and DayNight')
    dfnetwork = dfnetwork[['winner', 'loser0','loser1', 'loser2', 'loser3', 'loser4', 'loser5', 'loser6']]
    dfnetwork = dfnetwork.set_index('winner')
    dfnetwork.index.name = ''
    dfnetwork.columns = dfnetwork.index
    return dfnetwork

def get_transitivityindex_ordered(dfa):
    common = np.minimum(dfa,dfa.T)
    dominant = dfa - common    
    Gtemp = dominant.copy()
    Glist = list(Gtemp.columns)
    s1 = []
    s2 = []
    while len(Glist)>0:
        # get in and out degree
        indeg = get_indegree(Gtemp)
        outdeg = get_outdegree(Gtemp)
        # check for a sink (zero outdegree), or source (zero indegree)
        sinkind = np.where(outdeg==0)[0]
        sourceind = np.where(indeg==0)[0]
        if len(sinkind)>0:
            toremove = sinkind[0] # remove the first one.. hopefully doesn't matter if there are multiple
            # if len(sinkind)>1:
            #     print('more than one sink')
            # add to s2
            toremovename = Glist[toremove]
            s2.append(toremovename)
            Glist.remove(toremovename)
            Gtemp = Gtemp.loc[Glist,Glist]
        elif len(sourceind)>0:
            toremove = sourceind[0] # remove the first one.. hopefully doesn't matter if there are multiple
            # if len(sourceind)>1:
            #     print('more than one source')        
            # add to s1
            toremovename = Glist[toremove]
            s1.append(toremovename)
            Glist.remove(toremovename)
            Gtemp = Gtemp.loc[Glist,Glist]
        else:
            # calculate difference in in and out degree
            delta = outdeg - indeg
            toremovename = Glist[np.argmax(delta)]
            # add to s1
            s1.append(toremovename)
            Glist.remove(toremovename)
            Gtemp = Gtemp.loc[Glist,Glist]
    Glist = s1+s2
    dm = dominant.loc[Glist,Glist]
    return np.sum(np.triu(dm))/np.sum(dm.values)

def get_symmetryindex(dfa):
    common = np.minimum(dfa,dfa.T)
    return np.sum(common.values)/np.sum(dfa.values)

def get_reaching_gini(dfa):
    return gini(get_localreaching(dfa))

############################################################
##### NETWORK-RELATED FUNCTIONS #####
############################################################
def get_indegree(df):
    return np.sum(df,axis=1)/(len(df)-1)
def get_outdegree(df):
    return np.sum(df,axis=0)/(len(df)-1)
getG = lambda x: nx.from_pandas_adjacency(x,create_using=nx.DiGraph)

def makediff(df):
    dd = df.copy()
    diff = dd - dd.T
    diff[diff<0]=0
    return diff

def get_localreaching(dfsel):
    # calculate this on the difference, because it should be that way.  
    # Note that if the diff is already calculated, its fine:  makediff(makediff(dfsel))==makediff(dfsel)
    lr = np.array([nx.local_reaching_centrality(getG(makediff(dfsel)),uid,weight='weight') for uid in dfsel.columns])
    lr[(get_indegree(dfsel)==0)&(get_outdegree(dfsel)==0)] = np.nan
    return lr

def getmeancontests(dfa):  # mean number of contests
    # this is an inefficient implementation but it doesn' matter, its conceptually easier because its consistent with the other file
    totaldeg = get_indegree(dfa) + get_outdegree(dfa)
    # remove ones with zero for the average - this only happens when rats are not present in the experiment (died or removed)
    return np.mean(totaldeg[totaldeg>0])

def gini(x):
    # from:  https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)
    # Mean absolute difference
    x = x[np.logical_not(np.isnan(x))]
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def get_symmetryindex_special(dfa,
                              which='rest'  # 'rest' = all others, 'dominant' = the rat with highest reaching cent
                             ):
    cols = dfa.columns
    localreaching = get_localreaching(dfa)
    dind = np.argmax(localreaching)
    new = dfa.copy()
    if which=='rest':
        # set the dominant to zero
        new.loc[cols[dind],:] = 0
        new.loc[:,cols[dind]] = 0
    else:
        # set all others to zero
        new.loc[cols.drop(cols[dind]),cols.drop(cols[dind])] = 0

    # calculate symmetry on the remaining
    common = np.minimum(new,new.T)
    return np.sum(common.values)/np.sum(new.values)

def get_degreefrac_special(dfa,
                              which='dominant',  # 'rest' = all others, 'dominant' = the rat with highest reaching cent
                               rank=0
                             ):
    cols = dfa.columns
    localreaching = get_localreaching(dfa)
    dind = np.nanargmax(localreaching)
    new = dfa.copy()
    if which=='rest':
        # set the dominant to zero
        new.loc[cols[dind],:] = 0
        new.loc[:,cols[dind]] = 0
    elif which=='sub':
        dind = np.argsort(localreaching)[rank]
        new.loc[cols.drop(cols[dind]),cols.drop(cols[dind])] = 0
    elif which=='dominant':
        # set all others to zero
        new.loc[cols.drop(cols[dind]),cols.drop(cols[dind])] = 0

    # calculate degree on the remaining as a fraction
    return np.sum(new.values)/np.sum(dfa.values)

############################################################
##### DISPLAY FUNCTIONS #####
############################################################
def pcacomponentplots(ax,vh,ylabels,colors=''):
    numev = vh.shape[0]
    xlim = 1.05*np.max(np.abs(vh))
    if colors=='':
        colors = np.tile([0.35,0.35,0.35],(numev,1))
    elif len(colors)<numev:
        colors = np.tile(colors,(numev,1))
    for evnum in range(numev):
        a=ax[evnum]
        x=vh[evnum]
        y = np.flipud(np.arange(vh.shape[1]))
    #     ax.plot(x,y,'-o',label='$\\vec e_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',c=snscolors[evnum])
        thickness=0.75
        a.barh(y+0*(evnum-1)*thickness,x,height=thickness,color=colors[evnum])
        a.set_yticks(y)
        a.set_yticklabels(ylabels,rotation='horizontal',fontsize=14)
        a.axvline(0,c='k',linestyle='--')
#         a.set_title(label='$\\vec v_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',fontsize=14)
        a.set_xlim([-xlim,xlim])
        a.tick_params(labelsize=14)
    
def plot_tsne_withcolors(ax,tsne_result,quantity,title,labels=[],labeloffset=(0,0),
                         corrskip=1,plotskip=1,colortype='scalar',qmin=0,qmax=1,alphaval=0.3,s=4,
                         coloroffset=0,cmapname='cool',marker='o',snscolors=snscolors):
    colordata = quantity.copy()
    uniquecolors = np.unique(colordata)
    for i,u in enumerate(uniquecolors):
        colordata[quantity==u] = i
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    if len(marker)==1:
        scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=False,marker=marker)
    else:
        for x,y,c,m in zip(tp[::plotskip,0],tp[::plotskip,1],colors[::plotskip],marker[::plotskip]):
            scatterplot = ax.scatter(x,y,s=s,alpha=alphaval,color=c,rasterized=False,marker=m)
#     ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
#     ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    ax.set_title(title,fontsize=16)       
    if len(labels)>0:
        for lbl,x,y,clr in zip(labels,tp[:,0],tp[:,1],colors):
            ax.annotate(lbl, (x+labeloffset[0], y+labeloffset[1]),color=clr,fontsize=12)        
    return scatterplot, colordata       

def categorydists(membership,quantityvals,labels,pointskip=1,coloroffset=0,ax='',f=''):
    # get unique membership, and use this in the plots
    uniquemem = np.unique(membership)
    n_clusters = len(uniquemem)
    
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,n_clusters,sharex=True,sharey=False)
        f.set_size_inches(5*n_clusters,4)
    
    for i,memsel in enumerate(uniquemem):
        a = ax[i]
        clr = snscolors[i+coloroffset] if coloroffset>=0 else 'k'
        sel = membership == memsel
        a.set_title(memsel,fontsize=14)
        for j, q in enumerate(quantityvals):
            tp = q[sel]
            alpha_scaled = 0.2
            xval = len(quantityvals)-j-1
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,vert=False,widths=0.9)
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xnoise=0.1
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(tp[::pointskip],xtp[::pointskip],color=clr,alpha=0.7,zorder=10,s=5,rasterized=True)
    for a in ax[0:n_clusters]:
        a.set_yticks(range(len(quantityvals)))
        a.set_yticklabels(np.flip(labels),rotation='horizontal',fontsize=12)
        a.axvline(0,c='k',linewidth=1)
        a.set_xlabel('Quantity value (std. dev from mean)',fontsize=12)
    return f,ax

cmap=plt.get_cmap('viridis_r')  # or 'cool'

def quantitydists(membership,quantityvals,labels,f='',ax='',coloroffset=0,xorder=[],colorsel=[],color='k',xnoise=0.1,pointlabels=[],pointlabelcolors=[]):
    # use coloroffset=100 in order to use specific colors
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,numq,sharex=True,sharey=True)
        f.set_size_inches(2*numq,3)
        
    # get unique membership, and use this in the plots
    uniquemem = np.unique(membership)
    n_clusters = len(uniquemem)

    if len(xorder)==0:
        xorder = np.arange(n_clusters)
    if len(colorsel)==0:
        colorsel=np.arange(n_clusters)
    # dmat_bygroup = [[ for n in range(n_clusters)] for qnum in range(dmat.shape[1])]
    for i in range(numq):
        a = ax[i]
        a.set_title(labels[i],fontsize=16)
        for j,memsel in enumerate(uniquemem):
            if coloroffset>=100:
                clr = (color[j] if len(color)==n_clusters else color)
            else:
                clr = snscolors[np.mod(colorsel[j]+coloroffset,10)]
            tp = quantityvals[i][membership==memsel]
            alpha_scaled = 0.2
            xval = xorder[j]
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,whis=100,vert=True,widths=(0.6 if len(pointlabels)==0 else 0.3))
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(xtp,tp,color=(clr if len(pointlabelcolors)==0 else pointlabelcolors[membership==memsel]),alpha=0.7,zorder=10,s=10,rasterized=False)
            if len(pointlabels)>0:
                pl = pointlabels[membership==memsel]
                plc = pointlabelcolors[membership==memsel]
                for k in range(len(xtp)):
                    a.annotate(pl[k], (xtp[k]+0.2, tp[k]-0.1),color=plc[k],fontsize=12)

    [a.set_xticks(range(n_clusters)) for a in ax]
    [a.set_xticklabels(uniquemem,fontsize=16) for a in ax]
    [a.tick_params(labelsize=16) for a in ax]
    ax[0].set_ylabel('Quantity value',fontsize=16)
    # [a.set_xlabel('Cluster number',fontsize=16) for a in ax]
    [a.axhline(0,c='k',linewidth=1) for a in ax]
    return f, ax


## network embedding plot
def showgraph2(df,a,dfuidlabels,widthscale=1/20,which='plt',layout='localreaching'):   # layout = ['shell','spring','force']
    diff = makediff(df)
    total = df+df.T
    total[diff==0]=0
    G = getG(diff)
    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()
    
    Gtotal = getG(total)
    widthstotal = nx.get_edge_attributes(Gtotal, 'weight')
    
    # pos = nx.shell_layout(G)
    # defining my own layout
    if layout=='frac':
        # xtp,ytp = get_outdegree(df)+get_indegree(df), get_indegree(df)/(get_outdegree(df)+get_indegree(df))
        # pos = dict(zip(xtp.index,np.array([xtp.values,ytp.values]).T))
        indlabel = get_outdegree(df).index
        ytp = get_indegree(df)/(get_outdegree(df)+get_indegree(df))
        ytp=ytp.values
    elif layout=='localreaching':
        indlabel = get_outdegree(df).index
        ytp = get_localreaching(df)

    np.random.seed(10)
    xtp = np.zeros(len(ytp))+np.random.rand(len(ytp))*0.05- 0.025
    xtp[np.argmax(ytp)]=0
    # iterate x embedding values with a spring constant
    springconst=1
    delta=0.01
    threshold = 0.5
    springforces = np.tile(100,len(ytp))
    #. just use spring forces!  using the other (to minimize line length), is too complicated, and would require another algorithm description
    while np.any(springforces>threshold):
        xdiff = xtp[np.newaxis,:]-xtp[:,np.newaxis]
        dists = np.sqrt((ytp[np.newaxis,:]-ytp[:,np.newaxis])**2 + (xdiff)**2)
        dists[np.arange(len(dists)),np.arange(len(dists))] = np.inf
        dists[np.isnan(dists)] = np.inf
        springforces = np.sum(springconst/dists*np.sign(xdiff),axis=1)
        otherforces = np.sign(xtp)*np.abs(xtp) + np.nanmean(diff)*np.sum(diff.values/dists*xdiff,axis=1)
        totalforces= springforces+0*otherforces
        xtp = xtp -delta*totalforces
        xtp[np.argmax(ytp)]=0


    # ytp = ytp-np.min(ytp)
    ytp = (ytp-np.nanmean(ytp))/np.nanstd(ytp)
    pos = dict(zip(indlabel,np.array([xtp,ytp]).T))
    # a.set_xlabel('Out-degree',fontsize=16,color='gray')
    # a.set_ylabel('In-degree',fontsize=16,color='gray')
        
    wvals = np.array(list(widths.values()))
    wvalstotal = np.array(list(widthstotal.values()))
    # for color, made it the fraction won
    diffscaled = wvals/wvalstotal
    ## for how bold, make it by mean/std.
    # need to calculate mean and std using the input matrix, because wvalstotal already filters!  and does not include all
    dfvals = df.values.astype(float)
    dfvals = dfvals+dfvals.T # this, because entries should be the average total per pair
    np.fill_diagonal(dfvals,np.nan) 
    dfvals
    totalscaled = (wvalstotal - np.nanmean(dfvals))/np.nanstd(dfvals)
    mintotal = 3
    totalscaled[totalscaled<-mintotal] = -mintotal
    totalscaled[totalscaled>mintotal] = mintotal
    minalpha=0.1
    totalscaled = (1-minalpha)*(totalscaled + mintotal)/(2*mintotal) + minalpha
    
    if which=='plt':
        nx.draw_networkx_edges(G,pos,edgelist = widths.keys(),width=3,edge_color=cmap(diffscaled),alpha=totalscaled,ax=a,arrowsize=20,)
        pad=0.08
        xpad, ypad = pad*(np.nanmax(xtp)-np.nanmin(xtp)), pad*(np.nanmax(ytp)-np.nanmin(ytp))
        xmax, ymax = np.nanmax(xtp)+xpad, np.nanmax(ytp)+ypad
        # xmin, ymin = -pad*xmax,-pad*ymax
        xmin, ymin = np.nanmin(xtp)-xpad, np.nanmin(ytp)-ypad
        xs,ys = (xmax-xmin,ymax-ymin)
        a.set_xlim([xmin,xmax])
        # a.set_ylim([ymin,ymax])
        for uid in nodelist:
            x,y = pos[uid]
            og_int = np.where(np.array(groupselnames)==dfuidlabels.loc[uid,'Group'])[0][0]
            lbl = dfuidlabels.loc[uid,'masslabel']
            a.annotate(lbl,xy=(x-(0.055-(0.015 if len(lbl)==2 else 0))*xs,y-0.02*ys),fontsize=12,zorder=2,color=groupcolors[og_int])
            a.scatter(x,y,facecolors='white', edgecolors=groupcolors[og_int],s=620,zorder=1,alpha=1)
        # a.axis(False)
    else:
        nx.draw_networkx_edges(G,pos,edgelist = widths.keys(),width=3,edge_color=cmap(diffscaled),alpha=totalscaled,ax=a,arrowsize=20)    
        nx.draw_networkx_labels(G, pos=pos,labels={u: uidmasslabels[u] for u in nodelist },font_color='grey',ax=a,font_size=16) 
       

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.analysis_tools import ndvi
from src.wood_constants import DataConstants as dc
import src.preprocess as prep

def get_sample(X,y,sample,pnts=None):
    xim = [X[i][sample] for i in range(len(X))]
    yim = [y[i][sample] for i in range(len(y))]
    if pnts is None:
        return(xim, yim, sample)
    else:
        uid = pnts[sample][2]
        return(xim, yim, uid) 

'''
#deprecated
def plot_samples(X,y,pnts,sample,outfilename):
    xim, yim, uid = get_sample(X,y,pnts,sample)
    #fix this function.
    colorlist =["saddlebrown","y","yellow","yellowgreen","forestgreen","darkgreen"]
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",colorlist)
               
    fontsize=24
    months = [0,3,6,10]
    classes =['No tree', 'Conifer', 'Decidous', 'Mixed', 'Dead']
    plt.rcParams['figure.figsize'] = (26,8)
    ax0 = plt.subplot2grid((2,8), (0,0), rowspan=2, colspan=2)
    ax0.imshow(xim[0][:,:,:3])
    ax0.set_axis_off()
    c = classes[np.argmax(yim[0])]
    var = yim[1][0]
    ax0.set_title('%s,\nClass:%s,\nBM:%.1f Mg/ha'%(uid,c,var), \
                                                          size=fontsize, loc='left')
    #polar plot
    ax1 = plt.subplot2grid((2,8), (0,2), rowspan=2, colspan=2, projection='polar')
    R = np.arange(0,1,0.01)
    theta = prep.ne_to_deg(xim[1][0,0])
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.plot(theta*np.ones(len(R)),R)
    width = np.pi/8.
    elev = prep.un_norm(xim[1][0,1], dc._elevation[0], dc._elevation[1])
    bottom = xim[1][0,1]
    bars = ax1.bar(theta*np.ones(len(R)), R*0.2, \
                   width=width, \
                   bottom=bottom)
    slope = xim[1][0,2]*dc._slope
    ax1.set_yticklabels(['treeline','','','','sea level'], size=14)               
    ax1.set_xticklabels([])               
    ax1.set_rlabel_position(90)
    ax1.set_title('Elev:%.0fm,\nSlope:%.1f deg,\nAspect:%.0f deg'\
                 %(elev, slope, theta), size=fontsize)
    #climate plot
    ax2 = plt.subplot2grid((2,8), (0,4), colspan=4)
    ax2.set_ylim([0.1,0.9])
    ax2.plot(xim[2][:,0,0,1], 'red')
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xlabel('Months before acquisition', size=fontsize)
    ax2.set_ylabel('Tmean normalized', size=fontsize)
    ax3 = ax2.twinx()
    ax3.set_ylim([0,0.9])
    ax3.plot(xim[2][:,0,0,0], 'blue')
    ax3.set_ylabel('Ppt normalized',size=fontsize)
    ax3.tick_params(axis='y', labelsize=18)
    #ndvi plots
    for m in range(len(months)):
        ax_ls = plt.subplot2grid((2,8), (1,4+m), rowspan=1, colspan=1)
        ls_0 = prep.ndvi(xim[-1][months[m],:,:,:])
        ax_ls.imshow(ls_0, vmin=-0.5, vmax=0.9, cmap=my_cmap)
        ax_ls.set_axis_off()
        ax_ls.set_title('NDVI month: %s'%(months[m]+1),size=fontsize)
    plt.tight_layout()
    print('wrote %s'%outfilename)
    plt.savefig(outfilename, bbox_inches='tight')
'''

def plot_samples(X,y,sample,pnts=None,outfilename=None):
    xim, yim, uid = get_sample(X,y,sample,pnts)
    #fix this function.
    colorlist =["saddlebrown","y","yellow","yellowgreen","forestgreen","darkgreen"]
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",colorlist)
               
    fontsize=24
    months = [0,3,6,10]
    classes =['No tree', 'Conifer', 'Decidous', 'Mixed', 'Dead']
    plt.rcParams['figure.figsize'] = (28,10)
    ax0 = plt.subplot2grid((2,8), (0,0), rowspan=2, colspan=2)
    ax0.imshow(xim[0][:,:,:3])
    ax0.set_axis_off()
    c = classes[np.argmax(yim[0])]
    var = yim[1][0]
    ax0.set_title('%s,\nClass:%s,\nBM:%.1f Mg/ha'%(uid,c,var), \
                                                          size=fontsize, loc='left')
    #polar plot
    ax1 = plt.subplot2grid((2,8), (0,2), rowspan=2, colspan=2, projection='polar')
    R = np.arange(0,1,0.01)
    theta = prep.ne_to_deg(xim[1][0,0])
    d = np.deg2rad(theta*np.ones(len(R)))
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.plot(d,R)
    width = np.pi/8.
    elev = prep.un_norm(xim[1][0,1], dc._elevation[0], dc._elevation[1])
    bottom = 1-xim[1][0,1]
    bars = ax1.bar(d, R*0.2, \
                   width=width, \
                   bottom=bottom)
    slope = xim[1][0,2]*dc._slope
    ax1.set_yticklabels(['treeline','','','sea level'], size=14)               
    ax1.set_xticklabels([])               
    ax1.set_rlabel_position(90)
    ax1.set_title('Elev:%.0fm,\nSlope:%.1f deg,\nAspect:%.0f deg'\
                 %(elev, slope, theta), size=fontsize)
    #ndvi plots
    for m in range(len(months)):
        ls_c = m+4
        ax_ls = plt.subplot2grid((2,8), (0,ls_c), rowspan=2, colspan=1)
        ls_0 = prep.ndvi(xim[-1][months[m],:,:,:])
        ax_ls.imshow(ls_0, vmin=-0.5, vmax=0.9, cmap=my_cmap)
        ax_ls.set_axis_off()
        ax_ls.set_title('NDVI month:\n %s'%(months[m]+1),size=fontsize)
    #plt.tight_layout()
    if outfilename:
        print('wrote %s'%outfilename)
        plt.savefig(outfilename, bbox_inches='tight')

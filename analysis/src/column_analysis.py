import os
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

def get_arr_info(arr, column, save_dir, render=False):
    
    """ Get info + figure about a column of DF """
    
    # Describe overview of data
    print ("Overview (num tokens):")
    print ("=========")
    print (arr.describe())
    
    # Box plot of the data
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("BuGn_r", 1)
    plt.figure(figsize=(10,5))
    
    ax1= plt.subplot(1,2,1)
    sns.boxplot(None, arr)
    ax1.set_title(column + " length (all)")
    ax1.set_ylabel('number of tokens')
    
    ax2= plt.subplot(1,2,2)
    sns.boxplot(None, arr, showfliers=False)
    ax2.set_title(column + " length w/o upper outliers")
    ax2.set_ylabel('number of tokens')
    
    plt.suptitle(column + ' length')
    plt.savefig(os.path.join(save_dir, 'boxplot_{}_length.png'.format(column)), dpi=300)
    if render: plt.show() 
    
    # Display Distrubtion of the data
    
    num_bins = 20
    
    sns.set()
    sns.set_style("white")
    sns.set_palette("hls")
    plt.figure(figsize=(15,5))
    
    ax4 = plt.subplot(1,2,1)
    ax4.hist(arr, bins=num_bins, alpha=1)
    sns.despine()
    ax4.set_xlabel("num of tokens")
    ax4.set_ylabel("num of " + column)
    
    Q1 = arr.quantile(.25)
    Q2 = arr.quantile(.5)
    Q3 = arr.quantile(.75)
    
    IQR = Q3 - Q1
    max_len = Q3 + IQR * 1.5
    
    ax5 = plt.subplot(1,2,2)
    ax5.hist(arr, bins=num_bins, range=(0, max_len), alpha=1)
    sns.despine()
    ax5.set_xlabel("num of tokens")
    ax5.set_ylabel("num of " + column)
    
    plt.suptitle("Distribution of Answer\'s token")
    plt.savefig(os.path.join(save_dir, 'dist_{}_token.png'.format(column)), dpi=300)
    if render: plt.show()
    
import numpy as np
import scipy.stats as sio
from matplotlib.pyplot import *
from matplotlib import cm
from decode_bucket_trials import MOUSE, CAGE, VELOCITY_THRESHOLD
from plot_SCE_analysis import divide_to_boxes

def add_subplot_axes(ax,rect,axisbg='w'):
    # Taken from: https://stackoverflow.com/questions/17458580/embedding-small
    # -plots-inside-subplots-in-matplotlib
    fig = gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def main():
    ###### Plot decoding histogram and number of events per bin decoding ######

    f2, axx2 = subplots(1, 2, sharex=True, sharey='row')
    decoded_bins_all_mice = {'envA': [], 'envB': []}
    decoded_env_all_mice = {'envA': [], 'envB': []}
    number_of_events_per_frame_all_mice = {'envA': [], 'envB': []}
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load('bucket_decoding_results_c%sm%s.npz' % (CAGE[i], mouse))
        nitzan_first_bucket= npzfile['nitzan_first_bucket']
        nitzan_last_bucket = npzfile['nitzan_last_bucket']
        bambi_first_bucket = npzfile['bambi_first_bucket']
        bambi_last_bucket = npzfile['bambi_last_bucket']

        nitzan_bucket = [np.concatenate([x, y]) for x, y in
                         zip(nitzan_first_bucket, nitzan_last_bucket)]
        bambi_bucket = [np.concatenate([x,y]) for x,y in
                        zip(bambi_first_bucket,bambi_last_bucket)]

        nitzan_all = np.concatenate(nitzan_bucket)
        bambi_all = np.concatenate(bambi_bucket)

        axx2[i].hist([nitzan_all[~np.isnan(nitzan_all)],
                         bambi_all[~np.isnan(bambi_all)]], normed=True,
                        align='right', label= ['Phase 0', 'Phase 1'])
        axx2[i].set_title('C%sM%s' % (CAGE[i], mouse), fontsize=18)
        axx2[i].set_xlabel('#Bins', fontsize=17)

        # # Plot histogram for each bucket trial:
        # f, axx=subplots(5,4, sharex=True, sharey=True)
        # for i in range(5):
        #     axx[i, 0].hist(nitzan_first_bucket[i][~np.isnan(nitzan_first_bucket[i])], normed=True)
        #     axx[i, 1].hist(nitzan_last_bucket[i][~np.isnan(nitzan_last_bucket[i])], normed=True)
        #     axx[i, 2].hist(bambi_first_bucket[i][~np.isnan(bambi_first_bucket[i])], normed=True)
        #     axx[i, 3].hist(bambi_last_bucket[i][~np.isnan(bambi_last_bucket[i])], normed=True)
        # axx[0, 0].set_title('Nitzan first')
        # axx[0, 1].set_title('Nitzan last')
        # axx[0, 2].set_title('bambi first')
        # axx[0, 3].set_title('bambi last')
        # f.show()
        #
        # # Plot histogram for session bucket trial:
        # f, axx=subplots(5,2, sharex=True, sharey=True)
        # for i in range(5):
        #     axx[i, 0].hist(nitzan_bucket[i][~np.isnan(nitzan_bucket[i])], normed=True)
        #     axx[i, 1].hist(bambi_bucket[i][~np.isnan(bambi_bucket[i])], normed=True)
        # axx[0, 0].set_title('Nitzan')
        # axx[0, 1].set_title('bambi')
        # f.show()

    axx2[0].set_ylabel('Density of decoding', fontsize=17)

    for j in range(2):
        for xtick in axx2[j].xaxis.get_major_ticks():
            xtick.label.set_fontsize(15)
        for ytick in axx2[j].yaxis.get_major_ticks():
            ytick.label.set_fontsize(15)
    legend(fontsize=17)
    f2.show()

    raw_input('press enter')

if __name__ == '__main__':
    main()
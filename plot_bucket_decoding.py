import numpy as np
import scipy.stats as sio
from matplotlib.pyplot import *
from matplotlib import cm
from decode_bucket_trials import MOUSE, CAGE, VELOCITY_THRESHOLD
from plot_SCE_analysis import divide_to_boxes

def main():
    ###### Plot decoding histogram and number of events per bin decoding ######

    f2, axx2 = subplots(1, 2, sharex=True, sharey='row')
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

        bins_density_nitzan = []
        bins_density_bambi = []
        for k in range(5):
            y, bin_edges = np.histogram(nitzan_bucket[k]\
                                        [~np.isnan(nitzan_bucket[k])], bins=10,
                                        normed=True)
            bins_density_nitzan.append(y)
            y, bin_edges = np.histogram(bambi_bucket[k]\
                                        [~np.isnan(bambi_bucket[k])], bins=10,
                                        normed=True)
            bins_density_bambi.append(y)

        bins_density_nitzan = np.vstack(bins_density_nitzan)
        bins_density_bambi = np.vstack(bins_density_bambi)

        mean_bins_density_nitzan = np.mean(bins_density_nitzan, axis=0)
        mean_bins_density_bambi = np.mean(bins_density_bambi, axis=0)
        std_bins_density_nitzan = np.std(bins_density_nitzan, axis=0)
        std_bins_density_bambi = np.std(bins_density_bambi, axis=0)

        axx2[i].bar(bin_edges[:-1], mean_bins_density_nitzan,
                    yerr=std_bins_density_nitzan, width=0.35, color='b', label= 'Phase 0')
        axx2[i].bar(bin_edges[:-1]+0.35, mean_bins_density_bambi,
                    yerr=std_bins_density_bambi, width=0.35, color='g', label='Phase 1')
        legend()
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

        # Plot histogram for session bucket trial:
        # f, axx=subplots(5,2, sharex=True, sharey=True)
        # for k in range(5):
        #     axx[k, 0].hist(nitzan_bucket[k][~np.isnan(nitzan_bucket[k])], normed=True)
        #     axx[k, 0].set_ylabel('Density session %s' %k, fontsize=16)
        #     axx[k, 1].hist(bambi_bucket[k][~np.isnan(bambi_bucket[k])], normed=True)
        # axx[0, 0].set_title('Nitzan', fontsize=16)
        # axx[4, 0].set_xlabel('#Bins', fontsize=16)
        # axx[0, 1].set_title('bambi', fontsize=16)
        # axx[4, 1].set_xlabel('#Bins', fontsize=16)
        # f.suptitle('C%sM%s' % (CAGE[i], mouse), fontsize=18)
        # for j in range(5):
        #     for k in range(2):
        #         for xtick in axx[j, k].xaxis.get_major_ticks():
        #             xtick.label.set_fontsize(15)
        #         for ytick in axx[j, k].yaxis.get_major_ticks():
        #             ytick.label.set_fontsize(15)
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
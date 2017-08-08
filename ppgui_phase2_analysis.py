import os.path
import csv
import cPickle

from numpy import *
from scipy.stats import *
from matplotlib.pyplot import *
from seaborn import *
import h5py
import scipy.io

import ppgui_data_analysis as pda
import decode_bucket_trials as dbt

# Number of sessions in all runs in the experiment
NUMBER_OF_SESSIONS = 5
BAMBI_NUMBER_OF_TRIALS = 7
MOUSE = '6'
CAGE = '40'
ROIS_INDICES = {}
REGISTRATION_FILENAME = {}

EDGE_BINS = [0, 9]
EDGE_PERCENT = 0.9
# For c40m3
REGISTRATION_FILENAME['3'] = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\c40m3_registered_141116-090317\registration\cellRegistered_Final_20170328_113021.mat'
ROIS_INDICES['3'] = [36, 53, 80, 89, 158, 181, 195, 229, 258, 290, 321, 336,
                     339, 357, 366, 392, 394, 399, 408, 439, 446, 448, 449,
                     465, 490]

# For c40m6
REGISTRATION_FILENAME['6'] = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\c40m6_registered_1202-0309\registration\cellRegistered_Final_20170423_123804.mat'
ROIS_INDICES['6'] = [44, 61, 78, 96, 154, 157, 172, 195, 214, 226, 244, 247,
                     259, 261, 262, 286, 287, 290, 301, 303, 314, 337, 340,
                     346, 348, 368, 372, 374, 383, 389, 391, 407, 415, 418,
                     419, 448, 448, 460, 472, 473, 474, 479, 488, 501, 517, 569]

def  load_cell_registration(mouse):
    #Taken from Or's script - for C40M3
    # Load the cell registration results
    cell_registration = h5py.File(REGISTRATION_FILENAME[mouse])['cell_registered_struct'][
        'optimal_cell_to_index_map'].value.astype(int)
    # Compensate for 0-based indexing
    cell_registration -= 1

    bambi_run = np.transpose(cell_registration[-5:])

    # Find the global numbering of the ROIs:
    if mouse == '3':
        session_neurons_indices = np.transpose(cell_registration[5])
    else:
        session_neurons_indices = np.transpose(cell_registration[0])

    global_numbering_roi = np.zeros(len(ROIS_INDICES[mouse]), dtype=int)
    for i, cell_index in enumerate(ROIS_INDICES[mouse]):
        global_numbering_roi[i] = \
            np.argwhere(session_neurons_indices == cell_index)


    return bambi_run, global_numbering_roi

def extract_bambi_data(cage, mouse):
    """Taken from OR's code dynamic_analysis"""

    BASE_DIRNAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_2_preprocessed'
    full_bins_traces = []
    full_velocity_traces = []
    full_events_traces = []
    frame_logs = []
    linear_trials = [0, 6]
    for i in xrange(NUMBER_OF_SESSIONS):
        bins_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1), 'c'+cage+'m'+mouse, 'my_mvmt.mat')
        my_mvmt = scipy.io.loadmat(bins_filename)['my_mvmt'][0]
        session_bins_traces = []
        session_velocity_traces = []
        for j in linear_trials:
            # All 3 first indices are magic to get to the required position.
            # The 1: is used to remove the first behavioral frame which is dropped
            # in the neuronal.
            session_bins_traces.append(my_mvmt[j][0][0][3][1:].T[0])
            session_velocity_traces.append(my_mvmt[j][0][0][4][1:].T[0])

        full_bins_traces.append(session_bins_traces)
        full_velocity_traces.append(session_velocity_traces)

        events_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+cage+'m'+mouse, 'finalEventsMat.mat')
        allEventsMat = scipy.io.loadmat(events_filename)['allEventsMat'].T
        full_events_traces.append(allEventsMat)

        frame_log_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+cage+'m'+mouse, 'frameLog.csv')
        frameLog = csv.reader(open(frame_log_filename, 'rb'))
        # Skip header
        frameLog.next()
        session_frame_log = []
        for line in frameLog:
            # Set indices to 0-index based, but the second index should end one frame after
            session_frame_log.append([int(line[2]) - 1, int(line[3])])
        frame_logs.append(session_frame_log)

    # Concatenate all trials for behavior and neuronal data
    bucket_events_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        bucket_events_traces.append(full_events_traces[i][:, frame_logs[i][1][0]:frame_logs[i][-1][0]])

    events_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        # The second index in frame_logs is for taking only the linear tracks
        current_events = []
        current_events.append(full_events_traces[i][:, frame_logs[i][0][0]:frame_logs[i][0][1]])
        current_events.append(full_events_traces[i][:, frame_logs[i][-1][0]:frame_logs[i][-1][1]])
        events_traces.append(np.hstack(current_events))

    bins_traces = []
    velocity_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        bins_trace = []
        velocity_trace = []
        for k, j in enumerate(linear_trials):
            # Remove missing frames
            microscope_statistics = cPickle.load(open(
                os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+cage+'m'+mouse, 'linear_trial_%d' % (j,), 'microscope_statistics.pkl')))
            fixed_bins_traces = delete(full_bins_traces[i][k], microscope_statistics['missing_frames'])
            fixed_velocity_traces = delete(full_velocity_traces[i][k],
                                       microscope_statistics['missing_frames'])
            bins_trace.extend(fixed_bins_traces)
            velocity_trace.extend(fixed_velocity_traces)

        bins_trace = array(bins_trace, dtype=int)
        velocity_trace = array(velocity_trace, dtype=int)
        # Fix 0-based indexing
        bins_trace -= 1

        bins_traces.append(array(bins_trace))
        velocity_traces.append(array(velocity_trace))
    # Remove extra frames
    for i in xrange(NUMBER_OF_SESSIONS):
        minimum_number_of_frames = min(len(bins_traces[i]), events_traces[i].shape[1])

        if minimum_number_of_frames != len(bins_traces[i]) or minimum_number_of_frames != events_traces[i].shape[1]:
            print 'Minimum number of frames in session %d is %d' % (i, minimum_number_of_frames)

            bins_traces[i] = bins_traces[i][:minimum_number_of_frames]
            velocity_traces[i] = velocity_traces[i][:minimum_number_of_frames]
            events_traces[i] = events_traces[i][:, :minimum_number_of_frames]

    print

    for i in xrange(NUMBER_OF_SESSIONS):
        print 'Number of frames in session %d: %d' % (i, events_traces[i].shape[1])

    return bins_traces, velocity_traces, events_traces, bucket_events_traces

def set_in_data(current_data, bins_traces, events_traces, bucket_events_traces,
                velocity_traces):
    current_data['bins_traces'] = bins_traces
    current_data['velocity_traces'] = velocity_traces
    current_data['events_traces'] = [(e > 0).astype('int') for e in events_traces]
    current_data['bucket_events_traces'] = [(e > 0).astype('int') for e in bucket_events_traces]

    return current_data

def plot_all_track_dynamics(data): ##### EDIT THIS ####
    bambi_track_dynamics_edge = \
        pda.analyze_track_dynamics(data, 'edge_cells')
    bambi_track_dynamics_non_edge = \
        pda.analyze_track_dynamics(data, 'non_edge_cells')
    bambi_track_dynamics_chosen = \
        pda.analyze_track_dynamics(data, 'chosen_rois')

    f, axx = subplots(4, 1, sharey='row', sharex='row')
    f.subplots_adjust(top=0.9)
    ###### plot recurrence ######
    # For edge cells
    bambi_average_edge_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_edge, 'recurrence')[0]
    bambi_average_chosen_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_chosen, 'recurrence')[0]
    bambi_average_non_edge_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_non_edge, 'recurrence')[0]
    a, s = pda.probabilities_properties(bambi_average_edge_recurrence)
    line1 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s, label='edge cells')
    a, s = pda.probabilities_properties(bambi_average_chosen_recurrence)
    line2 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s,
                       label='chosen cells')
    a, s = pda.probabilities_properties(bambi_average_non_edge_recurrence)
    line3= axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s,
                               label='non edge cells')

    axx[0].set_ylabel('Recurrence',fontsize=14)
    axx[0].set_xlabel('Session difference', fontsize=14)
    legend(bbox_to_anchor=(1.05, 1.),
           handles=[line1, line2, line3], fontsize=14)

    ###### plot ensamble correlation ######
    # For edge cells
    bambi_average_edge_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_edge, 'ensamble_correlation')[0]
    bambi_average_chosen_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_chosen, 'ensamble_correlation')[0]
    bambi_average_non_edge_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_non_edge,
                             'ensamble_correlation')[0]

    a, s = pda.probabilities_properties(bambi_average_edge_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(bambi_average_chosen_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(bambi_average_non_edge_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1].set_ylabel('Ensamble correlation', fontsize=14)
    axx[1].set_xlabel('Session difference', fontsize=14)

    ###### plot pv correlation ######
    # For edge cells
    bambi_average_edge_pv_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_edge,
                                     'pv_correlation')[0]
    bambi_average_chosen_pv_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_chosen,
                                     'pv_correlation')[0]
    bambi_average_non_edge_pv_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_non_edge,
                         'pv_correlation')[0]


    a, s = pda.probabilities_properties(bambi_average_edge_pv_correltaion)
    axx[2].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(bambi_average_chosen_pv_correltaion)
    axx[2].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(bambi_average_non_edge_pv_correltaion)
    axx[2].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[2].set_ylabel('PV correlation', fontsize=14)
    axx[2].set_xlabel('Session difference', fontsize=14)

    ###### plot number of events and cells ######
    # for edge cells
    
    bambi_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_track_dynamics_edge):
        bambi_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_chosen_event_rate = []
    for j, session in enumerate(bambi_track_dynamics_chosen):
        bambi_chosen_event_rate.append(session['events_rate'])

    bambi_edge_event_rate = np.vstack(bambi_edge_event_rate)
    bambi_chosen_event_rate = np.vstack(bambi_chosen_event_rate)
    bambi_edge_event_rate[bambi_edge_event_rate == 0] = nan
    bambi_chosen_event_rate[bambi_chosen_event_rate == 0] = nan
    bambi_average_edge_event_rate = np.nanmean(bambi_edge_event_rate, axis=0)
    bambi_average_chosen_event_rate = np.nanmean(bambi_chosen_event_rate, axis=0)
    bambi_std_edge_event_rate = np.nanstd(bambi_edge_event_rate, axis=0)
    bambi_std_chosen_event_rate = np.nanstd(bambi_chosen_event_rate, axis=0)

    axx[3].errorbar(arange(NUMBER_OF_SESSIONS), bambi_average_edge_event_rate,
                       bambi_std_edge_event_rate)
    axx[3].errorbar(arange(NUMBER_OF_SESSIONS),
                       bambi_average_chosen_event_rate,
                       bambi_std_chosen_event_rate)
    axx[3].set_ylabel('Number of events', fontsize=14)
    axx[3].set_xlabel('Session number', fontsize=14)

    # For non edge cells
    bambi_non_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_track_dynamics_non_edge):
        bambi_non_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_non_edge_event_rate = np.vstack(bambi_non_edge_event_rate)
    bambi_non_edge_event_rate[bambi_non_edge_event_rate == 0] = nan
    bambi_average_non_edge_event_rate = np.nanmean(bambi_non_edge_event_rate, axis=0)
    bambi_std_non_edge_event_rate = np.nanstd(bambi_non_edge_event_rate, axis=0)
    axx[3].errorbar(arange(NUMBER_OF_SESSIONS),
                       bambi_average_non_edge_event_rate,
                       bambi_std_non_edge_event_rate)

    setp(axx, xticks=range(5), xticklabels=['1', '2', '3', '4', '5'])
    for i in range(4):
        for xtick in axx[i].xaxis.get_major_ticks():
            xtick.label.set_fontsize(14)
        for ytick in axx[i].yaxis.get_major_ticks():
            ytick.label.set_fontsize(14)
        box = axx[i].get_position()
        axx[i].set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
    f.suptitle('C%sM%s' %(CAGE, MOUSE), fontsize=25)
    f.show()
    return

def analyze_bucket_dynamics(data, cell_type):
    # Calculate recurrence probability, event rate, and ensamble correlation and
    # PV for a data set, given its global_numbering_events and edge_cells calculated
    # before. for all sessions cell_type is either: 'edge_cells' or
    # 'non_edge_cells'
        events_threshold = 5
        track_sessions_dynamics = []
        number_of_neurons = data['global_numbering_bucket'][0].shape[0]

        for session_index in xrange(NUMBER_OF_SESSIONS):
            dynamics = {}
            if cell_type == 'edge_cells':
                cells_indices = data[cell_type][session_index]
            elif cell_type == 'non_edge_cells':
                edge_cells = data['edge_cells'][session_index]
                all_neurons_indexing = np.arange(number_of_neurons)
                non_edge_cells = all_neurons_indexing
                non_edge_cells[edge_cells] = 0
                cells_indices = non_edge_cells[non_edge_cells > 0]
            elif cell_type == 'chosen_rois':
                cells_indices = data['chosen roi indices']

            track_events = data['global_numbering_bucket']

            dynamics['recurrence'] = pda.recurrence_probability \
                (track_events, cells_indices, events_threshold)

            dynamics['ensamble_correlation'] = pda.calculate_ensamble_correlation \
                (track_events, cells_indices)

            dynamics['events_rate'] = pda.count_events_per_session \
                (track_events, cells_indices)

            track_sessions_dynamics.append(dynamics)

        return track_sessions_dynamics

def plot_all_bucket_dynamics(data):  ##### EDIT THIS ####
    bambi_track_dynamics_edge = \
        analyze_bucket_dynamics(data, 'edge_cells')
    bambi_track_dynamics_non_edge = \
        analyze_bucket_dynamics(data, 'non_edge_cells')
    bambi_track_dynamics_chosen = \
        analyze_bucket_dynamics(data, 'chosen_rois')

    f, axx = subplots(3, 1, sharey='row', sharex='row')
    f.subplots_adjust(top=0.9)
    ###### plot recurrence ######
    # For edge cells
    bambi_average_edge_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_edge, 'recurrence')[0]
    bambi_average_chosen_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_chosen, 'recurrence')[0]
    bambi_average_non_edge_recurrence = pda.average_dynamics \
        (bambi_track_dynamics_non_edge, 'recurrence')[0]
    a, s = pda.probabilities_properties(bambi_average_edge_recurrence)
    line1 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s,
                            label='edge cells')
    a, s = pda.probabilities_properties(bambi_average_chosen_recurrence)
    line2 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s,
                            label='chosen cells')
    a, s = pda.probabilities_properties(bambi_average_non_edge_recurrence)
    line3 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), a, s,
                            label='non edge cells')

    axx[0].set_ylabel('Recurrence', fontsize=14)
    axx[0].set_xlabel('Session difference', fontsize=14)
    legend(bbox_to_anchor=(1.05, 1.),
           handles=[line1, line2, line3], fontsize=14)

    ###### plot ensamble correlation ######
    # For edge cells
    bambi_average_edge_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_edge, 'ensamble_correlation')[
            0]
    bambi_average_chosen_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_chosen,
                             'ensamble_correlation')[0]
    bambi_average_non_edge_ensamble_correltaion = \
        pda.average_dynamics(bambi_track_dynamics_non_edge,
                             'ensamble_correlation')[0]

    a, s = pda.probabilities_properties(bambi_average_edge_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(
        bambi_average_chosen_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = pda.probabilities_properties(
        bambi_average_non_edge_ensamble_correltaion)
    axx[1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1].set_ylabel('Ensamble correlation', fontsize=14)
    axx[1].set_xlabel('Session difference', fontsize=14)

    ###### plot number of events and cells ######
    # for edge cells

    bambi_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_track_dynamics_edge):
        bambi_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_chosen_event_rate = []
    for j, session in enumerate(bambi_track_dynamics_chosen):
        bambi_chosen_event_rate.append(session['events_rate'])

    bambi_edge_event_rate = np.vstack(bambi_edge_event_rate)
    bambi_chosen_event_rate = np.vstack(bambi_chosen_event_rate)
    bambi_edge_event_rate[bambi_edge_event_rate == 0] = nan
    bambi_chosen_event_rate[bambi_chosen_event_rate == 0] = nan
    bambi_average_edge_event_rate = np.nanmean(bambi_edge_event_rate, axis=0)
    bambi_average_chosen_event_rate = np.nanmean(bambi_chosen_event_rate,
                                                 axis=0)
    bambi_std_edge_event_rate = np.nanstd(bambi_edge_event_rate, axis=0)
    bambi_std_chosen_event_rate = np.nanstd(bambi_chosen_event_rate, axis=0)

    axx[2].errorbar(arange(NUMBER_OF_SESSIONS), bambi_average_edge_event_rate,
                    bambi_std_edge_event_rate)
    axx[2].errorbar(arange(NUMBER_OF_SESSIONS),
                    bambi_average_chosen_event_rate,
                    bambi_std_chosen_event_rate)
    axx[2].set_ylabel('Number of events', fontsize=14)
    axx[2].set_xlabel('Session number', fontsize=14)

    # For non edge cells
    bambi_non_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_track_dynamics_non_edge):
        bambi_non_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_non_edge_event_rate = np.vstack(bambi_non_edge_event_rate)
    bambi_non_edge_event_rate[bambi_non_edge_event_rate == 0] = nan
    bambi_average_non_edge_event_rate = np.nanmean(bambi_non_edge_event_rate,
                                                   axis=0)
    bambi_std_non_edge_event_rate = np.nanstd(bambi_non_edge_event_rate, axis=0)
    axx[2].errorbar(arange(NUMBER_OF_SESSIONS),
                    bambi_average_non_edge_event_rate,
                    bambi_std_non_edge_event_rate)

    setp(axx, xticks=range(5), xticklabels=['1', '2', '3', '4', '5'])
    for i in range(3):
        for xtick in axx[i].xaxis.get_major_ticks():
            xtick.label.set_fontsize(14)
        for ytick in axx[i].yaxis.get_major_ticks():
            ytick.label.set_fontsize(14)
        box = axx[i].get_position()
        axx[i].set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
    f.suptitle('C%sM%s' % (CAGE, MOUSE), fontsize=25)
    f.show()
    return

def test_all_bucket_trials(data):
    # test all bucket trials that exist in data[name]
    decoded_bins = []
    number_of_sessions = len(data['events_traces'])

    for session_index in range(number_of_sessions):
        p_neuron_bin = data['p_neuron_bin'][session_index]
        place_cells = data['place_cells'][session_index]
        bucket_events = data['bucket_events_traces'][session_index]
        decoded_bins.append \
            (dbt.test_bucket_trial(bucket_events[place_cells, :],
                               p_neuron_bin, EDGE_BINS))

    return decoded_bins

def plot_bucket_decoding(data):
    all_bucket = np.concatenate(data['decoded_bins_bucket'])
    all_bins = np.concatenate(data['bins_traces'])
    f0 = figure()
    hist([all_bucket, all_bins], normed=True,
          align='right', label= ['decoded bins in bucket',
                                 'linear track occupancy'])
    title('C%sM%s' %(CAGE, MOUSE), fontsize=25)
    xlabel('#Bins', fontsize=17)
    ylabel('Density', fontsize=17)
    yticks(fontsize=16)
    xticks(fontsize=16)
    legend(fontsize=17)
    f0.show()

    number_of_sessions = len(data['decoded_bins_bucket'])
    f, axx = subplots(1, number_of_sessions, sharex=True, sharey=True)
    for i in range(number_of_sessions):
        axx[i].hist([data['decoded_bins_bucket'][i],
                     data['bins_traces'][i]],  normed=True,
                     align='right', label= ['decoded bins',
                                 'linear track occupancy'])
        axx[i].set_title('Session %d' %i, fontsize=17)
        axx[i].set_xlabel('#Bins', fontsize=17)
        axx[i].set_ylabel('Density', fontsize=17)

    for j in range(number_of_sessions):
        for xtick in axx[j].xaxis.get_major_ticks():
            xtick.label.set_fontsize(15)
        for ytick in axx[j].yaxis.get_major_ticks():
            ytick.label.set_fontsize(15)
    legend(fontsize=17)
    f.show()
    return

def main():
    data = {}
    [bins_traces, velocity_traces, events_traces, bucket_events_traces] = extract_bambi_data(CAGE, MOUSE)
    data = set_in_data(data, bins_traces, events_traces, bucket_events_traces,
                       velocity_traces)
    bambi_registration, ROI_global_indices = load_cell_registration(MOUSE)

    # Matching all neurons' IDs to the global ID

    data['global_numbering_events'] = pda.renumber_sessions_cells_ID\
        (data['events_traces'],bambi_registration)

    data['chosen roi indices'] = ROI_global_indices

    data['global_numbering_bucket'] = \
        pda.renumber_sessions_cells_ID(
            data['bucket_events_traces'], bambi_registration)

    # Finding edge cells from all days
    data['edge_cells'] = pda.find_edge_cells_for_all_sessions\
        (data['bins_traces'], data['global_numbering_events'],
         EDGE_PERCENT, EDGE_BINS)
    data['place_cells'] = dbt.find_place_cells_for_all_sessions(
        bins_traces, velocity_traces, events_traces)
    # Calculate p_neuron_bin for all sessions
    data['p_neuron_bin'] = \
        dbt.calculate_p_neuron_bin_directions_for_all_sessions(bins_traces,
                                                           velocity_traces,
                                                           events_traces,
                                                           data['place_cells'])
    data['decoded_bins_bucket'] = test_all_bucket_trials(data)

    plot_all_bucket_dynamics(data)
    plot_all_track_dynamics(data)
    plot_bucket_decoding(data)
    raw_input('Press enter to quit')

if __name__ == '__main__':
    main()
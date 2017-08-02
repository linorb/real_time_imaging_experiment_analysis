import os.path
import csv
import cPickle

from numpy import *
from scipy.stats import *
from matplotlib.pyplot import *
from seaborn import *
import h5py
import scipy.io

from bambi.tools.activity_loading import unite_sessions
from zivlab.analysis.place_cells import find_place_cells, calculate_event_rate_distribution

# Number of sessions in all runs in the experiment
NUMBER_OF_SESSIONS = 5
BAMBI_NUMBER_OF_TRIALS = 7
MOUSE = '6'
CAGE = '40'

EDGE_BINS = [0, 1, 10, 11]
EDGE_PERCENT = 0.9
# For c40m3
CELL_REGISTRATION_FILENAME = r'Z:\Short term data storage\Data storage (1 year)\Nitzan\c40m3\registration_110_days\cellRegistered_Final_16-Mar-2017_133500.mat'

# For c40m6
NITZAN_CELL_REGISTRATION_FILENAME = r'D:\dev\replays\work_data\recall\c40m6\cellRegistered_linear.mat'
BAMBI_CELL_REGISTRATION_FILNAME = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\c40m6_registered_1202-0309\registration\cellRegistered_Final_20170423_123804.mat'

def  load_cell_registration():
    #Taken from Or's script - for C40M3
    # Load the cell registration results
    if MOUSE == '3':
        cell_registration = h5py.File(CELL_REGISTRATION_FILENAME)['cell_registered_struct'][
            'optimal_cell_to_index_map'].value.astype(int)
        # Compensate for 0-based indexing
        cell_registration -= 1

        nitzan_run = np.transpose(cell_registration[:5])
        bambi_run = np.transpose(cell_registration[-5:])
    elif MOUSE == '6':
        # Load the cell registration results
        cell_registration = scipy.io.loadmat(NITZAN_CELL_REGISTRATION_FILENAME)[
            'optimal_cell_to_index_map'].astype(int)
        # Compensate for 0-based indexing
        cell_registration -= 1
        nitzan_run = cell_registration[:, :5]

        cell_registration = \
        h5py.File(BAMBI_CELL_REGISTRATION_FILNAME)['cell_registered_struct'][
            'optimal_cell_to_index_map'].value.astype(int)
        cell_registration -= 1
        bambi_run = np.transpose(cell_registration[1:6])

    return nitzan_run, bambi_run

def extract_nitzans_data():
    """Taken from OR's code dynamic_analysis"""
    full_bins_traces = []
    full_events_traces = []
    frame_logs = []
    missing_trials = []

    for i in xrange(NUMBER_OF_SESSIONS):
        print i
        bins_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\tracking\Cage%s_Mouse%s' %(CAGE, MOUSE),
                                     'Day%d.mat' % (i + 1,))
        my_mvmt = scipy.io.loadmat(bins_filename)['my_mvmt'][0]
        session_bins_traces = []
        for j in xrange(1, my_mvmt.shape[0]):
            print j
            # All 3 first indices are magic to get to the required position.
            # The 1: is used to remove the first behavioral frame which is dropped
            # in the neuronal.
            try:
                session_bins_traces.append(my_mvmt[j][0][0][3][1:].T[0])
            except IndexError:
                missing_trials.append([i,j])
        full_bins_traces.append(session_bins_traces)

        events_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\Pre_processing\c%sm%s' %(CAGE, MOUSE),
                                       'day%d' % (i + 1,),
                                       'linear',
                                       'finalResults',
                                       'finalEventsMat.mat')
        allEventsMat = scipy.io.loadmat(events_filename)['allEventsMat'].T
        full_events_traces.append(allEventsMat)

        frame_log_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\Pre_processing\c%sm%s' %(CAGE, MOUSE),
                                          'day%d' % (i + 1,),
                                          'linear',
                                          'finalResults',
                                          'frameLog.csv')
        frameLog = csv.reader(open(frame_log_filename, 'rb'))
        # Skip header
        frameLog.next()
        session_frame_log = []
        for line in frameLog:
            # Set indices to 0-index based, but the second index should end one frame after
            session_frame_log.append([int(line[2]) - 1, int(line[3])])
        frame_logs.append(session_frame_log)

    # Concatenate all trials for behavior and neuronal data
    events_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        # The second index in frame_logs is for taking only the linear trials indices
        if len(missing_trials) == 0:
            events_traces.append(full_events_traces[i][:, frame_logs[i][1][0]:frame_logs[i][-1][0]])
        else:
            missing_trials = np.array(missing_trials)
            if i in missing_trials[:, 0]:
                linear_trials =  []
                missing_session_trials = missing_trials[missing_trials[:, 0] == i, 1]
                number_of_trials = len(frame_logs[i])
                for j in range(1, number_of_trials-1):
                    if not(j in missing_session_trials):
                        linear_trials.append(full_events_traces[i][:, frame_logs[i][j][0]:frame_logs[i][j][1]])
                linear_trials = np.concatenate(linear_trials, axis=1)
                events_traces.append(linear_trials)
            else:
                events_traces.append(full_events_traces[i][:, frame_logs[i][1][0]:frame_logs[i][-1][0]])

    bucket_events_traces = {'first': [], 'last': []}
    for i in xrange(NUMBER_OF_SESSIONS):
        # The second index in frame_logs is for taking only the bucket
        bucket_events_traces['first'].append(full_events_traces[i][:, frame_logs[i][0][0]:frame_logs[i][0][1]])
        bucket_events_traces['last'].append(full_events_traces[i][:, frame_logs[i][-1][0]:frame_logs[i][-1][1]])

    bins_traces = []
    fixed_bins_traces = full_bins_traces[:]
    # For c40m3 do this correction:
    if MOUSE == '3':
        fixed_bins_traces[4] = fixed_bins_traces[4][:2] + fixed_bins_traces[4][3:]
    for i in xrange(NUMBER_OF_SESSIONS):
        bins_trace = []
        for j in xrange(len(fixed_bins_traces[i])):
            bins_trace.extend(fixed_bins_traces[i][j])

        bins_trace = array(bins_trace)
        # Fix 0-based indexing
        bins_trace -= 1

        # Rebin from the range 0..23 to 0..11
        bins_trace = floor(bins_trace.astype(float) / 24 * 10).astype(int)

        bins_traces.append(array(bins_trace))

    # Remove extra frames
    for i in xrange(NUMBER_OF_SESSIONS):
        number_of_extra_frames = len(bins_traces[i]) - events_traces[i].shape[1]
        if number_of_extra_frames > 0:
            print 'Removing %d frames from session %d' % (number_of_extra_frames, i)

            bins_traces[i] = bins_traces[i][:-1]



    for i in xrange(NUMBER_OF_SESSIONS):
        print 'Number of frames in session %d: %d' % (i, events_traces[i].shape[1])

    return bins_traces, events_traces, bucket_events_traces

def extract_bambi_data():
    """Taken from OR's code dynamic_analysis"""

    BASE_DIRNAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_1_preprocessed'
    full_bins_traces = []
    full_events_traces = []
    frame_logs = []

    for i in xrange(NUMBER_OF_SESSIONS):
        bins_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+CAGE+'m'+MOUSE, 'my_mvmt.mat')
        my_mvmt = scipy.io.loadmat(bins_filename)['my_mvmt'][0]
        session_bins_traces = []
        for j in xrange(1, my_mvmt.shape[0]):
            # All 3 first indices are magic to get to the required position.
            # The 1: is used to remove the first behavioral frame which is dropped
            # in the neuronal.
            session_bins_traces.append(my_mvmt[j][0][0][3][1:].T[0])
        full_bins_traces.append(session_bins_traces)

        events_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+CAGE+'m'+MOUSE, 'finalEventsMat.mat')
        allEventsMat = scipy.io.loadmat(events_filename)['allEventsMat'].T
        full_events_traces.append(allEventsMat)

        frame_log_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+CAGE+'m'+MOUSE, 'frameLog.csv')
        frameLog = csv.reader(open(frame_log_filename, 'rb'))
        # Skip header
        frameLog.next()
        session_frame_log = []
        for line in frameLog:
            # Set indices to 0-index based, but the second index should end one frame after
            session_frame_log.append([int(line[2]) - 1, int(line[3])])
        frame_logs.append(session_frame_log)

    # Concatenate all trials for behavior and neuronal data
    events_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        events_traces.append(full_events_traces[i][:, frame_logs[i][1][0]:frame_logs[i][-1][0]])

    bucket_events_traces = {'first': [], 'last': []}
    for i in xrange(NUMBER_OF_SESSIONS):
        # The second index in frame_logs is for taking only the bucket
        bucket_events_traces['first'].append(full_events_traces[i][:, frame_logs[i][0][0]:frame_logs[i][0][1]])
        bucket_events_traces['last'].append(full_events_traces[i][:, frame_logs[i][-1][0]:frame_logs[i][-1][1]])

    bins_traces = []
    fixed_bins_traces = []
    for i in xrange(NUMBER_OF_SESSIONS):
        bins_trace = []
        for j in xrange(len(full_bins_traces[i])):
            # Remove missing frames
            microscope_statistics = cPickle.load(open(
                os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,), 'c'+CAGE+'m'+MOUSE, 'linear_trial_%d' % (j+1,), 'microscope_statistics.pkl')))
            fixed_bins_traces = delete(full_bins_traces[i][j], microscope_statistics['missing_frames'])
            bins_trace.extend(fixed_bins_traces)

        bins_trace = array(bins_trace, dtype=int)
        # Fix 0-based indexing
        bins_trace -= 1

        bins_traces.append(array(bins_trace))

    # Remove extra frames
    for i in xrange(NUMBER_OF_SESSIONS):
        minimum_number_of_frames = min(len(bins_traces[i]), events_traces[i].shape[1])

        if minimum_number_of_frames != len(bins_traces[i]) or minimum_number_of_frames != events_traces[i].shape[1]:
            print 'Minimum number of frames in session %d is %d' % (i, minimum_number_of_frames)

            bins_traces[i] = bins_traces[i][:minimum_number_of_frames]
            events_traces[i] = events_traces[i][:, :minimum_number_of_frames]

    print

    for i in xrange(NUMBER_OF_SESSIONS):
        print 'Number of frames in session %d: %d' % (i, events_traces[i].shape[1])

    return bins_traces, events_traces, bucket_events_traces

def set_in_data(current_data, bins_traces, events_traces, bucket_events_traces):
    current_data['bins_traces'] = bins_traces
    current_data['events_traces'] = [(e > 0).astype('int') for e in events_traces]
    current_data['bucket_events_traces'] = {}
    current_data['bucket_events_traces']['first'] = [(e > 0).astype('int') for e in bucket_events_traces['first']]
    current_data['bucket_events_traces']['last'] = [(e > 0).astype('int') for e in bucket_events_traces['last']]

    return current_data

def find_edge_cells(bins, events, edge_percent, edge_bins):
    # count number of events per neuron
    number_of_neuron_events = np.sum(events, axis=1)

    #count the number of events in the edges per neuron
    edge_frames_indices = np.zeros_like(bins, dtype=bool)
    for bin in edge_bins:
        edge_frames_indices[bins == bin] = True
    number_of_edge_events = np.sum(events[:, edge_frames_indices], axis=1)

    percent_edge_per_neuron = np.divide(number_of_edge_events, number_of_neuron_events)
    edge_neurons = np.squeeze(np.argwhere(percent_edge_per_neuron >= edge_percent))

    return edge_neurons

def find_edge_cells_for_all_sessions(bins_traces, events_traces, edge_percent, edge_bins):
    edge_cells = []
    for events, bins in zip(events_traces, bins_traces):
        edge_cells.append(find_edge_cells(bins, events, edge_percent, edge_bins))

    return edge_cells

def renumber_sessions_cells_ID(events_traces, cell_to_index_map):
    """Renumbers the events' list, as inserted to the data struct, to the global
    cell's ID"""
    global_numbering_events = []
    for i, session in enumerate(events_traces):
        global_session = unite_sessions([session], [i], cell_to_index_map)
        global_numbering_events.append(global_session)

    return global_numbering_events

def recurrence_probability(global_numbering_events, cells_indices, events_threshold):
    # Calculate recurrence probability. assuming global_numbering_events is a list of events matrices that has global
    # numbering of neurons. and cells_indices are global indices

    number_of_sessions = len(global_numbering_events)
    over_threshold_events = []

    # Counting number of events per cell per session for the cell indices
    for session in global_numbering_events:
        over_threshold_events.append(np.sum(session[cells_indices, :], axis=1) > events_threshold)


    # Probabilities
    p = np.zeros((number_of_sessions, number_of_sessions))

    for i in xrange(number_of_sessions):
        for j in xrange(number_of_sessions):
            try:
                p[i, j] = float(np.count_nonzero(over_threshold_events[i] & over_threshold_events[j])) / \
                          float(np.count_nonzero(over_threshold_events[i]))
            except ZeroDivisionError:
                print "Oops, no events above threshold for session %d" %i

    return p

def probabilities_properties(probabilities):
    diagonals = [diagonal(probabilities, offset=i) for i in xrange(NUMBER_OF_SESSIONS)]

    averages = [mean(d) for d in diagonals]
    stds = [std(d) for d in diagonals]

    return averages, stds

def count_events_per_session(global_numbering_events, cells_indices):
    # Calculate the events rate through all sessions

    number_of_sessions = len(global_numbering_events)
    number_of_events_per_session = np.zeros((len(cells_indices), number_of_sessions))

    # Counting number of events per cell per session for the cell indices
    for i, session in enumerate(global_numbering_events):
        number_of_events_per_session[:, i] = np.sum(session[cells_indices, :], axis=1)

    return number_of_events_per_session

def calculate_ensamble_correlation(global_numbering_events, cells_indices):
    number_of_sessions = len(global_numbering_events)
    number_of_frames = global_numbering_events[0].shape[1]

    activity_vectors = []

    for session in global_numbering_events:
        activity_vectors.append(np.sum(session[cells_indices, :], axis=1).astype(float) / number_of_frames)

    rho = np.zeros((number_of_sessions, number_of_sessions))
    for i in xrange(number_of_sessions):
        for j in xrange(number_of_sessions):
            rho[i, j] = np.corrcoef(activity_vectors[i], activity_vectors[j])[0, 1]

    return rho

def analyze_and_separate_bucket_dynamics_for_edge(data):
    # Calculate recurrence probability, event rate, and ensamble correlation for a data set, given its
    # global_numbering_events and edge_cells calculated before. for all sessions
    # with separating between first and last bucket trials
    events_threshold = 5
    first_bucket_sessions_dynamics = []
    last_bucket_sessions_dynamics = []

    for session_index in xrange(NUMBER_OF_SESSIONS):

        dynamics = {}
        edge_cells = data['edge_cells'][session_index]
        bucket_events = data['global_numbering_bucket']['first']

        dynamics['recurrence'] = recurrence_probability(bucket_events, edge_cells, events_threshold)

        dynamics['ensamble_correlation'] = calculate_ensamble_correlation(bucket_events, edge_cells)

        dynamics['events_rate'] = count_events_per_session(bucket_events, edge_cells)
        first_bucket_sessions_dynamics.append(dynamics)

        dynamics = {}
        bucket_events = data['global_numbering_bucket']['last']

        dynamics['recurrence'] = recurrence_probability(bucket_events, edge_cells, events_threshold)

        dynamics['ensamble_correlation'] = calculate_ensamble_correlation(bucket_events, edge_cells)

        dynamics['events_rate'] = count_events_per_session(bucket_events, edge_cells)
        last_bucket_sessions_dynamics.append(dynamics)

    return first_bucket_sessions_dynamics, last_bucket_sessions_dynamics

def analyze_bucket_dynamics(data, cell_type):
    # Calculate recurrence probability, event rate, and ensamble correlation for
    # a data set, given its global_numbering_events and edge_cells calculated
    # before. for all sessions cell_type is either: 'edge_cells' or
    # 'non_edge_cells'
    events_threshold = 5
    bucket_sessions_dynamics = []
    number_of_neurons = data['global_numbering_bucket']['first'][0].shape[0]

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

        bucket_events = []
        for i, first_bucket_trial in enumerate(data['global_numbering_bucket']['first']):
            last_bucket_trial = data['global_numbering_bucket']['last'][i]
            session_bucket_trials = np.hstack(
                [first_bucket_trial, last_bucket_trial])
            bucket_events.append(session_bucket_trials)

        dynamics['recurrence'] = recurrence_probability\
            (bucket_events, cells_indices, events_threshold)

        dynamics['ensamble_correlation'] = calculate_ensamble_correlation\
            (bucket_events, cells_indices)

        dynamics['events_rate'] = count_events_per_session\
            (bucket_events, cells_indices)
        bucket_sessions_dynamics.append(dynamics)

    return bucket_sessions_dynamics


def analyze_bucket_dynamics_for_non_edge(data):
    # Calculate recurrence probability, event rate, and ensamble correlation for a data set, given its
    # global_numbering_events and edge_cells calculated before. for all sessions
    events_threshold = 5
    first_bucket_sessions_dynamics = []
    last_bucket_sessions_dynamics = []
    number_of_neurons = data['global_numbering_bucket']['first'][0].shape[0]

    for session_index in xrange(NUMBER_OF_SESSIONS):
        dynamics = {}
        edge_cells = data['edge_cells'][session_index]
        all_neurons_indexing = np.arange(number_of_neurons)
        non_edge_cells = all_neurons_indexing
        non_edge_cells[edge_cells] = 0
        non_edge_cells = non_edge_cells[non_edge_cells > 0]

        bucket_events = data['global_numbering_bucket']['first']

        dynamics['recurrence'] = recurrence_probability(bucket_events, non_edge_cells, events_threshold)

        dynamics['ensamble_correlation'] = calculate_ensamble_correlation(bucket_events, non_edge_cells)

        dynamics['events_rate'] = count_events_per_session(bucket_events, non_edge_cells)
        first_bucket_sessions_dynamics.append(dynamics)

        dynamics = {}
        bucket_events = data['global_numbering_bucket']['last']

        dynamics['recurrence'] = recurrence_probability(bucket_events, non_edge_cells, events_threshold)

        dynamics['ensamble_correlation'] = calculate_ensamble_correlation(bucket_events, non_edge_cells)

        dynamics['events_rate'] = count_events_per_session(bucket_events, non_edge_cells)
        last_bucket_sessions_dynamics.append(dynamics)

    return first_bucket_sessions_dynamics, last_bucket_sessions_dynamics

def plot_dynamics(first_bucket_sessions_dynamics, last_bucket_sessions_dynamics, name):
    # plot recurrence
    f, axx = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(15, 10))
    f.suptitle('Recurrence Probability ' + name, fontsize=14)
    f.tight_layout()
    f.subplots_adjust(top=0.9)

    for i in xrange(NUMBER_OF_SESSIONS):
        a, s = probabilities_properties(first_bucket_sessions_dynamics[i]['recurrence'])
        line1 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        axx[i].set_title('Session no.%d' %i)
        a, s = probabilities_properties(last_bucket_sessions_dynamics[i]['recurrence'])
        line2 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        legend((line1, line2), ('first bucket', 'last bucket'))
    f.show()

    # plot event rate
    f, axx = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(15, 10))
    f.suptitle('Event rate ' + name, fontsize=14)
    f.tight_layout()
    f.subplots_adjust(top=0.9)

    for i in xrange(NUMBER_OF_SESSIONS):
        a = mean(first_bucket_sessions_dynamics[i]['events_rate'], axis=0)
        s = std(first_bucket_sessions_dynamics[i]['events_rate'], axis=0)
        line1 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        axx[i].set_title('Session no.%d' % i)
        a = mean(last_bucket_sessions_dynamics[i]['events_rate'], axis=0)
        s = std(last_bucket_sessions_dynamics[i]['events_rate'], axis=0)
        line2 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        legend((line1, line2), ('first bucket', 'last bucket'))
    f.show()

    # plot ensamble correlation
    f, axx = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(15, 10))
    f.suptitle('Ensamble correlation ' + name, fontsize=14)
    f.tight_layout()
    f.subplots_adjust(top=0.9)

    for i in xrange(NUMBER_OF_SESSIONS):
        a, s = probabilities_properties(first_bucket_sessions_dynamics[i]['ensamble_correlation'])
        line1 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        axx[i].set_title('Session no.%d' % i)
        a, s = probabilities_properties(last_bucket_sessions_dynamics[i]['ensamble_correlation'])
        line2 = axx[i].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        legend((line1, line2), ('first bucket', 'last bucket'))

    f.show()

def average_dynamics(sessions_dynamics, field_name):
    concatenated_field = []
    for session in sessions_dynamics:
        concatenated_field.append(session[field_name])
    concatenated_field = np.stack(concatenated_field, axis=2)
    average_field = np.mean(concatenated_field, axis=2)

    return average_field

def plot_average_recurrence(first_bucket_sessions_dynamics, last_bucket_sessions_dynamics, name):
    # plot recurrence

    first_bucket_average = average_dynamics(first_bucket_sessions_dynamics, 'recurrence')
    last_bucket_average = average_dynamics(last_bucket_sessions_dynamics, 'recurrence')

    f, axx = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(15, 10))
    f.suptitle('Recurrence Probability ' + name, fontsize=14)
    f.tight_layout()
    f.subplots_adjust(top=0.9)

    a, s = probabilities_properties(first_bucket_average)
    line1 = axx.errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(last_bucket_average)
    line2 = axx.errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    legend((line1, line2), ('first bucket', 'last bucket'))
    f.show()

def plot_compare_average(nitzan_bucket_session_dynamics, bambi_bucket_session_dynamics, field_name):
    # The input dynamics are lists of the first and second dynamics
    nitzan_average = []
    bambi_average = []

    if not(field_name == 'events_rate'):
        for i in np.arange(len(nitzan_bucket_session_dynamics)):
            nitzan_average.append(average_dynamics(nitzan_bucket_session_dynamics[i], field_name))
            bambi_average.append(average_dynamics(bambi_bucket_session_dynamics[i], field_name))

        nitzan_average = np.mean(np.stack(nitzan_average, axis=2), axis=2)
        bambi_average = np.mean(np.stack(bambi_average, axis=2), axis=2)
        f, axx = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(15, 10))
        f.suptitle(field_name, fontsize=14)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        a, s = probabilities_properties(nitzan_average)
        line1 = axx.errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        a, s = probabilities_properties(bambi_average)
        line2 = axx.errorbar(arange(NUMBER_OF_SESSIONS), a, s)
        legend((line1, line2), ('nitzan', 'bambi'))

    else: # field_name == 'events_rate'
        nitzan_concatenated_field = []
        bambi_concatenated_field = []
        nitzan_number_of_cells = []
        bambi_number_of_cells = []
        for i in np.arange(len(nitzan_bucket_session_dynamics)):
            sessions_dynamics = nitzan_bucket_session_dynamics[i]
            for j, session in enumerate(sessions_dynamics):
                nitzan_concatenated_field.append(session[field_name])
                nitzan_number_of_cells.append(np.count_nonzero(session[field_name][:, j]))

            sessions_dynamics = bambi_bucket_session_dynamics[i]
            for j, session in enumerate(sessions_dynamics):
                bambi_concatenated_field.append(session[field_name])
                bambi_number_of_cells.append(np.count_nonzero(session[field_name][:, j]))

        nitzan_concatenated_field = np.vstack(nitzan_concatenated_field)
        bambi_concatenated_field = np.vstack(bambi_concatenated_field)
        nitzan_average = np.mean(nitzan_concatenated_field, axis=0)
        bambi_average = np.mean(bambi_concatenated_field, axis = 0)
        nitzan_std = np.std(nitzan_concatenated_field, axis=0)
        bambi_std = np.std(bambi_concatenated_field, axis=0)

        nitzan_concatenated_field[nitzan_concatenated_field == 0]= nan
        bambi_concatenated_field[bambi_concatenated_field == 0] = nan
        nitzan_no_zeros_average = np.nanmean(nitzan_concatenated_field, axis=0)
        bambi_no_zeros_average = np.nanmean(bambi_concatenated_field, axis=0)
        nitzan_no_zeros_std = np.nanstd(nitzan_concatenated_field, axis=0)
        bambi_no_zeros_std = np.nanstd(bambi_concatenated_field, axis=0)

        f, axx = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
        f.suptitle(field_name, fontsize=14)
        f.tight_layout()
        f.subplots_adjust(top=0.9)
        line1 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), nitzan_average, nitzan_std)
        line2 = axx[0].errorbar(arange(NUMBER_OF_SESSIONS), bambi_average, bambi_std)
        axx[0].set_title('Average of events number')
        line3 = axx[1].errorbar(arange(NUMBER_OF_SESSIONS), nitzan_no_zeros_average, nitzan_no_zeros_std)
        line4 = axx[1].errorbar(arange(NUMBER_OF_SESSIONS), bambi_no_zeros_average, bambi_no_zeros_std)
        axx[1].set_title('Average of events number (no zeros)')
        line5 = axx[2].plot(arange(NUMBER_OF_SESSIONS), nitzan_number_of_cells[:5])
        line6 = axx[2].plot(arange(NUMBER_OF_SESSIONS), bambi_number_of_cells[:5])
        axx[2].set_title('Number of cells')
        legend((line1, line2), ('nitzan', 'bambi'))

    f.show()

def plot_all_bucket_dynamics(data):
    nitzan_bucket_dynamics_edge = \
        analyze_bucket_dynamics(data['nitzan'], 'edge_cells')
    bambi_bucket_dynamics_edge = \
        analyze_bucket_dynamics(data['bambi'], 'edge_cells')
    nitzan_bucket_dynamics_non_edge = \
        analyze_bucket_dynamics(data['nitzan'], 'non_edge_cells')
    bambi_bucket_dynamics_non_edge = \
        analyze_bucket_dynamics(data['bambi'], 'non_edge_cells')

    f, axx = subplots(3, 2, sharey='row', sharex='row')
    f.subplots_adjust(top=0.9)
    ###### plot recurrence ######
    # For edge cells
    nitzan_average_edge_recurrence = average_dynamics\
        (nitzan_bucket_dynamics_edge, 'recurrence')
    bambi_average_edge_recurrence = average_dynamics \
        (bambi_bucket_dynamics_edge, 'recurrence')
    a, s = probabilities_properties(nitzan_average_edge_recurrence)
    axx[0, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_edge_recurrence)
    axx[0, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[0, 0].set_ylabel('Recurrence',fontsize=14)
    axx[0, 0].set_title('Edge cells', fontsize=20)
    axx[0, 0].set_xlabel('Session difference', fontsize=14)

    # For non edge cells
    nitzan_average_non_edge_recurrence = average_dynamics \
        (nitzan_bucket_dynamics_non_edge, 'recurrence')
    bambi_average_non_edge_recurrence = average_dynamics \
        (bambi_bucket_dynamics_non_edge, 'recurrence')
    a, s = probabilities_properties(nitzan_average_non_edge_recurrence)
    line1 = axx[0, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s, label='Phase 0')
    a, s = probabilities_properties(bambi_average_non_edge_recurrence)
    line2 = axx[0, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s, label='Phase 1')
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           handles=[line1, line2], fontsize=14)
    axx[0, 1].set_title('Non edge cells', fontsize=20)
    axx[0, 1].set_xlabel('Session difference', fontsize=14)

    ###### plot ensamble correlation ######
    # For edge cells
    nitzan_average_edge_ensamble_correltaion = \
        average_dynamics(nitzan_bucket_dynamics_edge, 'ensamble_correlation')
    bambi_average_edge_ensamble_correltaion = \
        average_dynamics(bambi_bucket_dynamics_edge, 'ensamble_correlation')

    a, s = probabilities_properties(nitzan_average_edge_ensamble_correltaion)
    axx[1, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_edge_ensamble_correltaion)
    axx[1, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1, 0].set_ylabel('Ensamble correlation', fontsize=14)
    axx[1, 0].set_xlabel('Session difference', fontsize=14)

    # For non edge cells
    nitzan_average_non_edge_ensamble_correltaion = \
        average_dynamics(nitzan_bucket_dynamics_non_edge, 'ensamble_correlation')
    bambi_average_non_edge_ensamble_correltaion = \
        average_dynamics(bambi_bucket_dynamics_non_edge, 'ensamble_correlation')

    a, s = probabilities_properties(nitzan_average_non_edge_ensamble_correltaion)
    axx[1, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_non_edge_ensamble_correltaion)
    axx[1, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1, 1].set_xlabel('Session difference', fontsize=14)

    ###### plot number of events and cells ######
    # for edge cells
    nitzan_edge_event_rate = []
    nitzan_number_of_cells = []
    # count the event rate only in active cells
    for j, session in enumerate(nitzan_bucket_dynamics_edge):
        nitzan_edge_event_rate.append(session['events_rate'])
        nitzan_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_bucket_dynamics_edge):
        bambi_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    nitzan_edge_event_rate = np.vstack(nitzan_edge_event_rate)
    bambi_edge_event_rate = np.vstack(bambi_edge_event_rate)
    nitzan_edge_event_rate[nitzan_edge_event_rate == 0] = nan
    bambi_edge_event_rate[bambi_edge_event_rate == 0] = nan
    nitzan_average_edge_event_rate = np.nanmean(nitzan_edge_event_rate, axis=0)
    bambi_average_edge_event_rate = np.nanmean(bambi_edge_event_rate, axis=0)
    nitzan_std_edge_event_rate = np.nanstd(nitzan_edge_event_rate, axis=0)
    bambi_std_edge_event_rate = np.nanstd(bambi_edge_event_rate, axis=0)

    axx[2, 0].errorbar(arange(NUMBER_OF_SESSIONS), nitzan_average_edge_event_rate,
                       nitzan_std_edge_event_rate)
    axx[2, 0].errorbar(arange(NUMBER_OF_SESSIONS), bambi_average_edge_event_rate,
                       bambi_std_edge_event_rate)
    axx[2, 0].set_ylabel('Number of events', fontsize=14)
    axx[2, 0].set_xlabel('Session number', fontsize=14)

    # For non edge cells
    nitzan_non_edge_event_rate = []
    nitzan_number_of_cells = []
    # count the event rate only in active cells
    for j, session in enumerate(nitzan_bucket_dynamics_non_edge):
        nitzan_non_edge_event_rate.append(session['events_rate'])
        nitzan_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_non_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_bucket_dynamics_non_edge):
        bambi_non_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    nitzan_non_edge_event_rate = np.vstack(nitzan_non_edge_event_rate)
    bambi_non_edge_event_rate = np.vstack(bambi_non_edge_event_rate)
    nitzan_non_edge_event_rate[nitzan_non_edge_event_rate == 0] = nan
    bambi_non_edge_event_rate[bambi_non_edge_event_rate == 0] = nan
    nitzan_average_non_edge_event_rate = np.nanmean(nitzan_non_edge_event_rate, axis=0)
    bambi_average_non_edge_event_rate = np.nanmean(bambi_non_edge_event_rate, axis=0)
    nitzan_std_non_edge_event_rate = np.nanstd(nitzan_non_edge_event_rate, axis=0)
    bambi_std_non_edge_event_rate = np.nanstd(bambi_non_edge_event_rate, axis=0)

    axx[2, 1].errorbar(arange(NUMBER_OF_SESSIONS),
                       nitzan_average_non_edge_event_rate,
                       nitzan_std_non_edge_event_rate)
    axx[2, 1].errorbar(arange(NUMBER_OF_SESSIONS),
                       bambi_average_non_edge_event_rate,
                       bambi_std_non_edge_event_rate)
    axx[2, 1].set_xlabel('Session number', fontsize=14)

    setp(axx, xticks=range(5), xticklabels=['1', '2', '3', '4', '5'])
    for i in range(3):
        for j in range(2):
            for xtick in axx[i, j].xaxis.get_major_ticks():
                xtick.label.set_fontsize(14)
            for ytick in axx[i, j].yaxis.get_major_ticks():
                ytick.label.set_fontsize(14)
            box = axx[i, j].get_position()
            axx[i, j].set_position([box.x0, box.y0 + box.height * 0.2,
                                 box.width, box.height * 0.8])
    f.suptitle('C%sM%s' %(CAGE, MOUSE), fontsize=25)
    f.show()
    return

def plot_all_track_dynamics(data): ##### EDIT THIS ####
    nitzan_bucket_dynamics_edge = \
        analyze_bucket_dynamics(data['nitzan'], 'edge_cells')
    bambi_bucket_dynamics_edge = \
        analyze_bucket_dynamics(data['bambi'], 'edge_cells')
    nitzan_bucket_dynamics_non_edge = \
        analyze_bucket_dynamics(data['nitzan'], 'non_edge_cells')
    bambi_bucket_dynamics_non_edge = \
        analyze_bucket_dynamics(data['bambi'], 'non_edge_cells')

    f, axx = subplots(3, 2, sharey='row', sharex='row')
    f.subplots_adjust(top=0.9)
    ###### plot recurrence ######
    # For edge cells
    nitzan_average_edge_recurrence = average_dynamics\
        (nitzan_bucket_dynamics_edge, 'recurrence')
    bambi_average_edge_recurrence = average_dynamics \
        (bambi_bucket_dynamics_edge, 'recurrence')
    a, s = probabilities_properties(nitzan_average_edge_recurrence)
    axx[0, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_edge_recurrence)
    axx[0, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[0, 0].set_ylabel('Recurrence',fontsize=14)
    axx[0, 0].set_title('Edge cells', fontsize=20)
    axx[0, 0].set_xlabel('Session difference', fontsize=14)

    # For non edge cells
    nitzan_average_non_edge_recurrence = average_dynamics \
        (nitzan_bucket_dynamics_non_edge, 'recurrence')
    bambi_average_non_edge_recurrence = average_dynamics \
        (bambi_bucket_dynamics_non_edge, 'recurrence')
    a, s = probabilities_properties(nitzan_average_non_edge_recurrence)
    line1 = axx[0, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s, label='Phase 0')
    a, s = probabilities_properties(bambi_average_non_edge_recurrence)
    line2 = axx[0, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s, label='Phase 1')
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           handles=[line1, line2], fontsize=14)
    axx[0, 1].set_title('Non edge cells', fontsize=20)
    axx[0, 1].set_xlabel('Session difference', fontsize=14)

    ###### plot ensamble correlation ######
    # For edge cells
    nitzan_average_edge_ensamble_correltaion = \
        average_dynamics(nitzan_bucket_dynamics_edge, 'ensamble_correlation')
    bambi_average_edge_ensamble_correltaion = \
        average_dynamics(bambi_bucket_dynamics_edge, 'ensamble_correlation')

    a, s = probabilities_properties(nitzan_average_edge_ensamble_correltaion)
    axx[1, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_edge_ensamble_correltaion)
    axx[1, 0].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1, 0].set_ylabel('Ensamble correlation', fontsize=14)
    axx[1, 0].set_xlabel('Session difference', fontsize=14)

    # For non edge cells
    nitzan_average_non_edge_ensamble_correltaion = \
        average_dynamics(nitzan_bucket_dynamics_non_edge, 'ensamble_correlation')
    bambi_average_non_edge_ensamble_correltaion = \
        average_dynamics(bambi_bucket_dynamics_non_edge, 'ensamble_correlation')

    a, s = probabilities_properties(nitzan_average_non_edge_ensamble_correltaion)
    axx[1, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    a, s = probabilities_properties(bambi_average_non_edge_ensamble_correltaion)
    axx[1, 1].errorbar(arange(NUMBER_OF_SESSIONS), a, s)
    axx[1, 1].set_xlabel('Session difference', fontsize=14)

    ###### plot number of events and cells ######
    # for edge cells
    nitzan_edge_event_rate = []
    nitzan_number_of_cells = []
    # count the event rate only in active cells
    for j, session in enumerate(nitzan_bucket_dynamics_edge):
        nitzan_edge_event_rate.append(session['events_rate'])
        nitzan_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_bucket_dynamics_edge):
        bambi_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    nitzan_edge_event_rate = np.vstack(nitzan_edge_event_rate)
    bambi_edge_event_rate = np.vstack(bambi_edge_event_rate)
    nitzan_edge_event_rate[nitzan_edge_event_rate == 0] = nan
    bambi_edge_event_rate[bambi_edge_event_rate == 0] = nan
    nitzan_average_edge_event_rate = np.nanmean(nitzan_edge_event_rate, axis=0)
    bambi_average_edge_event_rate = np.nanmean(bambi_edge_event_rate, axis=0)
    nitzan_std_edge_event_rate = np.nanstd(nitzan_edge_event_rate, axis=0)
    bambi_std_edge_event_rate = np.nanstd(bambi_edge_event_rate, axis=0)

    axx[2, 0].errorbar(arange(NUMBER_OF_SESSIONS), nitzan_average_edge_event_rate,
                       nitzan_std_edge_event_rate)
    axx[2, 0].errorbar(arange(NUMBER_OF_SESSIONS), bambi_average_edge_event_rate,
                       bambi_std_edge_event_rate)
    axx[2, 0].set_ylabel('Number of events', fontsize=14)
    axx[2, 0].set_xlabel('Session number', fontsize=14)

    # For non edge cells
    nitzan_non_edge_event_rate = []
    nitzan_number_of_cells = []
    # count the event rate only in active cells
    for j, session in enumerate(nitzan_bucket_dynamics_non_edge):
        nitzan_non_edge_event_rate.append(session['events_rate'])
        nitzan_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    bambi_non_edge_event_rate = []
    bambi_number_of_cells = []
    for j, session in enumerate(bambi_bucket_dynamics_non_edge):
        bambi_non_edge_event_rate.append(session['events_rate'])
        bambi_number_of_cells.append(
            np.count_nonzero(session['events_rate'][:, j]))

    nitzan_non_edge_event_rate = np.vstack(nitzan_non_edge_event_rate)
    bambi_non_edge_event_rate = np.vstack(bambi_non_edge_event_rate)
    nitzan_non_edge_event_rate[nitzan_non_edge_event_rate == 0] = nan
    bambi_non_edge_event_rate[bambi_non_edge_event_rate == 0] = nan
    nitzan_average_non_edge_event_rate = np.nanmean(nitzan_non_edge_event_rate, axis=0)
    bambi_average_non_edge_event_rate = np.nanmean(bambi_non_edge_event_rate, axis=0)
    nitzan_std_non_edge_event_rate = np.nanstd(nitzan_non_edge_event_rate, axis=0)
    bambi_std_non_edge_event_rate = np.nanstd(bambi_non_edge_event_rate, axis=0)

    axx[2, 1].errorbar(arange(NUMBER_OF_SESSIONS),
                       nitzan_average_non_edge_event_rate,
                       nitzan_std_non_edge_event_rate)
    axx[2, 1].errorbar(arange(NUMBER_OF_SESSIONS),
                       bambi_average_non_edge_event_rate,
                       bambi_std_non_edge_event_rate)
    axx[2, 1].set_xlabel('Session number', fontsize=14)

    setp(axx, xticks=range(5), xticklabels=['1', '2', '3', '4', '5'])
    for i in range(3):
        for j in range(2):
            for xtick in axx[i, j].xaxis.get_major_ticks():
                xtick.label.set_fontsize(14)
            for ytick in axx[i, j].yaxis.get_major_ticks():
                ytick.label.set_fontsize(14)
            box = axx[i, j].get_position()
            axx[i, j].set_position([box.x0, box.y0 + box.height * 0.2,
                                 box.width, box.height * 0.8])
    f.suptitle('C%sM%s' %(CAGE, MOUSE), fontsize=25)
    f.show()
    return

def main():
    data = {'nitzan': {},
            'bambi': {}}

    [bins_traces, events_traces, bucket_events_traces] = extract_nitzans_data()
    data['nitzan'] = set_in_data(data['nitzan'], bins_traces, events_traces, bucket_events_traces)

    [bins_traces, events_traces, bucket_events_traces] = extract_bambi_data()
    data['bambi'] = set_in_data(data['bambi'], bins_traces, events_traces, bucket_events_traces)

    # Matching all neurons' IDs to the global ID
    nitzan_registration, bambi_registration = load_cell_registration()
    data['nitzan']['global_numbering_events'] = renumber_sessions_cells_ID(data['nitzan']['events_traces'],
                                                                           nitzan_registration)

    data['bambi']['global_numbering_events'] = renumber_sessions_cells_ID(data['bambi']['events_traces'],
                                                                               bambi_registration)

    data['nitzan']['global_numbering_bucket'] = {}
    data['nitzan']['global_numbering_bucket']['first'] = renumber_sessions_cells_ID(data['nitzan']['bucket_events_traces']['first'],
                                                                           nitzan_registration)

    data['nitzan']['global_numbering_bucket']['last'] = renumber_sessions_cells_ID(
        data['nitzan']['bucket_events_traces']['last'],
        nitzan_registration)

    data['bambi']['global_numbering_bucket'] = {}
    data['bambi']['global_numbering_bucket']['first'] = renumber_sessions_cells_ID(data['bambi']['bucket_events_traces']['first'],
                                                                          bambi_registration)

    data['bambi']['global_numbering_bucket']['last'] = renumber_sessions_cells_ID(
        data['bambi']['bucket_events_traces']['last'],
        bambi_registration)

    # Finding edge cells from all days
    data['nitzan']['edge_cells'] = find_edge_cells_for_all_sessions(data['nitzan']['bins_traces'],
                                                                    data['nitzan']['global_numbering_events'],
                                                                    EDGE_PERCENT, EDGE_BINS)

    data['bambi']['edge_cells'] = find_edge_cells_for_all_sessions(data['bambi']['bins_traces'],
                                                                    data['bambi']['global_numbering_events'],
                                                                    EDGE_PERCENT, EDGE_BINS)

    # Calculate recurrence probability, event rate, and ensamble correlation for edge cells
    # [nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics] = analyze_bucket_dynamics_for_edge(data['nitzan'])
    # [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics] = analyze_bucket_dynamics_for_edge(data['bambi'])
    #
    # plot_average_recurrence(nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics, 'Nitzan - edge cells')
    # plot_average_recurrence(bambi_first_bucket_dynamics, bambi_last_bucket_dynamics, 'Bambi - edge cells')

    # plot_compare_average([nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics],
    #                      [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics],'events_rate')
    # plot_compare_average([nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics],
    #                      [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics], 'ensamble_correlation')

    # Calculate recurrence probability, event rate, and ensamble correlation for non edge cells
    # [nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics] = analyze_bucket_dynamics_for_non_edge(data['nitzan'])
    # [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics] = analyze_bucket_dynamics_for_non_edge(data['bambi'])
    #
    # plot_average_recurrence(nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics, 'Nitzan - non edge cells')
    # plot_average_recurrence(bambi_first_bucket_dynamics, bambi_last_bucket_dynamics, 'Bambi - non edge cells')

    # plot_compare_average([nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics],
    #                      [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics], 'events_rate')
    # plot_compare_average([nitzan_first_bucket_dynamics, nitzan_last_bucket_dynamics],
    #                      [bambi_first_bucket_dynamics, bambi_last_bucket_dynamics], 'ensamble_correlation')

    plot_all_bucket_dynamics(data)
    raw_input('Press enter to quit')

if __name__ == '__main__':
    main()
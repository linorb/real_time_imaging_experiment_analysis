import os.path
import csv
import cPickle

from numpy import *
from scipy.stats import *
from matplotlib.pyplot import *
from seaborn import *
import h5py
import scipy.io

from real_time_imaging.tools.activity_loading import unite_sessions
from zivlab.analysis.place_cells import find_place_cells, calculate_event_rate_distribution

# Number of sessions in all runs in the experiment
NUMBER_OF_SESSIONS = 5
BAMBI_NUMBER_OF_TRIALS = 7
MOUSE = '3'
CAGE = '40'

EDGE_BINS = [1, 2, 3, 9, 10]
EDGE_PERCENT = 0.9

CELL_REGISTRATION_FILENAME = r'Z:\Short term data storage\Data storage (1 year)\Nitzan\c40m3\registration_110_days\cellRegistered_Final_16-Mar-2017_133500.mat'

def  load_cell_registration():
    #Taken from Or's script
    # Load the cell registration results
    cell_registration = h5py.File(CELL_REGISTRATION_FILENAME)['cell_registered_struct'][
        'optimal_cell_to_index_map'].value.astype(int)
    # Compensate for 0-based indexing
    cell_registration -= 1

    nitzan_run = np.transpose(cell_registration[:5])
    bambi_run = np.transpose(cell_registration[-5:])

    return nitzan_run, bambi_run

def extract_nitzans_data():
    """Taken from OR's code dynamic_analysis"""
    full_bins_traces = []
    full_events_traces = []
    frame_logs = []

    for i in xrange(NUMBER_OF_SESSIONS):
        bins_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\tracking\Cage%s_Mouse%s' %(CAGE, MOUSE),
                                     'Day%d.mat' % (i + 1,))
        my_mvmt = scipy.io.loadmat(bins_filename)['my_mvmt'][0]
        session_bins_traces = []
        for j in xrange(1, my_mvmt.shape[0]):
            # All 3 first indices are magic to get to the required position.
            # The 1: is used to remove the first behavioral frame which is dropped
            # in the neuronal.
            session_bins_traces.append(my_mvmt[j][0][0][3][1:].T[0])
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
        events_traces.append(full_events_traces[i][:, frame_logs[i][1][0]:frame_logs[i][-1][0]])

    bucket_events_traces = {'first': [], 'last': []}
    for i in xrange(NUMBER_OF_SESSIONS):
        # The second index in frame_logs is for taking only the bucket
        bucket_events_traces['first'].append(full_events_traces[i][:, frame_logs[i][0][0]:frame_logs[i][0][1]])
        bucket_events_traces['last'].append(full_events_traces[i][:, frame_logs[i][-1][0]:frame_logs[i][-1][1]])

    bins_traces = []
    fixed_bins_traces = full_bins_traces[:]
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

    print

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
    edge_neurons = np.argwhere(percent_edge_per_neuron >= edge_percent)

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
        print session.shape
        global_session = unite_sessions([session], [i], cell_to_index_map)
        global_numbering_events.append(global_session)

    return global_numbering_events


def recurrence_probability(global_numbering_events, cells_indices):
    # Calculate recurrence probability. assuming global_numbering_events is a list of events matrices that has global
    # numbering of neurons. and cells_indices are global indices

    number_of_sessions = len(global_numbering_events)
    number_of_events_per_session = []

    # Counting number of events per cell per session for the cell indices
    for session in global_numbering_events:
        number_of_events_per_session.append(np.sum(session[cells_indices, :], axis=1))


    # Probabilities
    p = zeros((number_of_sessions, number_of_sessions))

    for i in xrange(number_of_sessions):
        for j in xrange(number_of_sessions):
            p[i, j] = float(count_nonzero(number_of_events_per_session[i] & number_of_events_per_session[j])) / \
                      float(count_nonzero(number_of_events_per_session[i]))

    return p

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


    #


main()

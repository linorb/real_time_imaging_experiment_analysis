import os.path
import csv
import cPickle
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt

from zivlab.data_structs import my_mvmt
from bambi.tools.activity_loading import unite_sessions

# Number of sessions in all runs in the experiment
# only the first 4 sessions are taken since there is a problem in Nitzan's last session in C40M3
NUMBER_OF_SESSIONS = 4
BAMBI_NUMBER_OF_TRIALS = 7
MOUSE = '6'
CAGE = '40'
ROIS_INDICES = {}

EDGE_BINS = [0, 9]
EDGE_PERCENT = 0.9
# For c40m3
CELL_REGISTRATION_FILENAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_1_preprocessed\registration\c40m3\cellRegistered_Final_16-Mar-2017_133500.mat'
ROIS_INDICES['3'] = [36, 53, 80, 89, 158, 181, 195, 229, 258, 290, 321, 336,
                     339, 357, 366, 392, 394, 399, 408, 439, 446, 448, 449,
                     465, 490]

# For c40m6
NITZAN_CELL_REGISTRATION_FILENAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_1_preprocessed\registration\c40m6\cellRegistered_linear.mat'
BAMBI_CELL_REGISTRATION_FILNAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_1_preprocessed\registration\c40m6\cellRegistered_Final_20170423_123804.mat'
ROIS_INDICES['6'] = [44, 61, 78, 96, 154, 157, 172, 195, 214, 226, 244, 247,
                     259, 261, 262, 286, 287, 290, 301, 303, 314, 337, 340,
                     346, 348, 368, 372, 374, 383, 389, 391, 407, 415, 418,
                     419, 448, 448, 460, 472, 473, 474, 479, 488, 501, 517, 569]


def fix_extra_frames(events, movement):
    # Delete the extra frames either from events or movement of a single trial
    events_length = events.shape[1]
    behavior_length = len(movement['bin'])
    if events_length != behavior_length:
        print "Delete extra frames: %d" % (events_length - behavior_length)
        if events_length > behavior_length:
            fixed_events = events[:, :behavior_length]
            return fixed_events, movement
        else:
            key_names = list(movement.keys())
            fixed_movement = {}
            for key_name in key_names:
                fixed_movement[key_name] = movement[key_name][:events_length]
            return events, fixed_movement
    else:
        return events, movement


def fix_mvmt_missing_frames(movement, missing_frames):
    """Delete the missing frames from the movement trial"""
    key_names = list(movement.keys())
    fixed_movement = {}
    for key_name in key_names:
        fixed_movement[key_name] = np.delete(movement[key_name], missing_frames)

    return fixed_movement


def load_cell_registration(mouse):
    #Taken from Or's script - for C40M3
    # Load the cell registration results
    if mouse == '3':
        cell_registration = h5py.File(CELL_REGISTRATION_FILENAME)['cell_registered_struct'][
            'optimal_cell_to_index_map'].value.astype(int)
        # Compensate for 0-based indexing
        cell_registration -= 1

        nitzan_run = np.transpose(cell_registration[:5])
        bambi_run = np.transpose(cell_registration[-5:])

        # Find the global numbering of the ROIs:
        session_neurons_indices = np.transpose(cell_registration[7])
        global_numbering_roi = np.zeros(len(ROIS_INDICES['3']), dtype=int)
        for i, cell_index in enumerate(ROIS_INDICES['3']):
            global_numbering_roi[i] = \
                np.argwhere(session_neurons_indices == cell_index)

    elif mouse == '6':
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

        # Find the global numbering of the ROIs:
        session_neurons_indices = cell_registration[0]
        global_numbering_roi = np.zeros(len(ROIS_INDICES['6']), dtype=int)
        for i, cell_index in enumerate(ROIS_INDICES['6']):
            global_numbering_roi[i] = \
                np.argwhere(session_neurons_indices == cell_index)

    return nitzan_run, bambi_run, global_numbering_roi


def extract_bambi_data(cage, mouse, cell_to_index_map):
    """Based on OR's code dynamic_analysis"""
    BASE_DIRNAME = r'D:\dev\real_time_imaging_experiment_analysis\phase_1_preprocessed'

    all_sessions_events =[]
    all_sessions_behavior = []
    for i in xrange(NUMBER_OF_SESSIONS):
        mvmt_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1),
                                     'c'+cage+'m'+mouse, 'my_mvmt.mat')
        mvmt = my_mvmt.load_my_mvmt_file(mvmt_filename)

        events_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,),
                                       'c'+cage+'m'+mouse, 'finalEventsMat.mat')
        allEventsMat = scipy.io.loadmat(events_filename)['allEventsMat'].T

        global_numbering_events = unite_sessions([allEventsMat], [i],
                                                 cell_to_index_map)

        frame_log_filename = os.path.join(BASE_DIRNAME, 'session_%d' % (i+1,),
                                          'c'+cage+'m'+mouse, 'frameLog.csv')
        frameLog = csv.reader(open(frame_log_filename, 'rb'))
        # Skip header
        frameLog.next()
        session_frame_log = []
        for line in frameLog:
            # Set indices to 0-index based, but the second index should end one
            # frame after
            session_frame_log.append([int(line[2]) - 1, int(line[3])])

        trial_events = []
        trial_behavior = []
        number_of_trials = len(session_frame_log)
        for trial_index in np.arange(1, number_of_trials-1):
            current_events = global_numbering_events[:, session_frame_log
                                [trial_index][0]:session_frame_log[trial_index][1]]

            microscope_statistics = cPickle.load(open(
                os.path.join(BASE_DIRNAME, 'session_%d' % (i + 1,),
                             'c' + cage + 'm' + mouse,
                             'linear_trial_%d' % (trial_index,),
                             'microscope_statistics.pkl')))
            current_behavior = fix_mvmt_missing_frames(mvmt[trial_index - 1],
                                    microscope_statistics['missing_frames'])

            fixed_events , fixed_behavior = fix_extra_frames(current_events, current_behavior)
            trial_events.append(fixed_events)
            trial_behavior.append(fixed_behavior)

        all_sessions_events.append(trial_events)
        all_sessions_behavior.append(trial_behavior)

    print

    return all_sessions_events, all_sessions_behavior


def extract_nitzans_data(cage, mouse, cell_to_index_map):
    """Based on OR's code dynamic_analysis"""
    all_sessions_events = []
    all_sessions_behavior = []
    for i in xrange(NUMBER_OF_SESSIONS):
        print i
        mvmt_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\tracking\Cage%s_Mouse%s' %(cage, mouse),
                                     'Day%d.mat' % (i + 1,))
        mvmt = my_mvmt.load_my_mvmt_file(mvmt_filename)

        events_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\Pre_processing\c%sm%s' %(cage, mouse),
                                       'day%d' % (i + 1,),
                                       'linear',
                                       'finalResults',
                                       'finalEventsMat.mat')
        allEventsMat = scipy.io.loadmat(events_filename)['allEventsMat'].T
        global_numbering_events = unite_sessions([allEventsMat], [i],
                                                 cell_to_index_map)

        frame_log_filename = os.path.join(r'Z:\Short term data storage\Lab members\Nitzan\nov16_data\Pre_processing\c%sm%s' %(cage, mouse),
                                          'day%d' % (i + 1,),
                                          'linear',
                                          'finalResults',
                                          'frameLog.csv')
        frameLog = csv.reader(open(frame_log_filename, 'rb'))
        # Skip header
        frameLog.next()
        session_frame_log = []
        for k, line in enumerate(frameLog):
            # one trial is missing for c40m6 session 3 trial 6
            if (MOUSE == '6') & (i == 3) & (k == 6):
                continue
            # Set indices to 0-index based, but the second index should end one frame after
            session_frame_log.append([int(line[2]) - 1, int(line[3])])

        trial_events = []
        trial_behavior = []
        number_of_trials = len(session_frame_log)
        for trial_index in np.arange(1, number_of_trials - 1):
            current_events = global_numbering_events[:, session_frame_log
                [trial_index][0]:session_frame_log[trial_index][1]]
            current_behavior = mvmt[trial_index-1]
            fixed_events, fixed_behavior = fix_extra_frames(current_events, current_behavior)
            trial_events.append(fixed_events)
            trial_behavior.append(fixed_behavior)

        all_sessions_events.append(trial_events)
        all_sessions_behavior.append(trial_behavior)
    return all_sessions_events, all_sessions_behavior


def plot_cell_activity_through_all_sessions(cell_activity, behavior, title):
    """Plots the raw activity for a cell
    Args:
        cell_activity: List of cells' activity on each trial ans session
        behavior: List of mouse location on each trial and session
        title: String of the title that would appear in the figure
        """
    eps = 0.1
    line_hight = eps
    f = plt.figure()
    number_of_sessions = len(cell_activity)
    max_location = []
    for session_index in xrange(number_of_sessions):
        number_of_trials = len(cell_activity[session_index])
        for trial_index in xrange(number_of_trials):
            trial_activity = cell_activity[session_index][trial_index] > 0
            trial_location = behavior[session_index][trial_index]['x']
            trial_velocity = behavior[session_index][trial_index]['velocity']
            max_location.append(np.max(trial_location))
            plt.plot(trial_location[trial_velocity > 0],
                     line_hight*trial_activity[trial_velocity > 0], 'og')
            plt.plot(trial_location[trial_velocity < 0],
                     line_hight * trial_activity[trial_velocity < 0], 'or')
            line_hight += eps
        all_locations = np.arange(0, np.max(max_location), 1)
        plt.plot(all_locations, line_hight*np.ones_like(all_locations), 'k')
        line_hight += eps
    plt.ylim([eps/3, line_hight])
    f.suptitle(title)
    plt.savefig(title + '.tiff')
    plt.close(f)

def main():
    nitzan_registration, bambi_registration, ROI_global_indices = \
        load_cell_registration(MOUSE)

    bambi_events, bambi_behavior = extract_bambi_data(CAGE, MOUSE, bambi_registration)
    nitzan_events, nitzan_behavior = extract_nitzans_data(CAGE, MOUSE, nitzan_registration)

    number_of_sessions = len(bambi_events)
    number_of_cells = bambi_events[0][0].shape[0]
    for cell_index in xrange(number_of_cells):
        cell_activity = [[x[cell_index, :] for x in bambi_events[i]] for i in
                         xrange(number_of_sessions)]
        title = 'Cell no. %d Bambi experiment' % cell_index
        if cell_index in ROI_global_indices:
            title = 'Cell no. %d Bambi experiment - Chosen cell' % cell_index
        plot_cell_activity_through_all_sessions(cell_activity, bambi_behavior,
                                                title)

    number_of_sessions = len(nitzan_events)
    number_of_cells = nitzan_events[0][0].shape[0]
    for cell_index in xrange(number_of_cells):
        cell_activity = [[x[cell_index, :] for x in nitzan_events[i]] for i in
                         xrange(number_of_sessions)]
        title = 'Cell no. %d Nitzan experiment' % cell_index
        plot_cell_activity_through_all_sessions(cell_activity, nitzan_behavior,
                                                title)
    raw_input('Press Enter')

if __name__ == '__main__':
    main()
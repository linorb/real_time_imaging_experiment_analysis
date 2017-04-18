"""Statistics on events analysis
"""
import ConfigParser
import os
import time
import cPickle

import numpy as np
import scipy.io as sio

from real_time_imaging.tools.activity_loading import *
from real_time_imaging.tools.matlab import *

MOUSE = '3'
CAGE = '40'
RESULTS_PATH = r'D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\bucket_experiment'
MOVEMENT_PATH = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\%s\c%sm%s\tracking\C%sM%s_Day00%d.mat'
REAL_TIME_PATH = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\%s\c%sm%s\real_time_imaging'
LINEAR_EXPERIMENT_DATES = ['20170219',
                           '20170222',
                           '20170225',
                           '20170228',
                           '20170303']
BUCKET_EXPERIMENT_DATES = ['20170305',
                           '20170306',
                           '20170307',
                           '20170308',
                           '20170309']

DAYS = range(4, 9)
# FOR C40M3:
CHOSEN_CELLS = np.array([36, 53, 80, 89, 158, 181, 195, 229, 258,
                         290, 321, 336, 339, 357, 366, 392, 394,
                         399, 408, 439, 446, 448, 449, 465, 490])

# FOR C40M6:
# CHOSEN_CELLS = np.array([44, 61, 78, 96, 154, 157, 172, 195,
#                          214, 226, 244, 247, 259, 261, 262,
#                          286, 287, 290, 301, 303, 314, 337,
#                          340, 346, 348, 368, 372, 374, 383,
#                          389, 391, 407, 415, 418, 419, 448,
#                          448, 460, 472, 473, 474, 479, 488,
#                          501, 517, 569])


NUMBER_OF_ACTIVE_ROIS = 2
NUMBER_OF_PERMUTATIONS = 1000
F0_FRAMES = 200
MIDDLE_TRIALS_INDICES = range(1,5)
EDGE_BINS = [0, 1, 2, 8, 9]

def create_output_directory(dirname):
    try:
        os.mkdir(dirname)
    except WindowsError:
        # Directory already exists
        pass

def randomize_partial_events_mat(events, number_of_rois):
    """Create partial events matrix from the input events, that contains a
    only part of the ROIs in a random manner according to number_of_rois
     Args:
       events: [number of neurons, number of frames]
       number_of_rois: number of rois taken randomly from the full events
       matrix
     Return:
        partial_events_mat: [number_of_rois, number of frames]
    """

    number_of_neurons = events.shape[0]
    new_perm = np.random.permutation(number_of_neurons)[:number_of_rois]
    partial_events_mat = events[new_perm,:]

    return partial_events_mat

def count_number_of_activations(events, number_of_active_rois):
    """Count the number of activations in each frame in events matrix
    Args:
           events: [number of neurons, number of frames]
           number_of_active_rois: number of active rois together that creates an activation
    Return:
            number_of_activations: [1, number of neurons]
    """
    number_of_events_per_frame = np.sum(events, axis=0)
    number_of_activations = np.sum(number_of_events_per_frame > number_of_active_rois)

    return number_of_activations

def find_p_value_for_activation(events, number_of_rois,
                                number_of_active_rois,
                                number_of_permutations,
                                number_of_original_activations):
    """Find the P value for the number of activations in the original events
    matrix that was created at the experiment itself
    Args:
           events: [number of neurons, number of frames]
           number_of_rois: number of rois taken randomly from the full events
           matrix
           number_of_active_rois: number of active rois together that creates an activation
           number_of_permutations: number of permutations for calculating p
           value
           number_of_original_activations: the number of activations in the
           original run
    Return:
        p_value: int
    """
    number_of_activations = np.zeros((number_of_permutations))
    for i in range(number_of_permutations):
        partial_events_mat = randomize_partial_events_mat(events, number_of_rois)
        number_of_activations[i] = count_number_of_activations(partial_events_mat, number_of_active_rois)
    p_value = np.sum(number_of_activations < number_of_original_activations)/ float(number_of_permutations)

    return p_value, number_of_activations

def delete_rois_from_event_matrix(events, chosen_rois):
    """Delete the chosen ROIs from the event matrix
    Args:
        events: [number of neurons, number of frames]
        chosen_rois: array of ROIs indices
    Returns:
        chosen_events: [number of neurons - number of chosen rois, number of frames]
    """
    number_of_neurons = events.shape[0]
    neurons_indexing = np.arange(number_of_neurons)
    neurons_indexing[chosen_rois] = 0
    only_wanted_neurons = neurons_indexing > 0
    chosen_events = events[only_wanted_neurons, :]

    return chosen_events

def match_path(config_filename):
    """Match the config file that ran on a offline microscope mode to
    the path of the online results"""
    config = ConfigParser.RawConfigParser()
    config.read(config_filename)
    movie_filename = config.get('offline', 'input_video_file')
    wanted_index = movie_filename.index('frame')
    results_path = movie_filename[:wanted_index]

    return results_path

def calculate_p_value_for_one_trial(results_path, chosen_cells, number_of_active_rois):
    """Calculate P value for a trial. make the complete procedures,
     including the matching to the right live events"""
    # Loading events and real-time events
    # config_filename = results_path + "\\" + "config.txt"
    events_filename = results_path + "\\" + "events.npy"
    events = np.load(events_filename)

    # Live events are the events of the real experiment
    live_events = events[chosen_cells, :]
    # live_events_path = match_path(config_filename)
    # live_events = np.load(live_events_path + "events.npy")

    number_of_neurons = events.shape[0]

    # Creating an array that contains the numbers of cells that were not chosen:
    not_chosen_events = delete_rois_from_event_matrix(events, chosen_cells)

    number_of_rois = len(chosen_cells)

    number_of_original_activations = count_number_of_activations(live_events,
                                                                 number_of_active_rois)

    [p_value, number_of_activations] = find_p_value_for_activation(not_chosen_events,
                                          number_of_rois,
                                          number_of_active_rois,
                                          number_of_permutations,
                                          number_of_original_activations)

    return p_value, number_of_activations, number_of_original_activations

def concatenate_all_session_events(session_path, trials_indices):
    list_of_trial_dirs = extract_certain_trial_dirs(session_path, trials_indices)
    all_session_events = []

    for trial_dir_name in list_of_trial_dirs:
        results_path = session_path + "\\" + trial_dir_name
        trial_events = np.load(results_path + "\\" + "events.npy")
        all_session_events.append(trial_events)

    # Concatenate all events ot trials to one events mat of the whole session
    all_session_events = np.concatenate(all_session_events, axis=1)

    return all_session_events

def calculate_p_value_for_one_session(all_session_events, chosen_cells, number_of_active_rois, number_of_permutations):

    chosen_rois_events = all_session_events[chosen_cells, :]
    number_of_original_activations = count_number_of_activations(chosen_rois_events, number_of_active_rois)
    number_of_rois = len(chosen_cells)

    number_of_neurons = all_session_events.shape[0]

    # Creating an array that contains the numbers of cells that were not chosen:
    not_chosen_events = delete_rois_from_event_matrix(all_session_events, chosen_cells)

    [p_value_entire_session, number_of_activations_all_permutations] = find_p_value_for_activation(not_chosen_events,
                                                                  number_of_rois,
                                                                  number_of_active_rois,
                                                                  number_of_permutations,
                                                                  number_of_original_activations)

    return  p_value_entire_session, number_of_activations_all_permutations, number_of_original_activations

def main():
    """This main is for a full experiment p value calculation (more then one session)"""
    full_path = RESULTS_PATH
    list_of_sessions = os.listdir(full_path)
    permutations_number_of_activations = []
    p_value_per_session = []
    all_session_activations = []
    for dir_name in list_of_sessions:
        if dir_name[:4] == '2017':
            session_path = full_path + "\\" + dir_name + "\\" + 'c' + CAGE + 'm' + MOUSE
            all_session_events = concatenate_all_session_events(session_path, MIDDLE_TRIALS_INDICES)
            [p_value, number_of_activations, number_of_original_activations] = \
                calculate_p_value_for_one_session(all_session_events, CHOSEN_CELLS, NUMBER_OF_ACTIVE_ROIS, NUMBER_OF_PERMUTATIONS)
            p_value_per_session.append(p_value)
            permutations_number_of_activations.append(number_of_activations)
            all_session_activations.append(number_of_original_activations)

    # Saving results

    timed_output_dirname = '%s_%s' % (RESULTS_PATH + r'\p_values\c' + CAGE + 'm' + MOUSE + r'\p',
                                      time.strftime('%Y_%m_%d__%H_%M_%S'))
    create_output_directory(timed_output_dirname)

    np.savez(timed_output_dirname + r'\p_values.npz',
             p_value_per_session = p_value_per_session,
             all_permutatioms_activations = permutations_number_of_activations,
             all_session_activations = all_session_activations)

    sio.savemat(timed_output_dirname + r'\p_values.mat',
                {'p_value_per_session': p_value_per_session,
                 'all_permutatioms_activations': permutations_number_of_activations,
                 'all_session_activations': all_session_activations})

def calculate_p_correct_for_session(all_session_events, bins, number_of_active_rois, edge_bins):
    """Calculate P correct for one session. take only the activated frames,
    and calculate the precent of them in the edges """
    number_of_activations = np.sum(all_session_events, axis=0)
    active_frames_indices = number_of_activations > number_of_active_rois
    activated_bins = bins[active_frames_indices]

    correct_activations = np.zeros(activated_bins.shape, dtype=bool)
    for bin in edge_bins:
        correct_activations[activated_bins == bin] = True

    p_correct = float(np.sum(correct_activations))/activated_bins.shape[0]

    return p_correct

def extract_certain_trial_dirs(results_path, trials_indices):
    list_of_dirs = os.listdir(results_path)
    list_of_wanted_dirs = []
    for dir in list_of_dirs:
        if dir[:7] == 'results':
            list_of_wanted_dirs.append(dir)
    list_of_wanted_dirs = [list_of_wanted_dirs[i] for i in trials_indices]

    return list_of_wanted_dirs

def return_wanted_indices(not_wanted_indices, range_of_indices):
    all_indices = np.arange(range_of_indices)
    all_indices[not_wanted_indices] = 0
    wanted_indices = all_indices > 0

    return wanted_indices

def concatenate_bins_for_real_time_events(movement_data, results_path, trials_indices):
    """Create concatenated array of bins that fits the real_time events matrix that starts from frame 200
    and has missing frames unlike the function in real_time_imaging.tools.activity_loading that takes all
    the frames"""
    key = 'bin'

    linear_trials_dirs = extract_certain_trial_dirs(results_path, trials_indices)

    concatenated_info = []
    for i, trial_index in enumerate(trials_indices):
        microscope_statistic_filename = results_path + "\\" + linear_trials_dirs[i] + "\\" + r'microscope_statistics.pkl'
        microscope_statistic = cPickle.load(open(microscope_statistic_filename, "rb"))
        missing_frames = microscope_statistic['missing_frames']
        good_frames = return_wanted_indices(missing_frames, 1600)

        rel_info = movement_data[trial_index][key][F0_FRAMES:]
        try:
            concatenated_info.append(rel_info[good_frames])
        except IndexError:
            print "mismatch sizes: good_frames: %d , rel_info: %d" %(len(good_frames), len(rel_info))
            min_len = min(len(good_frames), len(rel_info))
            good_frames = good_frames[:min_len]
            rel_info = rel_info[:min_len]
            concatenated_info.append(rel_info[good_frames])

    concatenated_bins = np.concatenate(concatenated_info)

    return concatenated_bins

def create_match_events_and_bins(movement_data, results_path, trials_indices, real_time_path):
    key = 'bin'

    linear_trials_dirs = extract_certain_trial_dirs(results_path, trials_indices)
    if real_time_path:
        real_time_trials_dirs = extract_certain_trial_dirs(real_time_path, trials_indices)

    concatenated_bins = []
    concatenated_events = []
    for i, trial_index in enumerate(trials_indices):
        trial_path = results_path + "\\" + linear_trials_dirs[i]

        if not real_time_path:
            microscope_statistic_filename = trial_path + "\\" + r'microscope_statistics.pkl'
        else:
            microscope_statistic_filename = real_time_path + "\\" + real_time_trials_dirs[i] +\
                                            "\\" + r'microscope_statistics.pkl'

        microscope_statistic = cPickle.load(open(microscope_statistic_filename, "rb"))
        missing_frames = microscope_statistic['missing_frames']
        good_frames = return_wanted_indices(missing_frames, 1600)

        rel_bin_info = movement_data[trial_index][key][F0_FRAMES:]

        trial_events = np.load(trial_path + "\\" + r'events.npy')

        if len(rel_bin_info) == len(good_frames):
            rel_bin_info = rel_bin_info[good_frames]

            if len(rel_bin_info) == trial_events.shape[1]:
                concatenated_bins.append(rel_bin_info)
                concatenated_events.append(trial_events)
            else:
                print "mismatch sizes: trial_events: %d , rel_bin_info: %d" % (trial_events.shape[1], len(rel_bin_info))
                min_len = min(trial_events.shape[1], len(rel_bin_info))
                trial_events = trial_events[:, :min_len]
                rel_bin_info = rel_bin_info[:min_len]
                concatenated_bins.append(rel_bin_info)
                concatenated_events.append(trial_events)
        else:
            print "mismatch sizes: good_frames: %d , rel_bin_info: %d" % (len(good_frames), len(rel_bin_info))
            min_len = min(len(good_frames), len(rel_bin_info))
            good_frames = good_frames[:min_len]
            rel_bin_info = rel_bin_info[:min_len]
            rel_bin_info = rel_bin_info[good_frames]

            if len(rel_bin_info) == trial_events.shape[1]:
                concatenated_bins.append(rel_bin_info)
                concatenated_events.append(trial_events)
            else:
                print "mismatch sizes: trial_events: %d , rel_bin_info: %d" % (trial_events.shape[1], len(rel_bin_info))
                min_len = min(trial_events.shape[1], len(rel_bin_info))
                trial_events = trial_events[:, :min_len]
                rel_bin_info = rel_bin_info[:min_len]
                concatenated_bins.append(rel_bin_info)
                concatenated_events.append(trial_events)

    concatenated_bins = np.concatenate(concatenated_bins)
    concatenated_events = np.concatenate(concatenated_events, axis=1)

    return concatenated_bins, concatenated_events

def main2():
    """This main is for calculating P correct in the linear track experiment"""
    full_path = RESULTS_PATH
    list_of_sessions = LINEAR_EXPERIMENT_DATES
    p_correct_all_sessions = []
    for i, dir_name in enumerate(list_of_sessions):
        movement_file_path = MOVEMENT_PATH %(dir_name, CAGE, MOUSE, CAGE, MOUSE, DAYS[i])
        movement_data = load_mvmt_file(movement_file_path)
        real_time_results_path = REAL_TIME_PATH %(dir_name, CAGE, MOUSE)

        [bins, all_session_events] = create_match_events_and_bins(movement_data, real_time_results_path, MIDDLE_TRIALS_INDICES, [])
        p_correct = calculate_p_correct_for_session(all_session_events, bins , NUMBER_OF_ACTIVE_ROIS, EDGE_BINS)
        p_correct_all_sessions.append(p_correct)

    print p_correct_all_sessions

    timed_output_dirname = '%s_%s' % (RESULTS_PATH + r'\p_correct\c' + CAGE + 'm' + MOUSE + r'\p',
                                      time.strftime('%Y_%m_%d__%H_%M_%S'))
    create_output_directory(timed_output_dirname)

    np.savez(timed_output_dirname + r'\p_correct.npz',
             p_correct_all_sessions = p_correct_all_sessions)

    sio.savemat(timed_output_dirname + r'\p_correct.mat',
                {'p_correct_all_sessions': p_correct_all_sessions})

def main3():
    """This main computes the significance of the activity in the edges"""
    full_path = RESULTS_PATH
    list_of_sessions = LINEAR_EXPERIMENT_DATES
    p_value_per_session = []
    permutations_number_of_activations = []
    all_session_activations = []

    for i, dir_name in enumerate(list_of_sessions):
        movement_file_path = MOVEMENT_PATH % (dir_name, CAGE, MOUSE, CAGE, MOUSE, DAYS[i])
        movement_data = load_mvmt_file(movement_file_path)
        results_path = RESULTS_PATH + "\\" + dir_name + '\c' + CAGE + 'm' + MOUSE
        real_time_results_path = REAL_TIME_PATH %(dir_name, CAGE, MOUSE)

        [bins, all_session_events] = create_match_events_and_bins(movement_data, results_path,
                                                                  MIDDLE_TRIALS_INDICES, real_time_results_path)
        # find the edge bin's indices
        edge_bins_indices = np.zeros(bins.shape, dtype=bool)
        for bin in EDGE_BINS:
            edge_bins_indices[bins == bin] = True

        edge_events = all_session_events[:, edge_bins_indices]
        [p_value, number_of_activations, number_of_original_activations] = \
            calculate_p_value_for_one_session(edge_events,CHOSEN_CELLS , NUMBER_OF_ACTIVE_ROIS,
                                              NUMBER_OF_PERMUTATIONS)
        p_value_per_session.append(p_value)
        permutations_number_of_activations.append(number_of_activations)
        all_session_activations.append(number_of_original_activations)

        # Saving results
    print p_value_per_session

    timed_output_dirname = '%s_%s' % (RESULTS_PATH + r'\p_values\c' + CAGE + 'm' + MOUSE + r'\p',
                                      time.strftime('%Y_%m_%d__%H_%M_%S'))
    create_output_directory(timed_output_dirname)

    np.savez(timed_output_dirname + r'\p_values_edges.npz',
             p_value_per_session=p_value_per_session,
             all_permutatioms_activations=permutations_number_of_activations,
             all_session_activations=all_session_activations)

    sio.savemat(timed_output_dirname + r'\p_values_edges.mat',
                {'p_value_per_session': p_value_per_session,
                 'all_permutatioms_activations': permutations_number_of_activations,
                 'all_session_activations': all_session_activations})

def create_full_session_water_frames(session_path, trials_indices):
    list_of_trial_dirs = extract_certain_trial_dirs(session_path, trials_indices)
    all_session_water_frames = []

    for trial_dir_name in list_of_trial_dirs:
        full_filename = session_path + "\\" + trial_dir_name + r'\water_dispensed_frames.pkl'
        water_frames = cPickle.load(open(full_filename, "rb"))
        all_session_water_frames.append(water_frames)

    sio.savemat(session_path + '\water_dispensed_frames.mat', \
                {'all_session_water_frames': all_session_water_frames})

    return all_session_water_frames

def main4():
    """This main makes water dispensong frasmes for all frames"""
    for dir_name in BUCKET_EXPERIMENT_DATES:
        session_path = REAL_TIME_PATH %(dir_name, CAGE, MOUSE)
        create_full_session_water_frames(session_path, MIDDLE_TRIALS_INDICES)


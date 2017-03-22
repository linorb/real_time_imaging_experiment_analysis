"""Statistics on events analysis
"""
import ConfigParser
import os
import time

import numpy as np
import scipy.io as sio

MOUSE = '3'
CAGE = '40'
RESULTS_PATH = r'D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment'

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

def calculate_p_value_for_one_trial(results_path, chosen_cells):
    """Calculate P value for a trial. make the complete procedures,
     including the matching to the right live events"""
    # Loading events and real-time events
    config_filename = results_path + "\\" + "config.txt"
    events_filename = results_path + "\\" + "events.npy"
    events = np.load(events_filename)

    # Live events are the events of the real experiment
    live_events = events[chosen_cells, :]
    # live_events_path = match_path(config_filename)
    # live_events = np.load(live_events_path + "events.npy")

    number_of_neurons = events.shape[0]

    # Creating an array that contains the numbers of cells that were not chosen:
    rest_of_cells = np.arange(number_of_neurons)
    rest_of_cells[chosen_cells] = 0
    rest_of_cells = rest_of_cells > 0

    not_chosen_events = events[rest_of_cells, :]

    number_of_rois = len(chosen_cells)
    number_of_active_rois = 2
    number_of_permutations = 1000
    number_of_original_activations = count_number_of_activations(live_events,
                                                                 number_of_active_rois)

    [p_value, number_of_activations] = find_p_value_for_activation(not_chosen_events,
                                          number_of_rois,
                                          number_of_active_rois,
                                          number_of_permutations,
                                          number_of_original_activations)

    return p_value, number_of_activations, number_of_original_activations

def calculate_p_value_for_one_session(session_path, chosen_cells):
    list_of_dirs = os.listdir(session_path)
    p_value_entire_session = []
    number_of_activations_entire_session = []
    number_of_original_activations = []
    for dir_name in list_of_dirs:
        if dir_name[:7] == 'results':
            results_path = session_path + "\\" + dir_name
            [p_value, number_of_activations, original_activation] = calculate_p_value_for_one_trial(results_path, chosen_cells)
            p_value_entire_session.append(p_value)
            number_of_activations_entire_session.append(number_of_activations)
            number_of_original_activations.append(original_activation)

    return  p_value_entire_session, number_of_activations_entire_session, number_of_original_activations

def main():
    """This main is for a full experiment p value calculation (more then one session)"""
    full_path = RESULTS_PATH
    list_of_sessions = os.listdir(full_path)
    permutations_number_of_activations = []
    p_value_per_trial = []
    all_session_activations = []
    for dir_name in list_of_sessions:
        if dir_name[:4] == '2017':
            session_path = full_path + "\\" + dir_name + "\\" + 'c' + CAGE + 'm' + MOUSE
            [p_value, number_of_activations, number_of_original_activations] = \
                calculate_p_value_for_one_session(session_path, CHOSEN_CELLS)
            p_value_per_trial.append(p_value)
            permutations_number_of_activations.append(number_of_activations)
            all_session_activations.append(number_of_original_activations)

    # Saving results

    timed_output_dirname = '%s_%s' % (RESULTS_PATH + r'\p_values\c' + CAGE + 'm' + MOUSE + r'\p',
                                      time.strftime('%Y_%m_%d__%H_%M_%S'))
    create_output_directory(timed_output_dirname)

    np.savez(timed_output_dirname + r'\p_values.npz',
             p_value_per_trial = p_value_per_trial,
             all_permutatioms_activations = permutations_number_of_activations,
             all_session_activations = all_session_activations)

    sio.savemat(timed_output_dirname + r'\p_values.mat',
                {'p_value_per_trial': p_value_per_trial,
                 'all_permutatioms_activations': permutations_number_of_activations,
                 'all_session_activations': all_session_activations})


main()

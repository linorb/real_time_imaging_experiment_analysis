"""Statistics on events analysis
"""
import ConfigParser
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

MOUSE = '3'
CAGE = '40'
RESULTS_PATH = r'D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi'
CHOSEN_CELLS = np.array([36, 53, 80, 89, 158, 181, 195, 229, 258, 290, 321, 336,
                339, 357, 366, 392, 394, 399, 408, 439, 446, 448, 449, 465, 490])

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

    return p_value

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
    live_results_path = match_path(config_filename)
    live_events_filename = live_results_path + "\\" + "events.npy"
    live_events = np.load(live_events_filename)

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

    p_value = find_p_value_for_activation(not_chosen_events,
                                          number_of_rois,
                                          number_of_active_rois,
                                          number_of_permutations,
                                          number_of_original_activations)

    return p_value

def calculate_p_value_for_one_session(session_path, chosen_cells):
    list_of_dirs = os.listdir(session_path)
    p_value = []
    for dir_name in list_of_dirs:
        if dir_name[:7] == 'results':
            results_path = session_path + "\\" + dir_name
            p_value.append(calculate_p_value_for_one_trial(results_path, chosen_cells))

    session_p_value = np.mean(p_value)
    session_std = np.std(p_value)

    return session_p_value, session_std, p_value

def main():
    """This main is for a full experiment p value calculation (more then one session)"""
    full_path = RESULTS_PATH
    list_of_sessions = os.listdir(full_path)
    all_sessions_p_value = []
    all_sessions_std = []
    p_value_per_trial = []
    for dir_name in list_of_sessions:
        if dir_name[:6] == '201703':
            session_path = full_path + "\\" + dir_name + "\\" + 'c' + CAGE + 'm' + MOUSE
            [session_p_value, session_std, p_value] = \
                calculate_p_value_for_one_session(session_path, CHOSEN_CELLS)
            all_sessions_p_value.append(session_p_value)
            all_sessions_std.append(session_std)
            p_value_per_trial.append(p_value)

    # Saving results
    all_sessions_p_value = np.asarray(all_sessions_p_value)
    all_sessions_std = np.asarray(all_sessions_std)

    np.savez(RESULTS_PATH + r'\p_values.npz', all_sessions_p_value = all_sessions_p_value,
                                             all_sessions_std = all_sessions_std,
                                             p_value_per_trial = p_value_per_trial)
    sio.savemat(RESULTS_PATH + r'\p_values.mat',
                    {'all_sessions_p_value': all_sessions_p_value,
                     'all_sessions_std': all_sessions_std,
                     'p_value_per_trial': p_value_per_trial})

    # Plotting the P values for all sessions
    plt.figure()
    plt.errorbar(all_sessions_p_value, all_sessions_std)
    plt.title("P values across sessions")

main()

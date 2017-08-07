import os

from bambi.tools import matlab
from bambi.tools.activity_loading import *
from bambi.analysis import maximum_likelihood
from zivlab.analysis.place_cells import find_place_cells

from ppgui_data_analysis import extract_nitzans_data, extract_bambi_data, \
                                set_in_data


EDGE_BINS = [0, 9]
FRAME_RATE = 10 #Hz
MOUSE = ['3', '6']
CAGE = ['40', '40']
VELOCITY_THRESHOLD = 5

def find_place_cells_for_all_sessions(bins_traces, velocity_traces,
                                      events_traces):
    all_sessions_place_cells = []
    for events, bins, velocity in zip(events_traces, bins_traces,
                                      velocity_traces):
        velocity_positive = velocity > VELOCITY_THRESHOLD
        velocity_negative = velocity < -VELOCITY_THRESHOLD
        place_cells_positive, _, _ = find_place_cells\
            (bins[velocity_positive], events[:, velocity_positive])

        place_cells_negative, _, _ = find_place_cells\
            (bins[velocity_negative], events[:, velocity_negative])

        place_cells = np.concatenate\
            ([place_cells_positive, place_cells_negative])
        place_cells = np.unique(place_cells)
        #
        all_sessions_place_cells.append(place_cells)

    return all_sessions_place_cells

def calculate_p_neuron_bin_directions_for_all_sessions\
                (bins_traces, velocity_traces, events_traces, all_place_cells):
    all_p_neron_bins = []
    for events, bins, velocity, place_cells in zip(events_traces, bins_traces,
                                      velocity_traces, all_place_cells):
        p_neuron_bin = {}
        p_neuron_bin_positive = maximum_likelihood.calculate_p_r_s_matrix\
            (bins[velocity > VELOCITY_THRESHOLD],
             events[place_cells, :][:, velocity > VELOCITY_THRESHOLD])
        p_neuron_bin['positive'] = [p_neuron_bin_positive]

        p_neuron_bin_negative = maximum_likelihood.calculate_p_r_s_matrix\
            (bins[velocity < -VELOCITY_THRESHOLD],
             events[place_cells, :][:, velocity < -VELOCITY_THRESHOLD])
        p_neuron_bin['negative'] = [p_neuron_bin_negative]

        all_p_neron_bins.append(p_neuron_bin)

    return all_p_neron_bins

def test_bucket_trial(events, p_neuron_bin, edge_bins):
    number_of_frames = events.shape[1]

    decoded_bins = np.zeros((number_of_frames))

    # Decode each frame in events:
    for frame in range(number_of_frames):
        if np.sum(events[:, frame]) > 0:
            # Decode  using environments for separating directions
            decoded_bins[frame], _ = \
                decode_most_likely_bin_and_environment(
                np.expand_dims(events[:, frame], axis=1), p_neuron_bin)          
        else:
            decoded_bins[frame] = np.nan
          
    return decoded_bins
    
def test_all_bucket_trials(data, name):
    # test all bucket trials that exist in data[name]
    decoded_bins_first_bucket = []
    decoded_bins_last_bucket = []
    number_of_sessions = len(data[name]['events_traces'])
    
    for session_index in range(number_of_sessions):
        p_neuron_bin = data[name]['p_neuron_bin'][session_index]
        place_cells = data[name]['place_cells'][session_index]
        first_bucket_events = data[name]['bucket_events_traces']['first']\
                                                                [session_index]
        decoded_bins_first_bucket.append\
            (test_bucket_trial(first_bucket_events[place_cells,:],
                               p_neuron_bin, EDGE_BINS))
        last_bucket_events = data[name]['bucket_events_traces']['last']\
                                                               [session_index]
        decoded_bins_last_bucket.append\
            (test_bucket_trial(last_bucket_events[place_cells, :],
                               p_neuron_bin, EDGE_BINS))
    
    return decoded_bins_first_bucket, decoded_bins_last_bucket

def main():
    for i, mouse in enumerate(MOUSE):
        data = {'nitzan': {},
                'bambi': {}}
        #### For Nitzan
        # Load data
        [bins_traces, velocity_traces, events_traces,
         bucket_events_traces] = extract_nitzans_data(CAGE[i], mouse)
        data['nitzan'] = set_in_data(data['nitzan'], bins_traces, events_traces,
                                     bucket_events_traces)
        data['nitzan']['velocity_traces'] = velocity_traces
        # Find place cells
        data['nitzan']['place_cells'] = find_place_cells_for_all_sessions(
            bins_traces, velocity_traces, events_traces)
        # Calculate p_neuron_bin for all sessions
        data['nitzan']['p_neuron_bin'] = \
            calculate_p_neuron_bin_directions_for_all_sessions(bins_traces, 
                velocity_traces, events_traces, data['nitzan']['place_cells'])
        decoded_bins_first_bucket, decoded_bins_last_bucket = \
            test_all_bucket_trials(data, 'nitzan')
        data['nitzan']['decoded_bucket_bins'] = {}
        data['nitzan']['decoded_bucket_bins']['first'] = \
            decoded_bins_first_bucket
        data['nitzan']['decoded_bucket_bins']['last'] = \
            decoded_bins_last_bucket

        #### For bambi:
        # Load data
        [bins_traces, velocity_traces, events_traces, bucket_events_traces] = \
            extract_bambi_data(CAGE[i], mouse)
        data['bambi'] = set_in_data(data['bambi'], bins_traces, events_traces,
                                    bucket_events_traces)
        data['bambi']['velocity_traces'] = velocity_traces
        # Find place cells
        data['bambi']['place_cells'] = find_place_cells_for_all_sessions(
            bins_traces, velocity_traces, events_traces)
        # Calculate p_neuron_bin for all sessions
        data['bambi']['p_neuron_bin'] = \
            calculate_p_neuron_bin_directions_for_all_sessions(bins_traces, 
                velocity_traces, events_traces, data['bambi']['place_cells'])
        decoded_bins_first_bucket, decoded_bins_last_bucket = \
            test_all_bucket_trials(data, 'bambi')
        data['bambi']['decoded_bucket_bins'] = {}
        data['bambi']['decoded_bucket_bins']['first'] = \
            decoded_bins_first_bucket
        data['bambi']['decoded_bucket_bins']['last'] = \
            decoded_bins_last_bucket
            
        np.savez('bucket_decoding_results_c%sm%s' % (CAGE[i], mouse),
            nitzan_first_bucket = data['nitzan']['decoded_bucket_bins']['first'],
            nitzan_last_bucket = data['nitzan']['decoded_bucket_bins']['last'],
            bambi_first_bucket = data['bambi']['decoded_bucket_bins']['first'],
            bambi_last_bucket = data['bambi']['decoded_bucket_bins']['last'])
        
if __name__ == '__main__':
    main()

        


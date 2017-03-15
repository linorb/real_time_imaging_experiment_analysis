import numpy as np
import scipy.io as sio

from real_time_imaging.analysis.detector import GrowingWindowDetector

dir_path = r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\2017_02_01\c40m3\real_time_imaging\results\results_2017_02_01__15_54_17'
traces_filename = dir_path + r'\trace.npy'

trace = np.load(traces_filename)

fixed_traces = []
baseline = []
for current_trace in trace:
    detector = GrowingWindowDetector(absolute_median_threshold=5, baseline_window_size=20,
                                     length_of_event=4, smoothing_flag=True)
    for current_value in current_trace:
        detector.detect(current_value)
    fixed_traces.append(detector.fixed_trace)
    baseline.append(detector.baseline)

sio.savemat(dir_path + r'\fixed_trace.mat', {'fixed_trace': fixed_traces})
sio.savemat(dir_path + r'\baseline.mat', {'baseline': baseline})


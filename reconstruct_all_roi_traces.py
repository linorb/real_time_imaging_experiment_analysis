import ConfigParser
import os
import subprocess

CONFIG_FILE_PATH = r'D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\20170309\c40m6'
VIDEOS_PATH =  r'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging\20170219\c40m3\real_time_imaging'

list_of_dirs = os.listdir(VIDEOS_PATH)

config = ConfigParser.RawConfigParser()
config.read(CONFIG_FILE_PATH + '\config.txt')

for dir_name in list_of_dirs:
    if dir_name[:7] == 'results':
        input_video_file = VIDEOS_PATH + "\\" + dir_name + "\\" + 'frames.tif'
        config.set('offline', 'input_video_file', input_video_file)
        with open(CONFIG_FILE_PATH + '\config.txt', 'wb') as configfile:
            config.write(configfile)
        subprocess.call(r'python real_time_imaging\bin\main.py' + ' -c ' + CONFIG_FILE_PATH + '\config.txt' + ' -o ' + CONFIG_FILE_PATH + r'\results')

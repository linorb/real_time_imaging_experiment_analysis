import os

FILE_PATH = r'D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment'
list_of_dates = os.listdir(FILE_PATH)

MOUSE = '6'
CAGE = '40'
for date in list_of_dates
    mouse_folder = FILE_PATH + "\\" + date + "\\" + 'c%sm%s' %(CAGE, MOUSE)
    list_of_dirs = os.listdir(mouse_folder)
    for dir_name in list_of_dirs:
        try:
            file_name = mouse_folder + "\\" + dir_name + "\\" + 'frames.tif'
            os.remove(file_name)
        except OSError:
            print dir_name
            pass
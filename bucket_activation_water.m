CAGE = '40';
MOUSE = ['6', '3'];
BUCKET_TRACK_SESSIONS = {'20170305';
                         '20170306';
                         '20170307';
                         '20170308';
                         '20170309'};

figure;
color_ind = ['b', 'r'];
for i = 1:2
%% loading the wanted files
    mouse = MOUSE(i);
    if mouse == '6'
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\bucket_experiment\p_values\c40m6\p_2017_03_22__11_34_46\p_values');
    else 
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\bucket_experiment\p_values\c40m3\p_2017_03_22__11_35_10\p_values');
    end

    real_time_path = 'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging';

    %% count number of water frames for each session
    number_of_sessions = size(BUCKET_TRACK_SESSIONS, 1);
    session_count_of_water = zeros(1,number_of_sessions);
    for session = 1:number_of_sessions
        water_file_path = [real_time_path, '\', BUCKET_TRACK_SESSIONS{session}, '\c', CAGE, 'm', mouse, '\real_time_imaging\water_dispensed_frames.mat'];
        load(water_file_path);
        for trial = 1:length(all_session_water_frames)
            session_count_of_water(session) = session_count_of_water(session) + length(all_session_water_frames{trial});
        end
    end

    %%
    %% Create figures

    subplot(2,1,1);
        % Number of activations
        plot(all_session_activations, '-o', 'Color', color_ind(i)); hold on;
        ylim([200 1500]);
        xlim([0.9 5.1]);
        ylabel('# Bucekt activations', 'FontSize', 20);
        set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
    subplot(2,1,2);
        % Number of water frames
        plot(session_count_of_water, '-o', 'Color', color_ind(i)); hold on;
        ylim([0 70]);
        xlim([0.9 5.1]);
        ylabel('# Water rewards', 'FontSize', 20);
        xlabel('#Session', 'FontSize', 20);
        set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
end
legend('C40M6', 'C40M3')
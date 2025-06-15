%% Linear track activation figures 
CAGE = '40';
MOUSE = ['6', '3'];
LINEAR_TRACK_SESSIONS = {'20170219';
                         '20170222';
                         '20170225';
                         '20170228';
                         '20170303'};
                     
%% loading the wanted files
 figure;
 color_ind = ['b', 'r'];
for i=1:2
    mouse= MOUSE(i);
    if mouse =='6'
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment\p_correct\c40m6\p_2017_03_26__11_24_31\p_correct');
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment\p_values\c40m6\p_2017_03_26__11_13_02\p_values_edges');
    else
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment\p_values\c40m3\p_2017_03_26__11_33_52\p_values_edges.mat');
        load('D:\dev\real_time_imaging_experiment_analysis\reconstructing_traces_for_all_roi\linear_experiment\p_correct\c40m3\p_2017_03_26__11_26_05\p_correct.mat');
    end
    real_time_path = 'Z:\Short term data storage\Data storage (1 year)\experiments\real_time_imaging';

    %% count number of water frames for each session
    mouse = '3';
    real_time_path = 'Z:\experiments\projects\bambi\linear_track\analysis\nitzang_c40m3\1_linear_track';
    number_of_sessions = size(LINEAR_TRACK_SESSIONS, 1);
    session_count_of_water = zeros(1,number_of_sessions);
    for session = 1:number_of_sessions
        water_file_path = [real_time_path, '\', LINEAR_TRACK_SESSIONS{session}, '\water_dispensed_frames.mat'];
        load(water_file_path);
        for trial = 1:length(water_dispensed_frames)
            session_count_of_water(session) = session_count_of_water(session) + length(water_dispensed_frames{trial});
        end
    end

    %% Create figures
 
    subplot(3,1,1);
        % P correct subplot
        plot(p_correct_all_sessions, 'o-', 'Color', color_ind(i)); hold on;
        ylim([0.5 1.1]);
        xlim([0.9 5.1]);
        ylabel('P(correct)','FontSize', 20);
        set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
    % subplot(4,1,2);
    %     % p value
    %     plot(p_value_per_session, 'o-'); hold on;
    %     plot(0.95*ones(1,5),'r');
    %     ylim([0 1.1]);
    %     xlim([0.9 5.1]);
    %     title('P value for edges');
    %     ylabel('P');
    %     set(gca, 'Xtick', 1:number_of_sessions);
    subplot(3,1,2);
        % Number of activations
        plot(all_session_activations, '-o', 'Color', color_ind(i)); hold on;
    %     ylim([100 450]);
        xlim([0.9 5.1]);
        ylabel('# Activations', 'FontSize', 20);
        set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
    subplot(3,1,3);
        % Number of water frames
        plot(session_count_of_water, '-o', 'Color', color_ind(i)); hold on;
        ylim([0 50]);
        xlim([0.9 5.1]);
        ylabel('# Water rewards', 'FontSize', 20);
        xlabel('#Session', 'FontSize', 20);
        set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
end

legend('C40M6', 'C40M3')



figure();
plot(session_count_of_water, '-o'); hold on;
ylim([0 50]);
xlim([0.9 5.1]);
ylabel('# Water rewards', 'FontSize', 20);
xlabel('#Session', 'FontSize', 20);
set(gca, 'Xtick', 1:number_of_sessions, 'FontSize', 20);
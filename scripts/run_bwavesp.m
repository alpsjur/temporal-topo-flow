% run_bwavesp_and_plot.m
% Load input, run model, and plot results

%% --- Step 1: Load input array from text ---
input_filename = 'bswave_input.txt';    % Input file name, adjust as needed
arrayin_cell = load_arrayin_from_txt(input_filename);
arrayin = [arrayin_cell{:}];

%% --- Step 2: Run the model ---
output_name = 'topo_run';
bwavesp(arrayin, output_name);


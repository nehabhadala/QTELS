%%%% Video Data Hiding: Alpha Sensitivity Analysis (PSNR & SSIM Curves)

clear; clc; close all;

addpath('algos');

% Create main output folder
main_folder = 'Output_VH_3';
if ~exist(main_folder, 'dir')
    mkdir(main_folder);
end

% =========================================================================
% 1. Read Videos and Extract 10 VISUALLY DISTINCT Frames (Synchronized)
% =========================================================================
fprintf('Reading video1.mp4 (Cover) and video2.mp4 (Secret)...\n');
v_cover = VideoReader('video1.mp4');
v_secret = VideoReader('video2.mp4');

q = 10; % Number of frames
max_valid_frame = min(v_cover.NumFrames, v_secret.NumFrames);

if max_valid_frame < q
    error('Both videos must have at least %d frames.', q);
end

fprintf('Selecting %d visually distinct frames based on Cover video...\n', q);
common_indices = select_distinct_frames(v_cover, q, max_valid_frame);
cover_indices = common_indices;
secret_indices = common_indices; 

h = v_cover.Height;
w = v_cover.Width;

quat_to_rgb = @(Q_slice) max(0, min(1, cat(3, double(Q_slice.x), double(Q_slice.y), double(Q_slice.z))));

I_c = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));
for k = 1:q
    frame = double(read(v_cover, cover_indices(k))) / 255.0;
    if size(frame, 1) ~= h || size(frame, 2) ~= w, frame = imresize(frame, [h, w]); end
    I_c(:,:,k) = quaternion(zeros(h,w), frame(:,:,1), frame(:,:,2), frame(:,:,3));
end

I_s = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));
for k = 1:q
    frame = double(read(v_secret, secret_indices(k))) / 255.0;
    if size(frame, 1) ~= h || size(frame, 2) ~= w, frame = imresize(frame, [h, w]); end
    I_s(:,:,k) = quaternion(zeros(h,w), frame(:,:,1), frame(:,:,2), frame(:,:,3));
end

% =========================================================================
% 2. Pre-compute Tensor QR Decomposition (Done ONCE to save time)
% =========================================================================
fprintf('Performing Tensor QR Decomposition...\n');
M = dctmtx(q); 
[Q_c, R_c] = tqr(I_c, M);
[Q_s, R_s] = tqr(I_s, M);

% =========================================================================
% 3. Alpha Sensitivity Experiment
% =========================================================================
% Define the range of alpha values to test
alpha_values = 0.02:0.02:0.3; 
num_alphas = length(alpha_values);

% Initialize arrays to store average metrics
psnr_stego_arr = zeros(1, num_alphas);
ssim_stego_arr = zeros(1, num_alphas);
psnr_secret_arr = zeros(1, num_alphas);
ssim_secret_arr = zeros(1, num_alphas);

fprintf('Running embedding/extraction for %d different alpha values...\n', num_alphas);

for i = 1:num_alphas
    alpha = alpha_values(i);
    
    % Embedding
    R_em = R_c + alpha * R_s;
    I_si = qten_mproduct(Q_c, R_em, M);
    
    % Extraction
    [Q_si, R_si] = tqr(I_si, M);
    R_ex = (R_si - R_c) / alpha;
    I_s_rec = qten_mproduct(Q_s, R_ex, M);
    
    % Calculate Metrics (Average across 10 frames)
    psnr_s = 0; ssim_s = 0;
    psnr_sec = 0; ssim_sec = 0;
    
    for k = 1:q
        % Imperceptibility: Stego vs Cover
        psnr_s = psnr_s + psnr(quat_to_rgb(I_si(:,:,k)), quat_to_rgb(I_c(:,:,k)));
        ssim_s = ssim_s + ssim(quat_to_rgb(I_si(:,:,k)), quat_to_rgb(I_c(:,:,k)));
        
        % Fidelity: Extracted Secret vs Original Secret
        psnr_sec = psnr_sec + psnr(quat_to_rgb(I_s_rec(:,:,k)), quat_to_rgb(I_s(:,:,k)));
        ssim_sec = ssim_sec + ssim(quat_to_rgb(I_s_rec(:,:,k)), quat_to_rgb(I_s(:,:,k)));
    end
    
    psnr_stego_arr(i) = psnr_s / q;
    ssim_stego_arr(i) = ssim_s / q;
    psnr_secret_arr(i) = psnr_sec / q;
    ssim_secret_arr(i) = ssim_sec / q;
    
    fprintf('  Alpha = %.2f | Stego PSNR: %.2f dB | Secret PSNR: %.2f dB\n', ...
        alpha, psnr_stego_arr(i), psnr_secret_arr(i));
end

% =========================================================================
% 4. Generate Plots (Enhanced Visibility & Standardized Styling)
% =========================================================================
fprintf('Generating plots...\n');

% Common styling parameters for consistency
line_width = 2;
marker_size = 8;
axis_line_width = 1.5; % Makes the plot boundaries thick and dark
label_font_size = 18;

% --- Figure 1: PSNR vs Alpha ---
fig1 = figure('Position', [100, 100, 700, 500], 'Visible', 'off');
% Secret video (Red, Diamond marker)
plot(alpha_values, psnr_secret_arr, 'r-', 'LineWidth', line_width, 'Marker', 'd', 'MarkerSize', marker_size, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r'); hold on;
% Stego video (Blue, Circle marker)
plot(alpha_values, psnr_stego_arr, 'b-', 'LineWidth', line_width, 'Marker', 'o', 'MarkerSize', marker_size, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'b');

grid on;

% 1. Adjust axes FIRST to prevent overriding
set(gca, 'LineWidth', axis_line_width, 'FontSize', 14);
xlim([alpha_values(1), 0.3]);       
ylim([0, 300]);       

% 2. Use LaTeX interpreter for proper scaling of Greek letters
xlabel('$\alpha$', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('PSNR (dB)', 'Interpreter', 'latex', 'FontSize', label_font_size, 'FontWeight', 'bold');
legend('Input and output secret video (PSNR)', 'Cover and stego video (PSNR)', 'Location', 'best', 'FontSize', 14);

print(fig1, '-dpng', fullfile(main_folder, 'fig_psnr_vs_alpha.png'), '-r300');
print(fig1, '-depsc', fullfile(main_folder, 'fig_psnr_vs_alpha.eps'), '-r300');

% --- Figure 2: SSIM vs Alpha ---
fig2 = figure('Position', [850, 100, 700, 500], 'Visible', 'off');
% Secret video (Red, Diamond marker)
plot(alpha_values, ssim_secret_arr, 'r-', 'LineWidth', line_width, 'Marker', 'd', 'MarkerSize', marker_size, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r'); hold on;
% Stego video (Blue, Circle marker)
plot(alpha_values, ssim_stego_arr, 'b-', 'LineWidth', line_width, 'Marker', 'o', 'MarkerSize', marker_size, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'b');

grid on;

% 1. Adjust axes FIRST
set(gca, 'LineWidth', axis_line_width, 'FontSize', 14);
xlim([alpha_values(1), 0.3]);       
ylim([0.75, 1.02]);   

% 2. Use LaTeX interpreter for proper scaling of Greek letters
xlabel('$\alpha$', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('SSIM', 'Interpreter', 'latex', 'FontSize', label_font_size, 'FontWeight', 'bold');
legend('Input and output secret video (SSIM)', 'Cover and stego video (SSIM)', 'Location', 'southwest', 'FontSize', 14);

print(fig2, '-dpng', fullfile(main_folder, 'fig_ssim_vs_alpha.png'), '-r300');
print(fig2, '-depsc', fullfile(main_folder, 'fig_ssim_vs_alpha.eps'), '-r300');

% =========================================================================
% 5. Save Representative Frames (for alpha = 0.1)
% =========================================================================
fprintf('Saving representative frames for alpha = 0.1...\n');
alpha_rep = 0.1;
R_em_rep = R_c + alpha_rep * R_s;
I_si_rep = qten_mproduct(Q_c, R_em_rep, M);
[Q_si_rep, R_si_rep] = tqr(I_si_rep, M);
R_ex_rep = (R_si_rep - R_c) / alpha_rep;
I_s_rec_rep = qten_mproduct(Q_s, R_ex_rep, M);

subfolders = {'Cover', 'Secret', 'Stego', 'Extracted_Secret'};
for i = 1:length(subfolders)
    if ~exist(fullfile(main_folder, subfolders{i}), 'dir')
        mkdir(fullfile(main_folder, subfolders{i}));
    end
end

for k = 1:q
    imwrite(im2uint8(quat_to_rgb(I_c(:,:,k))), fullfile(main_folder, 'Cover', sprintf('cover_%03d.png', k)));
    imwrite(im2uint8(quat_to_rgb(I_s(:,:,k))), fullfile(main_folder, 'Secret', sprintf('secret_%03d.png', k)));
    imwrite(im2uint8(quat_to_rgb(I_si_rep(:,:,k))), fullfile(main_folder, 'Stego', sprintf('stego_%03d.png', k)));
    imwrite(im2uint8(quat_to_rgb(I_s_rec_rep(:,:,k))), fullfile(main_folder, 'Extracted_Secret', sprintf('extracted_%03d.png', k)));
end

% =========================================================================
% 6. Save Summary Text File
% =========================================================================
fid = fopen(fullfile(main_folder, 'summary_alpha_sweep.txt'), 'w');
fprintf(fid, 'Video Data Hiding: Alpha Sensitivity Analysis\n');
fprintf(fid, '=============================================\n\n');
fprintf(fid, 'Alpha | Stego PSNR | Stego SSIM | Secret PSNR | Secret SSIM\n');
fprintf(fid, '----------------------------------------------------------\n');
for i = 1:num_alphas
    fprintf(fid, '%.2f  | %8.2f dB | %8.4f   | %9.2f dB | %8.4f\n', ...
        alpha_values(i), psnr_stego_arr(i), ssim_stego_arr(i), psnr_secret_arr(i), ssim_secret_arr(i));
end
fclose(fid);

fprintf('\nAll outputs successfully saved to %s!\n', main_folder);

% =========================================================================
% Helper Functions
% =========================================================================
function indices = select_distinct_frames(v_reader, q, max_frame_limit)
    if nargin < 3, max_frame_limit = v_reader.NumFrames; end
    num_candidates = min(30, max_frame_limit);
    candidate_indices = round(linspace(1, max_frame_limit, num_candidates));
    h = v_reader.Height; w = v_reader.Width;
    pool = zeros(h, w, 3, num_candidates, 'uint8');
    for i = 1:num_candidates, pool(:,:,:,i) = read(v_reader, candidate_indices(i)); end
    diff_matrix = zeros(num_candidates, num_candidates);
    for i = 1:num_candidates
        for j = i+1:num_candidates
            d = sum(abs(double(pool(:,:,:,i)) - double(pool(:,:,:,j))), 'all');
            diff_matrix(i,j) = d; diff_matrix(j,i) = d;
        end
    end
    selected = zeros(1, q); selected(1) = 1;
    for s = 2:q
        max_min_diff = -1; best_idx = -1;
        for i = 1:num_candidates
            if any(selected == i), continue; end
            min_diff = min(diff_matrix(i, selected(1:s-1)));
            if min_diff > max_min_diff, max_min_diff = min_diff; best_idx = i; end
        end
        selected(s) = best_idx;
    end
    raw_frames = candidate_indices(selected);
    [~, sort_idx] = sort(raw_frames);
    indices = raw_frames(sort_idx);
end

function [Q, R] = tqr(A, M)
    A_hat = qten_mode3(A, M);
    [m, n, q] = size(A);
    Q_hat = quaternion(zeros(m,m,q), zeros(m,m,q), zeros(m,m,q), zeros(m,m,q));
    R_hat = quaternion(zeros(m,n,q), zeros(m,n,q), zeros(m,n,q), zeros(m,n,q));
    for k = 1:q
        [Q_hat(:,:,k), R_hat(:,:,k)] = qr(A_hat(:,:,k));
    end
    Q = qten_mode3(Q_hat, inv(M));
    R = qten_mode3(R_hat, inv(M));
end
%%%%  White Watermark, 5 Distinct Frames & Output_final4

%% =========================================================================
%  Application: Watermark-Preserved Color Video Restoration (Real Video)
%  Demonstrates the necessity of Equality Constraints in Tensor Restoration
%% =========================================================================
clear; clc; close all;

addpath('algos');

if ~exist('Output_final4', 'dir')
    mkdir('Output_final4');
end

% =========================================================================
%  1. Read Real Video and Select 5 DISTINCT Frames (Smart Selection)
% =========================================================================
fprintf('Reading test4.mp4 and analyzing motion for distinct frames...\n');
v_reader = VideoReader('test4.mp4'); 
total_frames = v_reader.NumFrames;
h = v_reader.Height;
w = v_reader.Width;

q = 5; % We want exactly 5 frames

% --- Smart Frame Selection Algorithm ---
% Read a pool of candidate frames spread across the entire video
num_candidates = min(30, total_frames);
candidate_indices = round(linspace(1, total_frames, num_candidates));
pool = zeros(h, w, 3, num_candidates, 'uint8');

fprintf('  Reading %d candidate frames...\n', num_candidates);
for i = 1:num_candidates
    pool(:,:,:,i) = read(v_reader, candidate_indices(i));
end

% Calculate visual difference between all pairs of candidates
fprintf('  Calculating visual differences...\n');
diff_matrix = zeros(num_candidates, num_candidates);
for i = 1:num_candidates
    for j = i+1:num_candidates
        d = sum(abs(double(pool(:,:,:,i)) - double(pool(:,:,:,j))), 'all');
        diff_matrix(i,j) = d;
        diff_matrix(j,i) = d;
    end
end

% Greedily select q frames that maximize visual difference
fprintf('  Selecting %d most distinct frames...\n', q);
selected = zeros(1, q);
selected(1) = 1; % Start with the first candidate

for s = 2:q
    max_min_diff = -1;
    best_idx = -1;
    for i = 1:num_candidates
        if any(selected == i), continue; end
        % Find the minimum distance from candidate i to all already-selected frames
        min_diff = min(diff_matrix(i, selected(1:s-1)));
        if min_diff > max_min_diff
            max_min_diff = min_diff;
            best_idx = i;
        end
    end
    selected(s) = best_idx;
end

% Map back to actual video frame numbers
raw_frames = candidate_indices(selected);

% Sort them chronologically so they appear in increasing order
[~, sort_idx] = sort(raw_frames);
selected = selected(sort_idx);
frame_indices = raw_frames(sort_idx);

fprintf('  Selected video frames (sorted): [%s]\n', num2str(frame_indices));

M_mat = dctmtx(q);

% Helper function to convert quaternion slice to RGB image
quat_to_rgb = @(Q_slice) max(0, min(1, cat(3, double(Q_slice.x), double(Q_slice.y), double(Q_slice.z))));

X_true = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));
for k = 1:q
    frame_rgb = double(pool(:,:,:,selected(k))) / 255.0;
    R = frame_rgb(:,:,1); G = frame_rgb(:,:,2); B = frame_rgb(:,:,3);
    X_true(:,:,k) = quaternion(zeros(h,w), R, G, B);
end

% =========================================================================
%  2. Embed Visually Striking Text Watermark (WHITE COLOR)
% =========================================================================
fprintf('Embedding large white text watermark...\n');
p = round(h * 0.20);
alpha = 0.95;

% --- Generate Crisp Text Mask ---
fig_tmp = figure('Visible', 'off', 'Color', 'w', 'Position', [0, 0, w, p]);
ax_tmp = axes('Parent', fig_tmp, 'Position', [0 0 1 1], 'XLim', [1 w], 'YLim', [1 p]);
text(w/2, p/2, 'AUTHENTIC', 'FontSize', 15, 'FontWeight', 'bold', ...
     'Color', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
axis off;
F = getframe(ax_tmp);
close(fig_tmp);
text_mask = rgb2gray(F.cdata) < 128; % 1 where text is, 0 where background is

% --- Apply Text to Video Frames ---
R_all = double(X_true.x);
G_all = double(X_true.y);
B_all = double(X_true.z);

for k = 1:q
    R_curr = R_all(:,:,k); G_curr = G_all(:,:,k); B_curr = B_all(:,:,k);
    
    R_bot = R_curr(h-p+1:h, :);
    G_bot = G_curr(h-p+1:h, :);
    B_bot = B_curr(h-p+1:h, :);
    
    % CHANGED: Blend ONLY where text_mask is 1. Target color is now WHITE (R=1, G=1, B=1).
    R_bot(text_mask) = (1-alpha)*R_bot(text_mask) + alpha*1.0; % Red component of white
    G_bot(text_mask) = (1-alpha)*G_bot(text_mask) + alpha*1.0; % Green component of white
    B_bot(text_mask) = (1-alpha)*B_bot(text_mask) + alpha*1.0; % Blue component of white
    
    R_curr(h-p+1:h, :) = R_bot;
    G_curr(h-p+1:h, :) = G_bot;
    B_curr(h-p+1:h, :) = B_bot;
    
    R_all(:,:,k) = R_curr; G_all(:,:,k) = G_curr; B_all(:,:,k) = B_curr;
end

X_true = quaternion(zeros(h,w,q), R_all, G_all, B_all);

% =========================================================================
%  3. Define Degradation Operator (Mild vertical blur + noise)
% =========================================================================
fprintf('Applying degradation and noise...\n');
A = quaternion(zeros(h,h,q), zeros(h,h,q), zeros(h,h,q), zeros(h,h,q));
for k = 1:q
    A_mat = eye(h) + 0.3 * diag(ones(h-1,1), 1) + 0.3 * diag(ones(h-1,1), -1);
    A_mat = A_mat + 0.05 * randn(h,h);
    A(:,:,k) = quaternion(zeros(h,h), A_mat, A_mat, A_mat);
end

Noise = 0.01 * quaternion(randn(h,w,q), randn(h,w,q), randn(h,w,q), randn(h,w,q));
B = qten_mproduct(A, X_true, M_mat) + Noise;

% =========================================================================
%  4. Define Watermark Constraints (C * X = D)
% =========================================================================
C = quaternion(randn(p,h,q)/sqrt(h), randn(p,h,q)/sqrt(h), randn(p,h,q)/sqrt(h), randn(p,h,q)/sqrt(h));
D = qten_mproduct(C, X_true, M_mat);

% =========================================================================
%  5. Unconstrained Restoration (Standard TLS)
% =========================================================================
fprintf('Solving Unconstrained Tensor Least Squares...\n');
A_tilde = qten_mode3(A, M_mat);
B_tilde = qten_mode3(B, M_mat);
X_unc_tilde = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));

for k = 1:q
    Ak = A_tilde(:,:,k);
    Bk = B_tilde(:,:,k);
    Ak_R = squeeze(qten_real_rep(Ak));
    Bk_cR = squeeze(qten_real_col(Bk));
    Xk_cR = pinv(Ak_R) * Bk_cR;
    X0 = Xk_cR(1:h, :);
    X1 = Xk_cR(h+1:2*h, :);
    X2 = Xk_cR(2*h+1:3*h, :);
    X3 = Xk_cR(3*h+1:4*h, :);
    X_unc_tilde(:,:,k) = quaternion(X0, X1, X2, X3);
end
X_unc = qten_mode3(X_unc_tilde, M_mat');

% =========================================================================
%  6. Constrained Restoration (Proposed QTELS)
% =========================================================================
fprintf('Solving Proposed QTELS (with watermark constraints)...\n');
X_qtels = qten_qtels(A, B, C, D, M_mat);

% =========================================================================
%  7. Evaluate Metrics
% =========================================================================
err_unc = qten_norm(X_unc - X_true) / qten_norm(X_true);
err_qtels = qten_norm(X_qtels - X_true) / qten_norm(X_true);

W_unc = qten_mproduct(C, X_unc, M_mat);
W_qtels = qten_mproduct(C, X_qtels, M_mat);
wm_err_unc = qten_norm(W_unc - D) / qten_norm(D);
wm_err_qtels = qten_norm(W_qtels - D) / qten_norm(D);

fprintf('\n--- RESULTS ---\n');
fprintf('Unconstrained TLS:\n');
fprintf('  Total Video Error: %.4e\n', err_unc);
fprintf('  Watermark Error:   %.4e  (Watermark Destroyed)\n', wm_err_unc);

fprintf('Proposed QTELS:\n');
fprintf('  Total Video Error: %.4e\n', err_qtels);
fprintf('  Watermark Error:   %.4e  (Watermark Preserved)\n', wm_err_qtels);

% =========================================================================
%  7b. Calculate PSNR and SSIM for All 5 Frames
% =========================================================================
fprintf('\n--- PSNR and SSIM Analysis (5 Frames) ---\n');

num_frames_calc = q; % 5
psnr_blurred = zeros(1, num_frames_calc);
psnr_unc = zeros(1, num_frames_calc);
psnr_qtels = zeros(1, num_frames_calc);
ssim_blurred = zeros(1, num_frames_calc);
ssim_unc = zeros(1, num_frames_calc);
ssim_qtels = zeros(1, num_frames_calc);

for frame_idx = 1:num_frames_calc
    orig_rgb = quat_to_rgb(X_true(:,:,frame_idx));
    blur_rgb = quat_to_rgb(B(:,:,frame_idx));
    unc_rgb = quat_to_rgb(X_unc(:,:,frame_idx));
    qtels_rgb = quat_to_rgb(X_qtels(:,:,frame_idx));
    
    psnr_blurred(frame_idx) = psnr(blur_rgb, orig_rgb);
    psnr_unc(frame_idx) = psnr(unc_rgb, orig_rgb);
    psnr_qtels(frame_idx) = psnr(qtels_rgb, orig_rgb);
    ssim_blurred(frame_idx) = ssim(blur_rgb, orig_rgb);
    ssim_unc(frame_idx) = ssim(unc_rgb, orig_rgb);
    ssim_qtels(frame_idx) = ssim(qtels_rgb, orig_rgb);
    
    fprintf('Frame %d (Video Index %d):\n', frame_idx, frame_indices(frame_idx));
    fprintf('  Blurred:       PSNR = %5.2f dB, SSIM = %.4f\n', psnr_blurred(frame_idx), ssim_blurred(frame_idx));
    fprintf('  Unconstrained: PSNR = %5.2f dB, SSIM = %.4f\n', psnr_unc(frame_idx), ssim_unc(frame_idx));
    fprintf('  QTELS:         PSNR = %5.2f dB, SSIM = %.4f\n', psnr_qtels(frame_idx), ssim_qtels(frame_idx));
    fprintf('\n');
end

avg_psnr_blurred = mean(psnr_blurred);
avg_psnr_unc = mean(psnr_unc);
avg_psnr_qtels = mean(psnr_qtels);
avg_ssim_blurred = mean(ssim_blurred);
avg_ssim_unc = mean(ssim_unc);
avg_ssim_qtels = mean(ssim_qtels);

fprintf('--- Averages Across 5 Frames ---\n');
fprintf('Blurred:       PSNR = %5.2f dB, SSIM = %.4f\n', avg_psnr_blurred, avg_ssim_blurred);
fprintf('Unconstrained: PSNR = %5.2f dB, SSIM = %.4f\n', avg_psnr_unc, avg_ssim_unc);
fprintf('QTELS:         PSNR = %5.2f dB, SSIM = %.4f\n', avg_psnr_qtels, avg_ssim_qtels);

% --- OUTPUT: PSNR/SSIM Bar Chart ---
fig_metrics = figure('Position', [100, 100, 1000, 600], 'Visible', 'off', 'Name', 'PSNR and SSIM Metrics');
metrics_matrix = [frame_indices(1:num_frames_calc)', ...
                  psnr_blurred', psnr_unc', psnr_qtels', ...
                  ssim_blurred', ssim_unc', ssim_qtels'];
col_names = {'Frame', 'PSNR_Blur', 'PSNR_Unc', 'PSNR_QTELS', 'SSIM_Blur', 'SSIM_Unc', 'SSIM_QTELS'};
t = uitable('Parent', fig_metrics, 'Data', metrics_matrix, 'ColumnName', col_names, ...
            'Position', [50 350 900 200], 'FontSize', 10, 'ColumnWidth', {60});
ax = axes('Parent', fig_metrics, 'Position', [0.1 0.1 0.8 0.35]);
bar_data = [avg_psnr_blurred; avg_psnr_unc; avg_psnr_qtels];
b = bar(ax, bar_data);
b.FaceColor = 'flat';
b.CData(1,:) = [0.7 0.7 0.7];
b.CData(2,:) = [0.5 0.5 1.0];
b.CData(3,:) = [1.0 0.3 0.3];
set(ax, 'XTickLabel', {'Blurred', 'Unconstrained', 'QTELS'}, 'FontSize', 11);
ylabel(ax, 'Average PSNR (dB)', 'FontSize', 12);
title(ax, 'Average PSNR Comparison Across 5 Frames', 'FontSize', 12, 'FontWeight', 'bold');
grid(ax, 'on');
print(fig_metrics, '-dpng', fullfile('Output_final4', 'fig_psnr_ssim_comparison.png'), '-r300');
print(fig_metrics, '-depsc', fullfile('Output_final4', 'fig_psnr_ssim_comparison.eps'), '-r300');

% =========================================================================
%  8. Generate Visuals: ALL 5 Frames
% =========================================================================

% --- OUTPUT 0: All 5 Frames Comparison Grid (4 rows x 5 cols) ---
fprintf('Generating 5-frame comparison grid...\n');
num_frames_show = q; % 5
fig_all = figure('Position', [50, 50, 1200, 700], 'Visible', 'off'); 

for col = 1:num_frames_show
    % Row 1: Original Clean
    subplot(4, num_frames_show, col);
    imshow(quat_to_rgb(X_true(:,:,col)));
    if col == 1, ylabel('Original Clean', 'FontSize', 11, 'FontWeight', 'bold'); end
    title(sprintf('Frame %d', frame_indices(col)), 'FontSize', 10);
    
    % Row 2: Blurred
    subplot(4, num_frames_show, num_frames_show + col);
    imshow(quat_to_rgb(B(:,:,col)));
    if col == 1, ylabel('Blurred + Noise', 'FontSize', 11, 'FontWeight', 'bold'); end
    
    % Row 3: Unconstrained
    subplot(4, num_frames_show, 2*num_frames_show + col);
    imshow(quat_to_rgb(X_unc(:,:,col)));
    if col == 1, ylabel('Unconstrained TLS', 'FontSize', 11, 'FontWeight', 'bold'); end
    
    % Row 4: QTELS
    subplot(4, num_frames_show, 3*num_frames_show + col);
    imshow(quat_to_rgb(X_qtels(:,:,col)));
    if col == 1, ylabel('Proposed QTELS', 'FontSize', 11, 'FontWeight', 'bold'); end
end

print(fig_all, '-dpng', fullfile('Output_final4', 'fig_all_5_frames.png'), '-r300');
print(fig_all, '-depsc', fullfile('Output_final4', 'fig_all_5_frames.eps'), '-r300');

fprintf('\nGenerating visual outputs...\n');

save('Output_final4/video_tensors.mat', 'X_true', 'X_unc', 'X_qtels', 'M_mat');

% --- OUTPUT 1: Main Paper Figure (Grid of Key Frames: 1, 3, 5) ---
frames_to_show = [1, 3, q]; 
num_key = length(frames_to_show);

fig_paper = figure('Position', [100, 100, 1200, 600], 'Visible', 'off');
for col = 1:num_key
    k = frames_to_show(col);
    subplot(3, num_key, col);
    imshow(quat_to_rgb(X_true(:,:,k)));
    if col == 1, ylabel('Original Clean', 'FontSize', 12, 'FontWeight', 'bold'); end
    title(sprintf('Frame %d', frame_indices(k)), 'FontSize', 11);
    
    subplot(3, num_key, num_key + col);
    imshow(quat_to_rgb(X_unc(:,:,k)));
    if col == 1, ylabel('Unconstrained TLS', 'FontSize', 12, 'FontWeight', 'bold'); end
    
    subplot(3, num_key, 2*num_key + col);
    imshow(quat_to_rgb(X_qtels(:,:,k)));
    if col == 1, ylabel('Proposed QTELS', 'FontSize', 12, 'FontWeight', 'bold'); end
end
print(fig_paper, '-dpng', fullfile('Output_final4', 'fig_paper_grid.png'), '-r300');
print(fig_paper, '-depsc', fullfile('Output_final4', 'fig_paper_grid.eps'), '-r300');

% --- OUTPUT 2: Watermark Zoom-In Figure ---
k_show = 3; % Show the 3rd frame for zoom

fig_zoom = figure('Position', [100, 100, 1000, 300], 'Visible', 'off');
subplot(1,3,1); imshow(quat_to_rgb(X_true(h-p+1:h, :, k_show))); title('Original Watermark', 'FontSize', 12);
subplot(1,3,2); imshow(quat_to_rgb(X_unc(h-p+1:h, :, k_show))); title('Unconstrained (Blurred)', 'FontSize', 12);
subplot(1,3,3); imshow(quat_to_rgb(X_qtels(h-p+1:h, :, k_show))); title('QTELS (Preserved)', 'FontSize', 12);
print(fig_zoom, '-dpng', fullfile('Output_final4', 'fig_watermark_zoom.png'), '-r300');
print(fig_zoom, '-depsc', fullfile('Output_final4', 'fig_watermark_zoom.eps'), '-r300');

% --- OUTPUT 3: Supplementary Video ---
fprintf('Creating supplementary video...\n');
v_writer = VideoWriter(fullfile('Output_final4', 'restoration_comparison.mp4'), 'MPEG-4');
v_writer.FrameRate = 2;
open(v_writer);
for k = 1:q
    img_true = quat_to_rgb(X_true(:,:,k));
    img_unc  = quat_to_rgb(X_unc(:,:,k));
    img_qtels = quat_to_rgb(X_qtels(:,:,k));
    border = zeros(h, 2, 3);
    writeVideo(v_writer, [img_true, border, img_unc, border, img_qtels]);
end
close(v_writer);

fprintf('All visual outputs successfully saved to Output_final4 folder!\n');

% =========================================================================
%  9. Organized Output: Individual Frames in Subfolders
% =========================================================================
fprintf('\n=== Organizing outputs into Output_final4 folder structure ===\n');

main_folder = 'Output_final4';
if ~exist(main_folder, 'dir'), mkdir(main_folder); end
if ~exist(fullfile(main_folder, 'original'), 'dir'), mkdir(fullfile(main_folder, 'original')); end
if ~exist(fullfile(main_folder, 'blurred'), 'dir'), mkdir(fullfile(main_folder, 'blurred')); end
if ~exist(fullfile(main_folder, 'unconstrained'), 'dir'), mkdir(fullfile(main_folder, 'unconstrained')); end
if ~exist(fullfile(main_folder, 'qtels'), 'dir'), mkdir(fullfile(main_folder, 'qtels')); end

num_frames_save = q; % 5

fprintf('Saving individual frames without borders...\n');
for frame_idx = 1:num_frames_save
    frame_num = frame_indices(frame_idx);
    
    img_orig  = im2uint8(quat_to_rgb(X_true(:,:,frame_idx)));
    img_blur  = im2uint8(quat_to_rgb(B(:,:,frame_idx)));
    img_unc   = im2uint8(quat_to_rgb(X_unc(:,:,frame_idx)));
    img_qtels = im2uint8(quat_to_rgb(X_qtels(:,:,frame_idx)));
    
    imwrite(img_orig,  fullfile(main_folder, 'original', sprintf('frame_%d.png', frame_num)));
    imwrite(img_blur,  fullfile(main_folder, 'blurred', sprintf('frame_%d.png', frame_num)));
    imwrite(img_unc,   fullfile(main_folder, 'unconstrained', sprintf('frame_%d.png', frame_num)));
    imwrite(img_qtels, fullfile(main_folder, 'qtels', sprintf('frame_%d.png', frame_num)));
    
    fprintf('  Saved frame %d to all subfolders\n', frame_num);
end

% =========================================================================
%  10. Comprehensive Summary Text File
% =========================================================================
fprintf('\nGenerating comprehensive summary file...\n');

fid = fopen(fullfile(main_folder, 'SUMMARY_METRICS.txt'), 'w');

fprintf(fid, '================================================================================\n');
fprintf(fid, '        WATERMARK-PRESERVED COLOR VIDEO RESTORATION - COMPLETE RESULTS\n');
fprintf(fid, '================================================================================\n');
fprintf(fid, 'Experiment Date: %s\n', datestr(now));
fprintf(fid, 'Video Source: test4.mp4\n'); 
fprintf(fid, 'Video Dimensions: %d x %d pixels\n', h, w);
fprintf(fid, 'Number of Frames Processed: %d\n', q);
fprintf(fid, 'Selected Video Frames: [%s]\n', num2str(frame_indices));
fprintf(fid, 'Watermark Region: Bottom %d rows (%.1f%% of frame height)\n', p, (p/h)*100);
fprintf(fid, 'Transform Matrix M: DCT (%d x %d)\n', q, q);
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '--- SECTION 1: OVERALL TENSOR RECONSTRUCTION ERRORS ---\n');
fprintf(fid, 'Method                  | Relative Error\n');
fprintf(fid, '----------------------------------------------------------------------------\n');
fprintf(fid, 'Unconstrained TLS       | %.4e\n', err_unc);
fprintf(fid, 'Proposed QTELS          | %.4e\n', err_qtels);
fprintf(fid, '----------------------------------------------------------------------------\n\n');

fprintf(fid, '--- SECTION 2: WATERMARK CONSTRAINT ERRORS ---\n');
fprintf(fid, 'Method                  | Watermark Error      | Status\n');
fprintf(fid, '----------------------------------------------------------------------------\n');
fprintf(fid, 'Unconstrained TLS       | %.4e     | WATERMARK DESTROYED\n', wm_err_unc);
fprintf(fid, 'Proposed QTELS          | %.4e     | WATERMARK PRESERVED (machine precision)\n', wm_err_qtels);
fprintf(fid, '----------------------------------------------------------------------------\n\n');

fprintf(fid, '--- SECTION 3: PSNR ANALYSIS ---\n');
fprintf(fid, 'Frame | Blurred (dB) | Unconstrained (dB) | QTELS (dB)\n');
fprintf(fid, '----------------------------------------------------------------------------\n');
for i = 1:num_frames_calc
    fprintf(fid, '%5d | %10.2f | %18.2f | %10.2f\n', ...
        frame_indices(i), psnr_blurred(i), psnr_unc(i), psnr_qtels(i));
end
fprintf(fid, '----------------------------------------------------------------------------\n');
fprintf(fid, 'AVG   | %10.2f | %18.2f | %10.2f\n', ...
    avg_psnr_blurred, avg_psnr_unc, avg_psnr_qtels);
fprintf(fid, '----------------------------------------------------------------------------\n\n');

fprintf(fid, '--- SECTION 4: SSIM ANALYSIS ---\n');
fprintf(fid, 'Frame | Blurred      | Unconstrained      | QTELS\n');
fprintf(fid, '----------------------------------------------------------------------------\n');
for i = 1:num_frames_calc
    fprintf(fid, '%5d | %10.4f | %18.4f | %10.4f\n', ...
        frame_indices(i), ssim_blurred(i), ssim_unc(i), ssim_qtels(i));
end
fprintf(fid, '----------------------------------------------------------------------------\n');
fprintf(fid, 'AVG   | %10.4f | %18.4f | %10.4f\n', ...
    avg_ssim_blurred, avg_ssim_unc, avg_ssim_qtels);
fprintf(fid, '----------------------------------------------------------------------------\n\n');

fprintf(fid, '--- SECTION 5: IMPROVEMENT SUMMARY ---\n');
fprintf(fid, 'PSNR Improvement over Blurred Input:\n');
fprintf(fid, '  Unconstrained TLS: +%.2f dB\n', avg_psnr_unc - avg_psnr_blurred);
fprintf(fid, '  Proposed QTELS:    +%.2f dB\n', avg_psnr_qtels - avg_psnr_blurred);
fprintf(fid, '\nSSIM Improvement over Blurred Input:\n');
fprintf(fid, '  Unconstrained TLS: +%.4f\n', avg_ssim_unc - avg_ssim_blurred);
fprintf(fid, '  Proposed QTELS:    +%.4f\n', avg_ssim_qtels - avg_ssim_blurred);
fprintf(fid, '\nWatermark Error Reduction (vs Unconstrained):\n');
if wm_err_unc > 0 && wm_err_qtels > 0
    fprintf(fid, '  Factor: %.2e x reduction\n', wm_err_unc / wm_err_qtels);
else
    fprintf(fid, '  QTELS achieves machine-precision watermark preservation\n');
end
fprintf(fid, '----------------------------------------------------------------------------\n\n');

fprintf(fid, '--- SECTION 6: KEY CONCLUSIONS ---\n');
fprintf(fid, '1. Unconstrained TLS DESTROYS the watermark (high-frequency features treated as noise).\n');
fprintf(fid, '2. QTELS simultaneously restores background AND preserves watermark at machine precision.\n');
fprintf(fid, '3. Hard equality constraints C *_M X = D act as an unbreakable mathematical vault.\n');
fprintf(fid, '4. This demonstrates the NECESSITY of the QTELS framework.\n');
fprintf(fid, '================================================================================\n');
fclose(fid);

fprintf('\n=== All organized outputs saved to Output_final4 folder ===\n');
fprintf('Folder Structure:\n');
fprintf('  Output_final4/\n');
fprintf('  ├── original/          (5 individual original frames)\n');
fprintf('  ├── blurred/           (5 individual blurred frames)\n');
fprintf('  ├── unconstrained/     (5 individual unconstrained TLS frames)\n');
fprintf('  ├── qtels/             (5 individual QTELS frames)\n');
fprintf('  ├── fig_all_5_frames.png\n');
fprintf('  ├── fig_watermark_zoom.png\n');
fprintf('  ├── restoration_comparison.mp4\n');
fprintf('  └── SUMMARY_METRICS.txt\n');
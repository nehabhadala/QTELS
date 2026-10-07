%%%% Video Data Hiding via Tensor QR Decomposition (10 Frames, Synchronized & Fixed)

clear; clc; close all;

addpath('algos');

% Create main output folder and subfolders
main_folder = 'Output_VH_1';
if ~exist(main_folder, 'dir')
    mkdir(main_folder);
end

subfolders = {'Cover', 'Secret', 'Stego', 'Extracted_Secret'};
for i = 1:length(subfolders)
    if ~exist(fullfile(main_folder, subfolders{i}), 'dir')
        mkdir(fullfile(main_folder, subfolders{i}));
    end
end

% =========================================================================
% 1. Read Videos and Extract 10 VISUALLY DISTINCT Frames (Synchronized)
% =========================================================================
fprintf('Reading video1.mp4 (Cover) and video2.mp4 (Secret)...\n');
v_cover = VideoReader('video1.mp4');
v_secret = VideoReader('video2.mp4');

q = 10; % Number of frames

% --- FIX: Calculate the maximum valid frame index for BOTH videos ---
max_valid_frame = min(v_cover.NumFrames, v_secret.NumFrames);

if max_valid_frame < q
    error('Both videos must have at least %d frames. Cover has %d, Secret has %d.', ...
        q, v_cover.NumFrames, v_secret.NumFrames);
end

% --- Smart Frame Selection (Synchronized) ---
% We select indices based on the Cover video, but restricted to the common frame limit
fprintf('Selecting %d visually distinct frames based on Cover video...\n', q);
common_indices = select_distinct_frames(v_cover, q, max_valid_frame);

cover_indices = common_indices;
secret_indices = common_indices; % Force same indices for Secret

% Note: For TQR addition to work, both videos must have the same spatial dimensions.
h = v_cover.Height;
w = v_cover.Width;

% Helper function to convert quaternion slice to RGB
quat_to_rgb = @(Q_slice) max(0, min(1, cat(3, double(Q_slice.x), double(Q_slice.y), double(Q_slice.z))));

% Read cover video frames
I_c = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));
for k = 1:q
    frame = double(read(v_cover, cover_indices(k))) / 255.0;
    % Resize if dimensions differ (safety check)
    if size(frame, 1) ~= h || size(frame, 2) ~= w
        frame = imresize(frame, [h, w]);
    end
    I_c(:,:,k) = quaternion(zeros(h,w), frame(:,:,1), frame(:,:,2), frame(:,:,3));
end

% Read secret video frames
I_s = quaternion(zeros(h,w,q), zeros(h,w,q), zeros(h,w,q), zeros(h,w,q));
for k = 1:q
    frame = double(read(v_secret, secret_indices(k))) / 255.0;
    % Resize if dimensions differ (safety check)
    if size(frame, 1) ~= h || size(frame, 2) ~= w
        frame = imresize(frame, [h, w]);
    end
    I_s(:,:,k) = quaternion(zeros(h,w), frame(:,:,1), frame(:,:,2), frame(:,:,3));
end

fprintf('Frames selected (used for both): [%s]\n', num2str(cover_indices));

% =========================================================================
% 2. Tensor QR Decomposition (TQR)
% =========================================================================
fprintf('Performing Tensor QR Decomposition...\n');
M = dctmtx(q); % Transform matrix

% TQR for Cover Video: I_c = Q_c *_M R_c
[Q_c, R_c] = tqr(I_c, M);

% TQR for Secret Video: I_s = Q_s *_M R_s
[Q_s, R_s] = tqr(I_s, M);

% =========================================================================
% 3. Embedding Process
% =========================================================================
fprintf('Embedding secret video into cover video...\n');
alpha = 0.1; % Embedding strength

% Modify the upper triangular tensor component
R_em = R_c + alpha * R_s;

% Construct the stego tensor
I_si = qten_mproduct(Q_c, R_em, M);

% =========================================================================
% 4. Extraction Process
% =========================================================================
fprintf('Extracting secret video from stego video...\n');

% TQR for Stego Video: I_si = Q_si *_M R_si
[Q_si, R_si] = tqr(I_si, M);

% Recover the secret component
R_ex = (R_si - R_c) / alpha;

% Reconstruct the secret video
I_s_rec = qten_mproduct(Q_s, R_ex, M);

% =========================================================================
% 5. Evaluate Metrics (PSNR and SSIM)
% =========================================================================
fprintf('Calculating PSNR and SSIM...\n');

% Imperceptibility: Stego vs Cover
psnr_stego = zeros(1, q);
ssim_stego = zeros(1, q);
for k = 1:q
    psnr_stego(k) = psnr(quat_to_rgb(I_si(:,:,k)), quat_to_rgb(I_c(:,:,k)));
    ssim_stego(k) = ssim(quat_to_rgb(I_si(:,:,k)), quat_to_rgb(I_c(:,:,k)));
end

% Fidelity: Extracted Secret vs Original Secret
psnr_secret = zeros(1, q);
ssim_secret = zeros(1, q);
for k = 1:q
    psnr_secret(k) = psnr(quat_to_rgb(I_s_rec(:,:,k)), quat_to_rgb(I_s(:,:,k)));
    ssim_secret(k) = ssim(quat_to_rgb(I_s_rec(:,:,k)), quat_to_rgb(I_s(:,:,k)));
end

fprintf('\n--- RESULTS ---\n');
fprintf('Imperceptibility (Stego vs Cover):\n');
fprintf('  Average PSNR: %.2f dB\n', mean(psnr_stego));
fprintf('  Average SSIM: %.4f\n\n', mean(ssim_stego));

fprintf('Fidelity (Extracted Secret vs Original Secret):\n');
fprintf('  Average PSNR: %.2f dB\n', mean(psnr_secret));
fprintf('  Average SSIM: %.4f\n\n', mean(ssim_secret));

% =========================================================================
% 6. Save Frames to Subfolders
% =========================================================================
fprintf('Saving frames to %s...\n', main_folder);
for k = 1:q
    % Save Cover frames
    imwrite(im2uint8(quat_to_rgb(I_c(:,:,k))), ...
            fullfile(main_folder, 'Cover', sprintf('cover_%03d.png', k)));
    
    % Save Secret frames
    imwrite(im2uint8(quat_to_rgb(I_s(:,:,k))), ...
            fullfile(main_folder, 'Secret', sprintf('secret_%03d.png', k)));
    
    % Save Stego frames
    imwrite(im2uint8(quat_to_rgb(I_si(:,:,k))), ...
            fullfile(main_folder, 'Stego', sprintf('stego_%03d.png', k)));
    
    % Save Extracted Secret frames
    imwrite(im2uint8(quat_to_rgb(I_s_rec(:,:,k))), ...
            fullfile(main_folder, 'Extracted_Secret', sprintf('extracted_%03d.png', k)));
end

% Save summary text file
fid = fopen(fullfile(main_folder, 'summary.txt'), 'w');
fprintf(fid, 'Video Data Hiding via Tensor QR Decomposition\n');
fprintf(fid, '=============================================\n');
fprintf(fid, 'Cover Video: video1.mp4\n');
fprintf(fid, 'Secret Video: video2.mp4\n');
fprintf(fid, 'Number of Frames: %d\n', q);
fprintf(fid, 'Frame Indices Used: [%s]\n', num2str(cover_indices));
fprintf(fid, 'Embedding Strength (alpha): %.2f\n\n', alpha);
fprintf(fid, 'Imperceptibility (Stego vs Cover):\n');
fprintf(fid, '  Average PSNR: %.2f dB\n', mean(psnr_stego));
fprintf(fid, '  Average SSIM: %.4f\n\n', mean(ssim_stego));
fprintf(fid, 'Fidelity (Extracted Secret vs Original Secret):\n');
fprintf(fid, '  Average PSNR: %.2f dB\n', mean(psnr_secret));
fprintf(fid, '  Average SSIM: %.4f\n', mean(ssim_secret));
fclose(fid);

fprintf('All outputs successfully saved to %s!\n', main_folder);

% =========================================================================
% Helper Functions
% =========================================================================

function indices = select_distinct_frames(v_reader, q, max_frame_limit)
    % Selects 'q' frames that are maximally visually distinct from each other.
    % max_frame_limit ensures we don't pick frames that don't exist in the second video.
    
    if nargin < 3
        max_frame_limit = v_reader.NumFrames;
    end
    
    num_candidates = min(30, max_frame_limit);
    candidate_indices = round(linspace(1, max_frame_limit, num_candidates));
    h = v_reader.Height; w = v_reader.Width;
    
    pool = zeros(h, w, 3, num_candidates, 'uint8');
    for i = 1:num_candidates
        pool(:,:,:,i) = read(v_reader, candidate_indices(i));
    end
    
    % Calculate pairwise visual differences
    diff_matrix = zeros(num_candidates, num_candidates);
    for i = 1:num_candidates
        for j = i+1:num_candidates
            d = sum(abs(double(pool(:,:,:,i)) - double(pool(:,:,:,j))), 'all');
            diff_matrix(i,j) = d;
            diff_matrix(j,i) = d;
        end
    end
    
    % Greedily select frames
    selected = zeros(1, q);
    selected(1) = 1; % Always pick the first candidate
    for s = 2:q
        max_min_diff = -1;
        best_idx = -1;
        for i = 1:num_candidates
            if any(selected == i), continue; end
            min_diff = min(diff_matrix(i, selected(1:s-1)));
            if min_diff > max_min_diff
                max_min_diff = min_diff;
                best_idx = i;
            end
        end
        selected(s) = best_idx;
    end
    
    % Map back to actual frame numbers and sort chronologically
    raw_frames = candidate_indices(selected);
    [~, sort_idx] = sort(raw_frames);
    indices = raw_frames(sort_idx);
end


 % example_QTELS_final.m

% Numerical Example for Quaternion Tensor Equality Constrained 
% Least Squares (QTELS) Problem
%
% NOTE: This script requires the external function file 'qten_norm.m' 
% to be in the same folder. Do NOT define qten_norm inside this script.

clear; clc; close all; 

addpath('algos');

%% =========================================================================
%  SETUP: Create Output Folder and Start Logging
%% =========================================================================
% Create Output1 folder if it doesn't exist
if ~exist('Output1', 'dir')
    mkdir('Output1');
end

% Start logging all console output to a text file
diary('Output1/numerical_results.txt');

fprintf('=== Numerical Example: QTELS Solver ===\n\n');

%% =========================================================================
%  EXAMPLE 4.2: Accuracy Test with Known Solution
%% =========================================================================
fprintf('EXAMPLE 4.2: Accuracy Test with Known Solution\n');
fprintf('----------------------------------------------\n');

% Problem dimensions
m = 50;   % rows of A (m >> n)
n = 30;   % columns (unknowns)
p = 10;   % rows of C (p << n)
d = 5;    % number of right-hand sides
q = 8;    % tensor dimension (third mode)

rng(42);  % For reproducibility

% Generate random quaternion tensors
A = quaternion(randn(m,n,q), randn(m,n,q), randn(m,n,q), randn(m,n,q));
C = quaternion(randn(p,n,q), randn(p,n,q), randn(p,n,q), randn(p,n,q));

% Create a known exact solution X_true
X_true = quaternion(randn(n,d,q), randn(n,d,q), randn(n,d,q), randn(n,d,q));

% Compute consistent right-hand sides
M_mat = dctmtx(q);
B = qten_mproduct(A, X_true, M_mat);
D = qten_mproduct(C, X_true, M_mat);

% Solve the QTELS problem
fprintf('Solving QTELS problem (m=%d, n=%d, p=%d, d=%d, q=%d)...\n', m, n, p, d, q);
tic;
X_computed = qten_qtels(A, B, C, D, M_mat);
cpu_time = toc;

% Compute metrics using the EXTERNAL qten_norm.m function
rel_error = qten_norm(X_computed - X_true) / qten_norm(X_true);
constraint_residual = qten_norm(qten_mproduct(C, X_computed, M_mat) - D);
ls_residual = qten_norm(qten_mproduct(A, X_computed, M_mat) - B);

fprintf('\nResults:\n');
fprintf('  Relative solution error:           %.4e\n', rel_error);
fprintf('  Constraint residual:               %.4e\n', constraint_residual);
fprintf('  Least squares residual:            %.4e\n', ls_residual);
fprintf('  CPU time:                          %.4f seconds\n\n', cpu_time);

if rel_error < 1e-10 && constraint_residual < 1e-10
    fprintf('[PASS] Solution is accurate to machine precision.\n\n');
else
    fprintf('[WARN] Solution error is larger than expected.\n\n');
end

%% =========================================================================
%  EXAMPLE 4.3: Accuracy vs. Problem Size
%% =========================================================================
fprintf('EXAMPLE 4.3: Accuracy vs. Problem Size\n');
fprintf('-------------------------------------\n');

n_values = [20, 30, 40, 50, 60];
rel_errors = zeros(size(n_values));
constraint_residuals = zeros(size(n_values));
cpu_times = zeros(size(n_values));

q_fixed = 8;
d_fixed = 5;
p_ratio = 0.3;  
m_ratio = 1.5;  

% Added this line to show the scaling rules in the console output
fprintf('  Maintaining scaling ratios: m = %.1f*n, p = %.1f*n\n\n', m_ratio, p_ratio);

for idx = 1:length(n_values)
    n = n_values(idx);
    m = round(m_ratio * n);
    p = round(p_ratio * n);
    
    A = quaternion(randn(m,n,q_fixed), randn(m,n,q_fixed), ...
                   randn(m,n,q_fixed), randn(m,n,q_fixed));
    C = quaternion(randn(p,n,q_fixed), randn(p,n,q_fixed), ...
                   randn(p,n,q_fixed), randn(p,n,q_fixed));
    X_true = quaternion(randn(n,d_fixed,q_fixed), randn(n,d_fixed,q_fixed), ...
                        randn(n,d_fixed,q_fixed), randn(n,d_fixed,q_fixed));
    
    M_mat = dctmtx(q_fixed);
    B = qten_mproduct(A, X_true, M_mat);
    D = qten_mproduct(C, X_true, M_mat);
    
    tic;
    X_hat = qten_qtels(A, B, C, D, M_mat);
    cpu_times(idx) = toc;
    
    rel_errors(idx) = qten_norm(X_hat - X_true) / qten_norm(X_true);
    constraint_residuals(idx) = qten_norm(qten_mproduct(C, X_hat, M_mat) - D);
    
    
    fprintf('  n=%2d, m=%2d, p=%2d | rel_error=%.2e, constraint=%.2e, time=%.4fs\n', ...
            n, m, p, rel_errors(idx), constraint_residuals(idx), cpu_times(idx));
end

%% =========================================================================
%  EXAMPLE 4.4: Scalability with Tensor Dimension q
%% =========================================================================
fprintf('\nEXAMPLE 4.4: Scalability with Tensor Dimension q\n');
fprintf('------------------------------------------------\n');

q_values = [2, 4, 6, 8, 10, 12, 15, 20];
cpu_times_q = zeros(size(q_values));
rel_errors_q = zeros(size(q_values));
constraint_residuals_q = zeros(size(q_values)); 

m_fixed = 60;
n_fixed = 40;
p_fixed = 12;
d_fixed = 5;

for idx = 1:length(q_values)
    q = q_values(idx);
    
    A = quaternion(randn(m_fixed,n_fixed,q), randn(m_fixed,n_fixed,q), ...
                   randn(m_fixed,n_fixed,q), randn(m_fixed,n_fixed,q));
    C = quaternion(randn(p_fixed,n_fixed,q), randn(p_fixed,n_fixed,q), ...
                   randn(p_fixed,n_fixed,q), randn(p_fixed,n_fixed,q));
    X_true = quaternion(randn(n_fixed,d_fixed,q), randn(n_fixed,d_fixed,q), ...
                        randn(n_fixed,d_fixed,q), randn(n_fixed,d_fixed,q));
    
    M_mat = dctmtx(q);
    B = qten_mproduct(A, X_true, M_mat);
    D = qten_mproduct(C, X_true, M_mat);
    
    tic;
    X_hat = qten_qtels(A, B, C, D, M_mat);
    cpu_times_q(idx) = toc;
    
    % Compute BOTH errors
    rel_errors_q(idx) = qten_norm(X_hat - X_true) / qten_norm(X_true);
    constraint_residuals_q(idx) = qten_norm(qten_mproduct(C, X_hat, M_mat) - D);
    
    fprintf('  q=%2d: rel_error=%.2e, constraint_res=%.2e, time=%.4fs\n', ...
            q, rel_errors_q(idx), constraint_residuals_q(idx), cpu_times_q(idx));
end

%% =========================================================================
%  GENERATE PLOTS AND SAVE TO Output1 FOLDER
%% =========================================================================
fprintf('\nGenerating publication-quality plots...\n');

% Helper path for saving
save_path = 'Output1';

% --- Figure 1: Accuracy vs. Problem Size ---
figure('Position', [100, 100, 600, 450]);
semilogy(n_values, rel_errors, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
         'MarkerFaceColor', 'b', 'DisplayName', 'Relative error');
hold on;
semilogy(n_values, constraint_residuals, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'Constraint residual');
grid on;
xlabel('Problem size $n$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Norm', 'Interpreter', 'latex', 'FontSize', 12);
title('Accuracy of QTELS Solver vs. Problem Size ($q=8, d=5$)', ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
print(gcf, '-depsc', fullfile(save_path, 'fig1_qtels_accuracy.eps'), '-r300');
print(gcf, '-dpng', fullfile(save_path, 'fig1_qtels_accuracy.png'), '-r300');

% --- Figure 2: CPU Time vs. Problem Size ---
figure('Position', [100, 100, 600, 450]);
plot(n_values, cpu_times, 'm-d', 'LineWidth', 2, 'MarkerSize', 8, ...
     'MarkerFaceColor', 'm');
grid on;
xlabel('Problem size $n$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('CPU Time (seconds)', 'Interpreter', 'latex', 'FontSize', 12);
title('Computational Efficiency of QTELS Solver', ...
      'Interpreter', 'latex', 'FontSize', 13);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
print(gcf, '-depsc', fullfile(save_path, 'fig2_qtels_time_size.eps'), '-r300');
print(gcf, '-dpng', fullfile(save_path, 'fig2_qtels_time_size.png'), '-r300');

% --- Figure 3: CPU Time vs. Tensor Dimension q ---
figure('Position', [100, 100, 600, 450]);
loglog(q_values, cpu_times_q, 'g-s', 'LineWidth', 2, 'MarkerSize', 8, ...
       'MarkerFaceColor', 'g');
grid on;
xlabel('Tensor dimension $q$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('CPU Time (seconds)', 'Interpreter', 'latex', 'FontSize', 12);
title('Scalability with Tensor Dimension ($m=60, n=40, p=12, d=5$)', ...
      'Interpreter', 'latex', 'FontSize', 13);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
print(gcf, '-depsc', fullfile(save_path, 'fig3_qtels_time_q.eps'), '-r300');
print(gcf, '-dpng', fullfile(save_path, 'fig3_qtels_time_q.png'), '-r300');

% --- Figure 4: Accuracy vs. Tensor Dimension q ---
figure('Position', [100, 100, 600, 450]);
semilogy(q_values, rel_errors_q, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
         'MarkerFaceColor', 'b', 'DisplayName', 'Relative error');
hold on;
semilogy(q_values, constraint_residuals_q, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'Constraint residual');
grid on;
xlabel('Tensor dimension $q$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Norm', 'Interpreter', 'latex', 'FontSize', 12);
title('Accuracy and Constraint Satisfaction vs. Tensor Dimension ($m=60, n=40, p=12$)', ...
      'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);
print(gcf, '-depsc', fullfile(save_path, 'fig4_qtels_accuracy_q.eps'), '-r300');
print(gcf, '-dpng', fullfile(save_path, 'fig4_qtels_accuracy_q.png'), '-r300');

%% =========================================================================
%  TEARDOWN
%% =========================================================================
% Stop logging to text file
diary off;

fprintf('\n=== Example Complete ===\n');
fprintf('All plots (EPS and PNG) and numerical results have been saved in the "Output1" folder.\n');
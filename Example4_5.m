%% =========================================================================
%  Perturbation Analysis of QTELS Problem
%  Validates Theorem 3.1: Actual Error vs. Theoretical Upper Bound
%% =========================================================================
clear; clc; close all;

addpath('algos');

% Create output folder if it doesn't exist
if ~exist('Output3', 'dir')
    mkdir('Output3');
end

% Start logging all console output to a text file
diary('Output3/perturbation_results.txt');
fprintf('=== Perturbation Analysis of QTELS Problem ===\n\n');

%% =========================================================================
%  1. Setup Exact QTELS Problem
%% =========================================================================
fprintf('Setting up exact QTELS problem...\n');
m = 40; n = 20; p = 8; d = 4; q = 6;
M_mat = dctmtx(q);

% Generate random quaternion tensors
A = quaternion(randn(m,n,q), randn(m,n,q), randn(m,n,q), randn(m,n,q));
C = quaternion(randn(p,n,q), randn(p,n,q), randn(p,n,q), randn(p,n,q));

% Ensure C has full M-product row rank (add scaled identity to diagonal of slices)
for k = 1:q
    C(:,:,k) = C(:,:,k) + 0.5 * eye(p, n); 
end

% Generate exact solution
X_true = quaternion(randn(n,d,q), randn(n,d,q), randn(n,d,q), randn(n,d,q));

% Compute consistent right-hand sides
B = qten_mproduct(A, X_true, M_mat);
D = qten_mproduct(C, X_true, M_mat);

% Verify exact solver
X_exact = qten_qtels(A, B, C, D, M_mat);
fprintf('Exact solver relative error: %.2e\n\n', qten_norm(X_exact - X_true)/qten_norm(X_true));

%% =========================================================================
%  2. Define Noise Levels and Preallocate
%% =========================================================================
epsilons = logspace(-6, -1, 15);
num_eps = length(epsilons);

actual_errors = zeros(1, num_eps);
theoretical_bounds = zeros(1, num_eps);

% Precompute the constant part of the theoretical bound (independent of epsilon)
fprintf('Computing theoretical bound constant...\n');
U_Q_constant = compute_UQ_constant(A, B, C, D, X_true, M_mat);
fprintf('Theoretical bound constant (U_Q / eps) computed.\n\n');

%% =========================================================================
%  3. Sweep Through Noise Levels
%% =========================================================================
fprintf('Running perturbation experiments...\n');
fprintf('-------------------------------------------------------------\n');
fprintf('  %-10s | %-18s | %-18s\n', 'Epsilon', 'Actual Error', 'Theoretical Bound');
fprintf('-------------------------------------------------------------\n');

for i = 1:num_eps
    eps = epsilons(i);
    
    % Generate perturbations satisfying the uniform slice-wise bounds
    [A_hat, B_hat, C_hat, D_hat] = generate_perturbations(A, B, C, D, M_mat, eps);
    
    % Solve the perturbed QTELS problem
    X_hat = qten_qtels(A_hat, B_hat, C_hat, D_hat, M_mat);
    
    % Compute actual relative forward error
    actual_errors(i) = qten_norm(X_hat - X_true) / qten_norm(X_true);
    
    % Compute theoretical upper bound (ignoring O(eps^2) for the plot)
    theoretical_bounds(i) = eps * U_Q_constant;
    
    fprintf('  %-10.2e | %-18.2e | %-18.2e\n', ...
            eps, actual_errors(i), theoretical_bounds(i));
end
fprintf('-------------------------------------------------------------\n\n');

%% =========================================================================
%  4. Plot Results
%% =========================================================================
fprintf('Generating publication-quality plots...\n');

figure('Position', [100, 100, 700, 500]);
loglog(epsilons, actual_errors, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b'); hold on;
loglog(epsilons, theoretical_bounds, 'r--x', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('Noise Level $\epsilon$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Relative Forward Error', 'Interpreter', 'latex', 'FontSize', 14);
%title('Perturbation Analysis: Actual Error vs. Theoretical Upper Bound', 'Interpreter', 'latex', 'FontSize', 14);
legend('Actual Relative Error', 'Theoretical Bound $\mathcal{U}_Q$', 'Location', 'northwest', 'FontSize', 12);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

% Save plots
print(gcf, '-depsc', fullfile('Output3', 'fig_perturbation_analysis.eps'), '-r300');
print(gcf, '-dpng', fullfile('Output3', 'fig_perturbation_analysis.png'), '-r300');

%% =========================================================================
%  TEARDOWN
%% =========================================================================
% Stop logging to text file
diary off;

fprintf('\n=== Perturbation Analysis Complete ===\n');
fprintf('All plots (EPS and PNG) and numerical results have been saved in the "Output3" folder.\n');



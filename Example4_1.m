%% =========================================================================
%  EXAMPLE 4.1: Random A, B, C, D (Checking Rank Conditions & Inconsistent LS)
%% =========================================================================

addpath('algos');

fprintf('\nEXAMPLE 4.1: Random A, B, C, D (Checking Theoretical Assumptions)\n');
fprintf('---------------------------------------------------------------\n');

m = 50; n = 30; p = 10; d = 5; q = 8;
rng(99); % Different random seed

% 1. Generate completely random, independent quaternion tensors
A = quaternion(randn(m,n,q), randn(m,n,q), randn(m,n,q), randn(m,n,q));
B = quaternion(randn(m,d,q), randn(m,d,q), randn(m,d,q), randn(m,d,q));
C = quaternion(randn(p,n,q), randn(p,n,q), randn(p,n,q), randn(p,n,q));
D = quaternion(randn(p,d,q), randn(p,d,q), randn(p,d,q), randn(p,d,q));

M_mat = dctmtx(q);

% 2. CHECK THEORETICAL ASSUMPTIONS (Existence and Uniqueness)
% Transform A and C to the spectral domain
A_tilde = qten_mode3(A, M_mat);
C_tilde = qten_mode3(C, M_mat);

% FIX: Extract all components to standard double arrays FIRST to avoid indexing errors
A_s = double(A_tilde.s); A_x = double(A_tilde.x); 
A_y = double(A_tilde.y); A_z = double(A_tilde.z);

C_s = double(C_tilde.s); C_x = double(C_tilde.x); 
C_y = double(C_tilde.y); C_z = double(C_tilde.z);

rank_C_full = true;
rank_stacked_full = true;

for k = 1:q
    % Now safely extract slice k from the double arrays
    A0 = A_s(:,:,k); A1 = A_x(:,:,k); A2 = A_y(:,:,k); A3 = A_z(:,:,k);
    C0 = C_s(:,:,k); C1 = C_x(:,:,k); C2 = C_y(:,:,k); C3 = C_z(:,:,k);
    
    % Form real representations
    A_R = [A0, -A1, -A2, -A3; 
           A1,  A0, -A3,  A2; 
           A2,  A3,  A0, -A1; 
           A3, -A2,  A1,  A0];
           
    C_R = [C0, -C1, -C2, -C3; 
           C1,  C0, -C3,  C2; 
           C2,  C3,  C0, -C1; 
           C3, -C2,  C1,  C0];
           
    % Check Existence: C must have full M-product row rank (rank = 4p)
    if rank(C_R) < 4*p
        rank_C_full = false;
    end
    
    % Check Uniqueness: Stacked matrix [A; C] must have full column rank (rank = 4n)
    M_stacked_R = [A_R; C_R];
    if rank(M_stacked_R) < 4*n
        rank_stacked_full = false;
    end
end

fprintf('Checking Theorem 2.1 Assumptions:\n');
if rank_C_full
    fprintf('  [PASS] C has full M-product row rank (Constraint is CONSISTENT).\n');
else
    fprintf('  [FAIL] C does not have full M-product row rank.\n');
end

if rank_stacked_full
    fprintf('  [PASS] null_M(A) intersect null_M(C) = {0} (Solution is UNIQUE).\n');
else
    fprintf('  [FAIL] Null space intersection condition failed.\n');
end

% 3. SOLVE THE FULL QTELS PROBLEM
fprintf('\nSolving full QTELS problem...\n');
tic;
X_hat = qten_qtels(A, B, C, D, M_mat);
cpu_time_ex4 = toc;
fprintf('CPU Time: %.4f seconds\n', cpu_time_ex4);

% 4. VERIFY THE SOLUTION
res_C_ex4 = qten_norm(qten_mproduct(C, X_hat, M_mat) - D);
res_A_ex4 = qten_norm(qten_mproduct(A, X_hat, M_mat) - B);

fprintf('\nResults:\n');
fprintf('  Constraint Residual || C *_M X - D ||_F : %.4e\n', res_C_ex4);
fprintf('  Least Squares Residual || A *_M X - B ||_F : %.4e\n', res_A_ex4);

if res_C_ex4 < 1e-10
    fprintf('\n[PASS] The solver strictly satisfied the equality constraint.\n');
end
if res_A_ex4 > 1e-5
    fprintf('[INFO] The least squares residual is > 0, confirming A *_M X = B is inconsistent.\n');
    fprintf('       The solver minimized the LS residual while keeping the constraint exact.\n');
end
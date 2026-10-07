function X = qten_qtels(A, B, C, D, M_mat)
% QTELS_QTELS: Solves the Quaternion Tensor Equality Constrained Least Squares problem
%  
%
%   Mathematical problem:
%       min || A *_M X - B ||_F   subject to   C *_M X = D
%
%   Inputs:
%       A     : quaternion tensor (m x n x q)
%       B     : quaternion tensor (m x d x q)
%       C     : quaternion tensor (p x n x q)
%       D     : quaternion tensor (p x d x q)
%       M_mat : real orthogonal matrix (q x q)
%
%   Output:
%       X     : quaternion tensor (n x d x q), the unique minimum-norm solution
%
%   Assumptions (from Theorem 3.2):
%       - m >> (n+d), p << n
%       - C has full M-product row rank
%       - null_M(A) intersect null_M(C) = {0}

    % Safe dimension extraction for A
    szA = size(A);
    m = szA(1);
    n = szA(2);
    q_dim = 1;
    if length(szA) >= 3
        q_dim = szA(3);
    end
    
    % Safe dimension extraction for B
    szB = size(B);
    d = szB(2);
    
    % Safe dimension extraction for C
    szC = size(C);
    p = szC(1);

    % =====================================================================
    % Step 1: Compute real representations (Lemma 1.6, 1.7 from the paper)
    % =====================================================================
    AR  = qten_real_rep(A);    % 4m x 4n     x q  (real tensor)
    BcR = qten_real_col(B);    % 4m x d      x q  (real tensor)
    CR  = qten_real_rep(C);    % 4p x 4n     x q  (real tensor)
    DcR = qten_real_col(D);    % 4p x d      x q  (real tensor)

    % =====================================================================
    % Step 2: M-product QR factorization of (C^R)^T
    %   (C^R)^T = Q *_M [R; 0]
    % =====================================================================
    CR_T = permute(CR, [2, 1, 3]);                  % 4n x 4p x q
    [Q_ten, R_full] = qten_mqr(CR_T, M_mat);
    % Q_ten  : 4n x 4n     x q  (M-orthogonal)
    % R_full : 4n x 4p     x q  (M-upper triangular, bottom zeros)

    % Extract the nonsingular upper 4p x 4p block (since p << n)
    R_ten = R_full(1:4*p, :, :);                    % 4p x 4p x q

    % =====================================================================
    % Step 3: Partition Q = [Q1, Q2]
    % =====================================================================
    Q1 = Q_ten(:, 1:4*p, :);                        % 4n x 4p       x q
    Q2 = Q_ten(:, 4*p+1:end, :);                    % 4n x (4n-4p)  x q

    % =====================================================================
    % Step 4: Compute A_a = A^R *_M Q1,  A_b = A^R *_M Q2
    % =====================================================================
    Aa = mproduct_real(AR, Q1, M_mat);              % 4m x 4p       x q
    Ab = mproduct_real(AR, Q2, M_mat);              % 4m x (4n-4p)  x q

    % =====================================================================
    % Step 5: Compute X_a = (R^T)^{-1} *_M D_c^R
    %   Solved efficiently via slice-wise triangular solve in transform domain
    % =====================================================================
    RT = permute(R_ten, [2, 1, 3]);                 % R^T (M-lower triangular)
    Xa = minverse_mproduct_real(RT, DcR, M_mat);    % 4p x d x q

    % =====================================================================
    % Step 6: Compute X_b = A_b^\dagger *_M (B_c^R - A_a *_M X_a)
    % =====================================================================
    Aa_Xa = mproduct_real(Aa, Xa, M_mat);           % 4m x d x q
    rhs   = BcR - Aa_Xa;                            % 4m x d x q

    Ab_pinv = qten_mpinv(Ab, M_mat);                % (4n-4p) x 4m x q
    Xb = mproduct_real(Ab_pinv, rhs, M_mat);        % (4n-4p) x d x q

    % =====================================================================
    % Step 7: Reconstruct full solution
    %   Xscr = Q *_M [Xa; Xb]
    % =====================================================================
    Xab = cat(1, Xa, Xb);                           % 4n x d x q
    Xscr = mproduct_real(Q_ten, Xab, M_mat);        % 4n x d x q

    % =====================================================================
    % Step 8: Extract quaternion components and reconstruct X
    % =====================================================================
    X0 = Xscr(1:n,         :, :);
    X1 = Xscr(n+1:2*n,     :, :);
    X2 = Xscr(2*n+1:3*n,   :, :);
    X3 = Xscr(3*n+1:4*n,   :, :);

    X = quaternion(X0, X1, X2, X3);
end




%% =========================================================================
%  Internal Helper: Solve R^T *_M X = D via triangular solve
%% =========================================================================
function X = minverse_mproduct_real(RT, D, M_mat)
% Computes X = (R^T)^{-1} *_M D efficiently by solving R^T *_M X = D
% slice-wise in the transform domain.
%   RT : M-lower triangular real tensor (p x p x q)
%   D  : real tensor (p x d x q)
%   X  : real tensor (p x d x q)

    % Safe dimension extraction
    szRT = size(RT);
    p = szRT(1);
    q_dim = 1;
    if length(szRT) >= 3
        q_dim = szRT(3);
    end
    
    szD = size(D);
    d = szD(2);

    % Forward transform
    RT_t = mode3_real_tensor(RT, M_mat);
    D_t  = mode3_real_tensor(D,  M_mat);

    % Slice-wise triangular solve (MATLAB's '\' handles this efficiently)
    X_t = zeros(p, d, q_dim);
    for k = 1:q_dim
        X_t(:,:,k) = RT_t(:,:,k) \ D_t(:,:,k);
    end

    % Inverse transform
    X = mode3_real_tensor(X_t, M_mat');
end


%% =========================================================================
%  Internal Helper: M-product for purely real tensors
%% =========================================================================
function C = mproduct_real(A, B, M_mat)
% M-product C = A *_M B for real tensors, using mode3_real_tensor.
    
    % Safe dimension extraction
    szA = size(A);
    m = szA(1);
    p = szA(2);
    q_dim = 1;
    if length(szA) >= 3
        q_dim = szA(3);
    end
    
    szB = size(B);
    n = szB(2);

    A_t = mode3_real_tensor(A, M_mat);
    B_t = mode3_real_tensor(B, M_mat);

    C_t = zeros(m, n, q_dim);
    for k = 1:q_dim
        C_t(:,:,k) = A_t(:,:,k) * B_t(:,:,k);
    end

    C = mode3_real_tensor(C_t, M_mat');
end
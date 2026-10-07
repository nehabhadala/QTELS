function [Q_ten, R_ten] = qten_mqr(A, M_mat)
% qten_mqr: M-product QR factorization of real tensor A.
%   A = Q *_M R,  with Q M-orthogonal, R M-upper triangular.
%
% Inputs:
%   A     : real tensor (m x n x q)
%   M_mat : real orthogonal matrix (q x q)
% Outputs:
%   Q_ten : real tensor (m x m x q), M-orthogonal
%   R_ten : real tensor (m x n x q), M-upper triangular

    % Safe dimension extraction
    sz = size(A);
    m = sz(1);
    n = sz(2);
    q = 1;
    if length(sz) >= 3
        q = sz(3);
    end

    % Step 1: Forward mode-3 transform
    A_tilde = mode3_real_tensor(A, M_mat);

    % Step 2: Slice-wise QR
    Q_tilde = zeros(m, m, q);
    R_tilde = zeros(m, n, q);
    for k = 1:q
        [Q_tilde(:,:,k), R_tilde(:,:,k)] = qr(A_tilde(:,:,k));
    end

    % Step 3: Inverse mode-3 transform
    Minv = M_mat';
    Q_ten = mode3_real_tensor(Q_tilde, Minv);
    R_ten = mode3_real_tensor(R_tilde, Minv);
end
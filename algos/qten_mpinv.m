function A_pinv = qten_mpinv(A, M_mat)
% qten_mpinv: M-pseudoinverse of real tensor A.
%   Computed slice-wise in the transform domain.
%
% Inputs:
%   A     : real tensor (m x n x q)
%   M_mat : real orthogonal matrix (q x q)
% Output:
%   A_pinv : real tensor (n x m x q)

    % Safe dimension extraction
    sz = size(A);
    m = sz(1);
    n = sz(2);
    q = 1;
    if length(sz) >= 3
        q = sz(3);
    end

    % Step 1: Forward mode-3 transform to the spectral domain
    A_tilde = mode3_real_tensor(A, M_mat);
    
    % Step 2: Slice-wise pseudoinverse in the transform domain
    Ap_tilde = zeros(n, m, q);
    for k = 1:q
        Ap_tilde(:,:,k) = pinv(A_tilde(:,:,k));
    end
    
    % Step 3: Inverse mode-3 transform back to the original domain
    % Since M is orthogonal, M^{-1} = M^T
    A_pinv = mode3_real_tensor(Ap_tilde, M_mat');
end
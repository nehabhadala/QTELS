function A_tilde = mode3_real_tensor(A, M_mat)
% mode3_real_tensor: Mode-3 product for purely real tensors
%   More efficient version avoiding permute operations
%
% Inputs:
%   A     : real tensor (m x n x q)
%   M_mat : real matrix (r x q)
% Output:
%   A_tilde : real tensor (m x n x r)

    sz = size(A);
    m = sz(1);
    n = sz(2);
    q = 1;
    if length(sz) >= 3
        q = sz(3);
    end
    
    szM = size(M_mat);
    r = szM(1);
    
    A_unfold = reshape(A, [m*n, q]);      % (mn) x q
    B_unfold = A_unfold * M_mat.';         % (mn) x r
    A_tilde = reshape(B_unfold, [m, n, r]); % m x n x r
end
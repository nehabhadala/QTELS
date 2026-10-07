function C = qten_mproduct(A, B, M_mat)
% qten_mproduct: Tensor M-product  C = A *_M B
%   = ( (A x_3 M) triangle (B x_3 M) ) x_3 M^{-1}
%
% Inputs:
%   A     : quaternion tensor (m x p x q)
%   B     : quaternion tensor (p x n x q)
%   M_mat : real orthogonal matrix (q x q)
% Output:
%   C     : quaternion tensor (m x n x q)

    Minv = M_mat';   % M is orthogonal, so M^{-1} = M^T

    A_tilde = qten_mode3(A, M_mat);
    B_tilde = qten_mode3(B, M_mat);
    C_tilde = qten_facewise(A_tilde, B_tilde);
    C       = qten_mode3(C_tilde, Minv);
end
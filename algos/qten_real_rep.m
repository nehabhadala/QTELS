function AR = qten_real_rep(A)
% qten_real_rep: Real block representation of quaternion tensor.
%   For A in Q^{m x n x q}, AR is in R^{4m x 4n x q}.
%
% Input:
%   A : quaternion tensor (m x n x q)
% Output:
%   AR : real tensor (4m x 4n x q)

    % Safe dimension extraction
    sz = size(A);
    m = sz(1);
    n = sz(2);
    q = 1;
    if length(sz) >= 3
        q = sz(3);
    end
    
    AR = zeros(4*m, 4*n, q);

    A0 = A.s;  A1 = A.x;  A2 = A.y;  A3 = A.z;

    for k = 1:q
        a0 = A0(:,:,k); a1 = A1(:,:,k);
        a2 = A2(:,:,k); a3 = A3(:,:,k);
        AR(:,:,k) = [ a0, -a1, -a2, -a3;
                      a1,  a0, -a3,  a2;
                      a2,  a3,  a0, -a1;
                      a3, -a2,  a1,  a0 ];
    end
end
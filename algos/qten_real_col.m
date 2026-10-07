function AcR = qten_real_col(A)
% qten_real_col: Real column representation of quaternion tensor.
%   For A in Q^{m x n x q}, AcR is in R^{4m x n x q}.
%
% Input:
%   A : quaternion tensor (m x n x q)
% Output:
%   AcR : real tensor (4m x n x q)

    % Safe dimension extraction
    sz = size(A);
    m = sz(1);
    n = sz(2);
    q = 1;
    if length(sz) >= 3
        q = sz(3);
    end
    
    AcR = zeros(4*m, n, q);

    A0 = A.s;  A1 = A.x;  A2 = A.y;  A3 = A.z;

    for k = 1:q
        AcR(:, :, k) = [ A0(:,:,k);
                         A1(:,:,k);
                         A2(:,:,k);
                         A3(:,:,k) ];
    end
end
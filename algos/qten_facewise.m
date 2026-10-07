function C = qten_facewise(A, B)
% qten_facewise: Face-wise product C = A triangle B.
%   C(:,:,k) = A(:,:,k) * B(:,:,k) for each slice k.
%   Uses MATLAB's native quaternion matrix multiplication.
%
% Inputs:
%   A : quaternion tensor (m x p x q) or matrix (m x p)
%   B : quaternion tensor (p x n x q) or matrix (p x n)
% Output:
%   C : quaternion tensor (m x n x q) or matrix (m x n)

    % Safe dimension extraction for A
    szA = size(A);
    m = szA(1);
    p = szA(2);
    q = 1;
    if length(szA) >= 3
        q = szA(3);
    end
    
    % Safe dimension extraction for B
    szB = size(B);
    n = szB(2);
    
    % Handle 2D case (q = 1) separately for efficiency
    if q == 1
        C = A * B;  % Direct quaternion matrix multiplication
        return;
    end
    
    % 3D case: slice-wise multiplication
    C = quaternion(zeros(m, n, q), zeros(m, n, q), zeros(m, n, q), zeros(m, n, q));
    for k = 1:q
        % MATLAB's quaternion toolbox handles Hamilton product via *
        C(:,:,k) = A(:,:,k) * B(:,:,k);
    end
end
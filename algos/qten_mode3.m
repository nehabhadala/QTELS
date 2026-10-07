function C = qten_mode3(A, M)
% qten_mode3: Mode-3 product of quaternion tensor A with real matrix M.
%   C = A x_3 M
%
% Inputs:
%   A : quaternion tensor (m x n x q)
%   M : real matrix       (r x q)
% Output:
%   C : quaternion tensor (m x n x r)

    % Safe dimension extraction
    szA = size(A);
    n1 = szA(1);
    n2 = szA(2);
    n3 = 1;
    if length(szA) >= 3
        n3 = szA(3);
    end
    
    szM = size(M);
    r = szM(1);
    s = szM(2);
    
    % Check dimensions
    if n3 ~= s
        error('Dimension mismatch for mode-3 product.');
    end
    
    % Extract the four real component tensors
    A0 = A.s; 
    A1 = A.x; 
    A2 = A.y; 
    A3 = A.z;
    
    % STEP 1: Unfold each component tensor along mode-3
    A0_unfolded = reshape(A0, [n1 * n2, n3]);
    A1_unfolded = reshape(A1, [n1 * n2, n3]);
    A2_unfolded = reshape(A2, [n1 * n2, n3]);
    A3_unfolded = reshape(A3, [n1 * n2, n3]);
    
    % STEP 2: Matrix Multiplication for each component
    C0_unfolded = A0_unfolded * M.'; 
    C1_unfolded = A1_unfolded * M.'; 
    C2_unfolded = A2_unfolded * M.'; 
    C3_unfolded = A3_unfolded * M.'; 
    
    % STEP 3: Fold each resulting matrix back into a 3D tensor
    C0 = reshape(C0_unfolded, [n1, n2, r]);
    C1 = reshape(C1_unfolded, [n1, n2, r]);
    C2 = reshape(C2_unfolded, [n1, n2, r]);
    C3 = reshape(C3_unfolded, [n1, n2, r]);
    
    % STEP 4: Reconstruct the final quaternion tensor
    C = quaternion(C0, C1, C2, C3);
end
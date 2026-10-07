function n = qten_norm(T)
% qten_norm: Computes the Frobenius norm of a quaternion tensor.
%   Works for quaternion arrays of any dimension (1D, 2D, 3D, etc.)
%   Compatible with all MATLAB versions.

    % Extract the four real components and convert to double
    a = double(T.s);
    b = double(T.x);
    c = double(T.y);
    d = double(T.z);
    
    % Compute Frobenius norm
    n = sqrt(sum(a(:).^2 + b(:).^2 + c(:).^2 + d(:).^2));
end
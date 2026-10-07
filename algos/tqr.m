function [Q, R] = tqr(A, M)
    % Transform along mode-3
    A_hat = qten_mode3(A, M);
    [m, n, q] = size(A);
    
    Q_hat = quaternion(zeros(m,m,q), zeros(m,m,q), zeros(m,m,q), zeros(m,m,q));
    R_hat = quaternion(zeros(m,n,q), zeros(m,n,q), zeros(m,n,q), zeros(m,n,q));
    
    for k = 1:q
        % Perform Quaternion QR decomposition on each frontal slice
        [Q_hat(:,:,k), R_hat(:,:,k)] = qr(A_hat(:,:,k));
    end
    
    % Inverse transform along mode-3
    Q = qten_mode3(Q_hat, inv(M));
    R = qten_mode3(R_hat, inv(M));
end
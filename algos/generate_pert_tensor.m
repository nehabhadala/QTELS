function Delta_T = generate_pert_tensor(T, M_mat, eps)
    szT = size(T);
    m = szT(1); n2 = szT(2); q = szT(3);
    
    % Transform T to spectral domain
    T_tilde = qten_mode3(T, M_mat);
    
    % Initialize perturbation in spectral domain
    Delta_T_tilde = quaternion(zeros(m, n2, q), zeros(m, n2, q), zeros(m, n2, q), zeros(m, n2, q)); 
    
    for k = 1:q
        % Generate random quaternion matrix
        E = quaternion(randn(m, n2), randn(m, n2), randn(m, n2), randn(m, n2)); 
        % Normalize to unit Frobenius norm
        E = E / qten_norm(E);
        % Scale to exactly match the slice-wise bound: eps * ||T_tilde^(k)||_F
        Delta_T_tilde(:,:,k) = eps * qten_norm(T_tilde(:,:,k)) * E;
    end
    
    % Inverse transform back to original domain
    Delta_T = qten_mode3(Delta_T_tilde, M_mat');
end
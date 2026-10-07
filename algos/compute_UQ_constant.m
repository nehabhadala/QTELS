%% =========================================================================
%  Helper Functions
%% =========================================================================

function U_const = compute_UQ_constant(A, B, C, D, X_true, M_mat)
    [m, n, q] = size(A);
    d = size(B, 2);
    p = size(C, 1);
    
    % Transform tensors to the spectral domain
    A_tilde = qten_mode3(A, M_mat);
    B_tilde = qten_mode3(B, M_mat);
    C_tilde = qten_mode3(C, M_mat);
    D_tilde = qten_mode3(D, M_mat);
    X_tilde = qten_mode3(X_true, M_mat);
    
    U_max = 0;
    for k = 1:q
        % Extract k-th quaternion slices
        Ak_q = A_tilde(:,:,k); Bk_q = B_tilde(:,:,k);
        Ck_q = C_tilde(:,:,k); Dk_q = D_tilde(:,:,k);
        Xk_q = X_tilde(:,:,k);
        
        % Map to real representations (squeeze to get 2D matrices)
        Ak = squeeze(qten_real_rep(Ak_q)); % 4m x 4n
        Bk = squeeze(qten_real_col(Bk_q)); % 4m x d
        Ck = squeeze(qten_real_rep(Ck_q)); % 4p x 4n
        Dk = squeeze(qten_real_col(Dk_q)); % 4p x d
        Xk = squeeze(qten_real_col(Xk_q)); % 4n x d
        
        % Compute Residual Rk
        Rk = Bk - Ak * Xk;
        
        % Compute Projectors and Lk
        I_4n = eye(4*n);
        Ck_pinv = pinv(Ck);
        Pk = I_4n - Ck_pinv * Ck;
        
        Ak_Pk = Ak * Pk;
        Ak_Pk_pinv = pinv(Ak_Pk);
        
        Lk = (I_4n - Ak_Pk_pinv * Ak) * Ck_pinv;
        
        % Compute Condition Numbers
        K_A = norm(Ck, 'fro') * norm(Lk, 2);
        K_B = norm(Ak, 'fro') * norm(Ak_Pk_pinv, 2);
        
        % Compute Norms
        norm_Ak = norm(Ak, 'fro'); norm_Bk = norm(Bk, 'fro');
        norm_Ck = norm(Ck, 'fro'); norm_Dk = norm(Dk, 'fro');
        norm_Xk = norm(Xk, 'fro'); norm_Rk = norm(Rk, 'fro');
        
        % Compute U_Q^(k) / epsilon (The constant factor)
        term1 = K_A * (norm_Dk / (norm_Ck * norm_Xk) + 1);
        term2 = K_B * (norm_Bk / (norm_Ak * norm_Xk) + 1);
        term3 = (K_B^2) * (norm_Ck / norm_Ak * norm(Ak * Lk, 2) + 1) * (norm_Rk / (norm_Ak * norm_Xk));
        
        U_k = term1 + term2 + term3;
        U_max = max(U_max, U_k);
    end
    U_const = U_max;
end




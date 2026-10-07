function [A_hat, B_hat, C_hat, D_hat] = generate_perturbations(A, B, C, D, M_mat, eps)
    A_hat = A + generate_pert_tensor(A, M_mat, eps);
    B_hat = B + generate_pert_tensor(B, M_mat, eps);
    C_hat = C + generate_pert_tensor(C, M_mat, eps);
    D_hat = D + generate_pert_tensor(D, M_mat, eps);
end
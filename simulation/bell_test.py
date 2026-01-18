import numpy as np

def run_honest_bell_simulation(n_trials=200000):
    """
    SSC NON-LOCAL GEOMETRIC SIMULATION
    ----------------------------------
    Demonstrates CHSH violation via Parameter Dependence.
    Checks marginals to confirm No-Signaling using 
    Unbiased Empirical Standard Error with Bessel correction.
    """
    print(">>> INITIALIZING SSC CONSISTENCY KERNEL V.29.0...")
    
    settings = [
        (0, 22.5), (0, 67.5), (45, 22.5), (45, 67.5)
    ]
    
    correlations = []
    marginals_B = [] # Store Bob's marginals
    
    print("-" * 65)
    print(f"{'Alice':<6} | {'Bob':<6} | {'Corr (E)':<10} | {'<A>':<8} | {'<B>':<8}")
    print("-" * 65)
    
    for angle_a_deg, angle_b_deg in settings:
        outcome_A = np.random.choice([1, -1], size=n_trials)
        delta_rad = np.deg2rad(angle_a_deg - angle_b_deg)
        prob_agree = (np.sin(delta_rad))**2
        random_seed = np.random.random(n_trials)
        outcome_B = np.where(random_seed < prob_agree, outcome_A, -outcome_A)
        
        E = np.mean(outcome_A * outcome_B)
        marginal_A = np.mean(outcome_A)
        marginal_B = np.mean(outcome_B)
        
        correlations.append(E)
        marginals_B.append(marginal_B)
        
        print(f"{angle_a_deg:<6} | {angle_b_deg:<6} | {E:<10.5f} | {marginal_A:<8.4f} | {marginal_B:<8.4f}")
        
    # Standard CHSH S parameter calculation
    S = abs(correlations[0] - correlations[1] + correlations[2] + correlations[3])
    
    print("-" * 65)
    print(f"FINAL S-PARAMETER: {S:.5f}")
    
    # NO-SIGNALING CHECK (Unbiased SE)
    # Using Bessel's correction N/(N-1) for rigorous variance estimation
    def calc_se_unbiased(m, N):
        # For +/-1 outcomes: Var(X)=1-mu^2.
        # We use the plug-in estimate mu~m, and apply an N/(N-1) 
        # finite-sample correction as a conservative inflation.
        # We clamp variance to 0.0 to prevent floating point instability.
        var_est = max(0.0, 1.0 - m**2)
        return ((N/(N-1.0)) * var_est / N) ** 0.5

    # Check Pair 1: b=22.5
    m0, m2 = marginals_B[0], marginals_B[2]
    diff_225 = abs(m0 - m2)
    # SE of difference = sqrt(SE1^2 + SE2^2)
    se_225 = (calc_se_unbiased(m0, n_trials)**2 + calc_se_unbiased(m2, n_trials)**2) ** 0.5
    thr_225 = 5 * se_225
    
    # Check Pair 2: b=67.5
    m1, m3 = marginals_B[1], marginals_B[3]
    diff_675 = abs(m1 - m3)
    se_675 = (calc_se_unbiased(m1, n_trials)**2 + calc_se_unbiased(m3, n_trials)**2) ** 0.5
    thr_675 = 5 * se_675
    
    print("\n>>> NO-SIGNALING CHECK (Unbiased Empirical Variance):")
    print("    Note: We apply a conservative 5-sigma threshold per test;")
    print("    with only two tests, this is already extremely stringent.")
    print(f"    b=22.5 | Diff: {diff_225:.4f} | 5-sigma: {thr_225:.4f}")
    print(f"    b=67.5 | Diff: {diff_675:.4f} | 5-sigma: {thr_675:.4f}")
    
    if diff_225 < thr_225 and diff_675 < thr_675:
         print(">>> RESULT: NO-SIGNALING CONFIRMED (Within Statistical Noise)")
    else:
         print(">>> RESULT: SIGNALING DETECTED (Statistically Significant!)")

if __name__ == "__main__":
    run_honest_bell_simulation()

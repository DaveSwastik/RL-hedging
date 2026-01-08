import itertools
from src.agents.train_custom import train

def run_sweep():
    # Define parameter ranges
    thetas = [0.04, 0.16] # Low/High mean reversion level
    rhos = [-0.7, -0.3]   # Strong/Weak leverage effect
    kappas = [1.0, 3.0]   # Slow/Fast mean reversion speed

    print(f"Starting Parameter Sweep...")
    print(f"Thetas: {thetas}")
    print(f"Rhos: {rhos}")
    print(f"Kappas: {kappas}")

    combinations = list(itertools.product(thetas, rhos, kappas))
    
    for i, (theta, rho, kappa) in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)}: theta={theta}, rho={rho}, kappa={kappa} ---")
        
        save_name = f"models/sweep/hedger_t{theta}_r{rho}_k{kappa}.pth"
        
        try:
            train(
                config_path='src/configs/default.yaml',
                save_path=save_name,
                theta=theta,
                rho=rho,
                kappa=kappa
            )
            print(f"Run {i+1} completed. Model saved to {save_name}")
        except Exception as e:
            print(f"Run {i+1} failed with error: {e}")

if __name__ == "__main__":
    run_sweep()

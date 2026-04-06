import requests
import json
import matplotlib.pyplot as plt
import numpy as np

# Server URL - adjust if needed
SERVER_URL = "http://localhost:8000"

def run_server_simulation():
    """Run simulation through the server API"""
    # Initialize a patient
    init_response = requests.post(f"{SERVER_URL}/init", params={"patient": "adolescent#003"})
    init_data = init_response.json()
    patient_id = init_data["patient_id"]
    
    # Store data for plotting
    t = []
    CHO = []
    insulin = []
    BG = []
    
    # Current time and initial BG
    current_time = 0
    current_bg = init_data["initial_glucose"]
    BG.append(current_bg)
    t.append(current_time)
    CHO.append(0)
    insulin.append(0)  # No insulin at t=0
    
    # Run simulation for 50 time steps (same as the original script)
    while current_time < 50:
        
        # Determine if carbs are given (at t=5 in original script)
        carbs = 20 if current_time == 5 else 0
        
        # Make step request to server
        step_response = requests.post(
            f"{SERVER_URL}/step/{patient_id}",
            json={
                "glucose_reading": current_bg,
                "carbs": carbs,
                "controller_algorithm": "basal_bolus"
            }
        )
        
        current_time += 1
        step_data = step_response.json()
        print(step_data)
        # Update current values
        current_bg = step_data["glucose"]
        current_insulin = step_data["insulin"]
        
        # Store data for plotting
        t.append(current_time)
        CHO.append(carbs)
        insulin.append(current_insulin)
        BG.append(current_bg)
        
        print(f"Time: {current_time}, BG: {current_bg:.2f}, Insulin: {current_insulin:.4f}, Carbs: {carbs}")
    
    # Save data to compare with original simulation
    with open("server_sim.json", "w") as f:
        json.dump({"bg": BG, "cho": CHO, "insulin": insulin, "t": t}, f)
    
    return {"t": t, "BG": BG, "CHO": CHO, "insulin": insulin}

def load_original_simulation():
    """Load the original simulation data"""
    with open("sim.json", "r") as f:
        data = json.load(f)
    
    return {
        "t": list(range(len(data["bg"]))),  # Assuming t starts at 0 and increments by 1
        "BG": data["bg"],
        "CHO": data["cho"],
        "insulin": data["insulin"]
    }

def compare_simulations():
    """Compare original and server-based simulations"""
    # Run server simulation
    print("Running server simulation...")
    server_data = run_server_simulation()
    
    # Load original simulation data
    print("Loading original simulation data...")
    try:
        original_data = load_original_simulation()
    except FileNotFoundError:
        print("Original simulation data (sim.json) not found. Only plotting server simulation.")
        original_data = None
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot blood glucose
    axes[0].plot(server_data["t"], server_data["BG"], label="Server Simulation", marker="o", markersize=4)
    if original_data:
        axes[0].plot(original_data["t"], original_data["BG"], label="Original Simulation", marker="x", markersize=4)
    axes[0].set_ylabel("Blood Glucose")
    axes[0].set_title("Blood Glucose Comparison")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot CHO
    axes[1].plot(server_data["t"], server_data["CHO"], label="Server Simulation", marker="o", markersize=4)
    if original_data:
        axes[1].plot(original_data["t"], original_data["CHO"], label="Original Simulation", marker="x", markersize=4)
    axes[1].set_ylabel("CHO")
    axes[1].set_title("Carbohydrate Intake Comparison")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot insulin
    axes[2].plot(server_data["t"], server_data["insulin"], label="Server Simulation", marker="o", markersize=4)
    if original_data:
        axes[2].plot(original_data["t"], original_data["insulin"], label="Original Simulation", marker="x", markersize=4)
    axes[2].set_ylabel("Insulin")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Insulin Delivery Comparison")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Calculate RMSE if original data is available
    if original_data:
        # Ensure both arrays have the same length
        min_len = min(len(server_data["BG"]), len(original_data["BG"]))
        server_bg = np.array(server_data["BG"][:min_len])
        original_bg = np.array(original_data["BG"][:min_len])
        
        rmse = np.sqrt(np.mean((server_bg - original_bg) ** 2))
        print(f"RMSE between server and original simulations: {rmse:.4f}")
        
        # Add RMSE to the plot title
        axes[0].set_title(f"Blood Glucose Comparison (RMSE: {rmse:.4f})")
    
    # Save plot
    plt.savefig("simulation_comparison.png")
    print("Plot saved as 'simulation_comparison.png'")
    
    # Optional: Show plot if in interactive environment
    plt.show()

if __name__ == "__main__":
    compare_simulations()

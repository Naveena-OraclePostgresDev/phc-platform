import numpy as np
import pandas as pd
import time

def main():
    start_time = time.time()
    np.random.seed(42)  # For reproducibility

    # =========================================================================
    # SECTION 1: Configurations and Parameter Setups
    # =========================================================================
    print("[1/7] Initializing configurations and parameters...")
    
    # 6 districts and their geographical centers (Tamil Nadu region)
    district_centers = {
        'Vellore': (12.9165, 79.1325),
        'Salem': (11.6643, 78.1460),
        'Erode': (11.3410, 77.7172),
        'Trichy': (10.7905, 78.7047),
        'Madurai': (9.9252, 78.1198),
        'Coimbatore': (11.0168, 76.9558)
    }
    
    phcs_per_district = 50
    districts_list = list(district_centers.keys())
    n_phcs = len(districts_list) * phcs_per_district  # 300 PHCs
    
    # Date range: 2024-09-01 to 2026-08-31 (730 days)
    dates = pd.date_range(start='2024-09-01', end='2026-08-31', freq='D')
    n_dates = len(dates)
    dates_str = dates.strftime('%Y-%m-%d').values
    months = dates.month.values  # Used for seasonality calculations
    
    # Medicines list and baseline consumption rate (units per patient)
    medicines = np.array([
        'Paracetamol', 'ORS', 'Amoxicillin', 'Metformin', 
        'Iron Folic Acid', 'Anti-Rabies Vaccine', 'Antimalarial', 'Anti-Snake Venom'
    ])
    n_meds = len(medicines)
    
    # Target minimum stock quantities to prevent unrealistic zeros
    med_min_targets = np.array([300, 200, 100, 150, 150, 50, 40, 20])
    # Baseline demand multipliers per medicine
    med_baseline_rates = np.array([1.5, 0.8, 0.3, 0.5, 0.6, 0.05, 0.02, 0.01])

    # =========================================================================
    # SECTION 2: Generate PHC Metadata (Clustered Coordinates, Capacity, Size)
    # =========================================================================
    print("[2/7] Generating clustered PHC metadata...")
    
    phc_ids = []
    phc_names = []
    phc_districts = []
    lats = []
    lons = []
    
    # Base indicators of clinic scale (size factors)
    phc_size_factors = []
    
    phc_idx = 1
    for dist in districts_list:
        center_lat, center_lon = district_centers[dist]
        for i in range(1, phcs_per_district + 1):
            phc_ids.append(f"PHC{phc_idx:03d}")
            phc_names.append(f"{dist} PHC {i:02d}")
            phc_districts.append(dist)
            
            # Cluster PHCs around district center with small random variation
            lat_jitter = np.random.normal(0, 0.12)
            lon_jitter = np.random.normal(0, 0.12)
            
            # Ensure coordinates stay within specified boundaries
            lats.append(np.clip(center_lat + lat_jitter, 8.5, 13.5))
            lons.append(np.clip(center_lon + lon_jitter, 76.5, 80.5))
            
            # Assign a random scale factor (0.5 = small/quiet, 2.0 = large/busy)
            phc_size_factors.append(np.random.uniform(0.5, 2.0))
            phc_idx += 1

    phc_ids = np.array(phc_ids)
    phc_names = np.array(phc_names)
    phc_districts = np.array(phc_districts)
    lats = np.array(lats)
    lons = np.array(lons)
    phc_size_factors = np.array(phc_size_factors)
    
    # Capacity parameters scaled to PHC size
    total_beds = np.round(phc_size_factors * 15 + np.random.uniform(2, 8, size=n_phcs)).astype(int)
    staff_sanctioned = np.round(phc_size_factors * 10 + np.random.uniform(4, 10, size=n_phcs)).astype(int)
    baseline_footfall = np.round(phc_size_factors * 30 + np.random.uniform(10, 20, size=n_phcs)).astype(int)

    # =========================================================================
    # SECTION 3: Simulate Temporal Attributes (Footfall, Beds, Staff)
    # =========================================================================
    print("[3/7] Simulating daily patient footfall, beds, and staff attendance...")
    
    # Broadcast dates and months to a 2D grid of shape (n_phcs, n_dates)
    monsoon_mask = (months >= 6) & (months <= 9)  # True for June to Sept
    monsoon_2d = np.tile(monsoon_mask, (n_phcs, 1))
    
    # Baseline footfall matrix
    footfall_base_2d = np.tile(baseline_footfall[:, np.newaxis], (1, n_dates))
    
    # 1.8x footfall surge during monsoons
    footfall_mult_2d = np.where(monsoon_2d, 1.8, 1.0)
    footfall_noise = np.random.normal(1.0, 0.12, size=(n_phcs, n_dates))
    
    patient_footfall_2d = np.round(footfall_base_2d * footfall_mult_2d * footfall_noise).astype(int)
    patient_footfall_2d = np.clip(patient_footfall_2d, 5, None)  # Ensure a rational minimum
    
    # Staff Attendance: 60% to 100% capacity
    attendance_rate = np.random.uniform(0.60, 1.00, size=(n_phcs, n_dates))
    staff_present_2d = np.round(staff_sanctioned[:, np.newaxis] * attendance_rate).astype(int)
    staff_present_2d = np.clip(staff_present_2d, 1, staff_sanctioned[:, np.newaxis])  # minimum 1 staff present
    
    # Bed Occupancy: 40% to 95%, correlated with relative patient footfall
    footfall_ratio = patient_footfall_2d / footfall_base_2d
    occupancy_rate = 0.40 + 0.45 * (footfall_ratio / np.max(footfall_ratio))
    occupancy_rate += np.random.normal(0, 0.05, size=(n_phcs, n_dates))  # Add minor noise
    occupancy_rate = np.clip(occupancy_rate, 0.40, 0.95)
    
    occupied_beds_2d = np.round(total_beds[:, np.newaxis] * occupancy_rate).astype(int)
    occupied_beds_2d = np.minimum(occupied_beds_2d, total_beds[:, np.newaxis])

    # =========================================================================
    # SECTION 4: Setup Medicine Demand and Replenishment Schedules
    # =========================================================================
    print("[4/7] Generating inventory configurations and restocking schedules...")
    
    # Total unique series: 300 PHCs * 8 medicines = 2,400 series
    n_series = n_phcs * n_meds
    
    # Map each series index (0 to 2399) back to its PHC index and Medicine index
    series_phc_idx = np.arange(n_series) // n_meds
    series_med_idx = np.arange(n_series) % n_meds
    
    # Repeat the 2D matrices across medicines to create matching 2,400 x 730 arrays
    footfall_series = patient_footfall_2d[series_phc_idx, :]
    monsoon_series = monsoon_2d[series_phc_idx, :]
    
    # Base consumption rates and monsoon surge multipliers
    med_rates = med_baseline_rates[series_med_idx][:, np.newaxis]
    
    # Generate daily medicine-specific monsoon multipliers
    # Paracetamol (index 0) and ORS (index 1) rise 2.5x to 3x in monsoon. Others remain stable.
    monsoon_mults_base = np.ones((n_series, 1))
    monsoon_mults_base[series_med_idx == 0] = 2.75
    monsoon_mults_base[series_med_idx == 1] = 2.75
    monsoon_mults_base[series_med_idx == 6] = 1.40  # Minor increase for antimalarials
    
    daily_med_mult = np.where(monsoon_series, np.tile(monsoon_mults_base, (1, n_dates)), 1.0)
    
    # Compute expected demand
    expected_demand = footfall_series * med_rates * daily_med_mult
    
    # Add daily demand noise (avoiding pure determinism)
    demand_noise = np.random.normal(1.0, 0.15, size=(n_series, n_dates))
    demand_matrix = np.round(expected_demand * demand_noise).astype(int)
    demand_matrix = np.maximum(0, demand_matrix)
    
    # Establish target/max stock capacity based on average demand to ensure safety stock
    avg_monthly_demand = np.mean(expected_demand, axis=1) * 30
    target_stock = np.round(avg_monthly_demand * 1.5).astype(int)
    
    # Clamp to minimum safe stocks per medicine type
    min_targets = min_targets_series = np.tile(med_min_targets, n_phcs)
    target_stock = np.maximum(target_stock, min_targets)
    
    # Pre-generate monthly restock arrivals with variable delays (average 30 days interval)
    restock_day = np.zeros((n_series, n_dates), dtype=bool)
    for s in range(n_series):
        # Sample random supply delivery intervals
        intervals = np.random.normal(30, 6, size=40).astype(int)
        intervals = np.clip(intervals, 15, 45)
        arrival_days = np.cumsum(intervals)
        arrival_days = arrival_days[arrival_days < n_dates]
        restock_day[s, arrival_days] = True

    # =========================================================================
    # SECTION 5: Run Vectorized Sequential Stock Simulation
    # =========================================================================
    print("[5/7] Simulating daily inventory changes (730 days vectorized)...")
    
    opening_stock = np.zeros((n_series, n_dates), dtype=int)
    units_consumed = np.zeros((n_series, n_dates), dtype=int)
    closing_stock = np.zeros((n_series, n_dates), dtype=int)
    
    # Initialize starting stock (somewhere between 50% and 100% of capacity)
    current_stock = (np.random.uniform(0.5, 1.0, size=n_series) * target_stock).astype(int)
    
    # Vectorized step forward through time
    for t in range(n_dates):
        # Capture opening stock for day t
        opening_stock[:, t] = current_stock
        
        # Calculate received deliveries (if scheduled for today)
        is_restock_today = restock_day[:, t]
        needed_qty = np.maximum(0, target_stock - current_stock)
        
        # SCM fulfillment rate of 85-100% to reflect minor distribution shortfalls
        fulfillment_rate = np.random.uniform(0.85, 1.00, size=n_series)
        restock_qty = np.where(is_restock_today, np.round(needed_qty * fulfillment_rate).astype(int), 0)
        
        # Total units available for the day
        total_available = current_stock + restock_qty
        
        # Realized consumption (demand limited by stock availability)
        day_demand = demand_matrix[:, t]
        day_consumed = np.minimum(total_available, day_demand)
        units_consumed[:, t] = day_consumed
        
        # Calculate closing stock
        day_closing = total_available - day_consumed
        closing_stock[:, t] = day_closing
        
        # Carry over to the next morning
        current_stock = day_closing

    # =========================================================================
    # SECTION 6: Vectorized DataFrame Assembly
    # =========================================================================
    print("[6/7] Compiling 1.75 million rows to final structure...")
    
    # Build a 3D index grid: PHCs x Dates x Medicines
    idx_phc, idx_date, idx_med = np.meshgrid(
        np.arange(n_phcs), 
        np.arange(n_dates), 
        np.arange(n_meds), 
        indexing='ij'
    )
    
    # Flatten the grids to build long-format arrays
    idx_phc_flat = idx_phc.ravel()
    idx_date_flat = idx_date.ravel()
    idx_med_flat = idx_med.ravel()
    
    # Calculate mapping array to index the 2,400 series
    series_mapping = idx_phc_flat * n_meds + idx_med_flat
    
    # Assemble variables using fast 1D indexing
    final_data = {
        'date': dates_str[idx_date_flat],
        'phc_id': phc_ids[idx_phc_flat],
        'phc_name': phc_names[idx_phc_flat],
        'district': phc_districts[idx_phc_flat],
        'latitude': np.round(lats[idx_phc_flat], 6),
        'longitude': np.round(lons[idx_phc_flat], 6),
        'medicine': medicines[idx_med_flat],
        'opening_stock': opening_stock[series_mapping, idx_date_flat],
        'units_consumed': units_consumed[series_mapping, idx_date_flat],
        'closing_stock': closing_stock[series_mapping, idx_date_flat],
        'patient_footfall': patient_footfall_2d[idx_phc_flat, idx_date_flat],
        'total_beds': total_beds[idx_phc_flat],
        'occupied_beds': occupied_beds_2d[idx_phc_flat, idx_date_flat],
        'staff_sanctioned': staff_sanctioned[idx_phc_flat],
        'staff_present': staff_present_2d[idx_phc_flat, idx_date_flat]
    }
    
    df = pd.DataFrame(final_data)

    # =========================================================================
    # SECTION 7: Output and Performance Diagnostics
    # =========================================================================
    print("[7/7] Writing data to 'phc_data.csv'...")
    df.to_csv('phc_data.csv', index=False)
    
    end_time = time.time()
    execution_duration = end_time - start_time
    
    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Total Rows Generated : {len(df):,}")
    print(f"File Saved To        : phc_data.csv")
    print(f"Execution Time       : {execution_duration:.2f} seconds")
    print("="*50)
    
    print("\nDataset Preview:")
    print(df.head(10).to_string())

if __name__ == '__main__':
    main()
import numpy as np

def get_zenith_r0(los_r0, zenith_deg):
    """
    Given a line-of-sight r0 (LOS), compute the expected zenith r0.
    Atmosphere converts zenith r0 to LOS r0 via: r0_los = r0_zenith * (cos(zenith))^(3/5).
    Therefore: r0_zenith = r0_los / (cos(zenith))^(3/5)
    """
    return los_r0 / (np.cos(np.deg2rad(zenith_deg))**(3/5))

# Target LOS r0 in meters
los_r0_targets = [0.05, 0.08, 0.10, 0.15, 0.21]

# Constant altitudes across all configurations
altitude = [100, 1500, 5000, 10000, 15000]

# 3 different draws featuring different wind conditions and fractionalR0 vertical distributions
draws_config = [
    {   # Ground dominated
        'fractionalR0': [0.55, 0.20, 0.12, 0.08, 0.05],
        'windDirection': [20, 70, 120, 210, 300], 
        'windSpeed': [8, 12, 16, 11, 7]
    },
    {   # Smoother profile
        'fractionalR0': [0.45, 0.25, 0.16, 0.09, 0.05],
        'windDirection': [55, 110, 185, 250, 20], 
        'windSpeed': [9, 14, 18, 10, 6]
    },
    {   # Stronger high-altitude turbulence
        'fractionalR0': [0.35, 0.30, 0.20, 0.10, 0.05],
        'windDirection': [350, 40, 150, 230, 290], 
        'windSpeed': [11, 10, 15, 13, 8]
    }
]

# Fixed zenith
zenith = 30

atm_cases = {}

for case_idx, los_r0 in enumerate(los_r0_targets, start=1):
    case_key = f'atm{case_idx}'
    atm_cases[case_key] = {}
    
    r0_zenith = get_zenith_r0(los_r0, zenith)
    
    for draw_idx, wind_conf in enumerate(draws_config, start=1):
        draw_key = f'draw{draw_idx}'
        
        atm_cases[case_key][draw_key] = {
            'r0': r0_zenith,
            'L0': 25,
            'fractionalR0': wind_conf['fractionalR0'],
            'altitude': altitude,
            'windDirection': wind_conf['windDirection'],
            'windSpeed': wind_conf['windSpeed'],
            'zenith': zenith,
            'los_r0': los_r0  # for reference
        }
from SAOS.Atmosphere import Atmosphere

# ============================================================
# CASE 1: BAD seeing (r0 = 0.06)
# ============================================================

# Draw 1
atm = Atmosphere(r0=0.06,
                 L0=25,
                 fractionalR0=[0.54, 0.27, 0.10, 0.06, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[40, 85, 140, 220, 310],
                 windSpeed=[12, 16, 20, 14, 9],
                 telescope=est_tel,
                 zenith=45,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0=0.06,
                 L0=25,
                 fractionalR0=[0.54, 0.27, 0.10, 0.06, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[75, 130, 200, 260, 15],
                 windSpeed=[10, 18, 17, 12, 8],
                 telescope=est_tel,
                 zenith=45,
                 logger=test_logger.logger)

# Draw 3
atm = Atmosphere(r0=0.06,
                 L0=25,
                 fractionalR0=[0.54, 0.27, 0.10, 0.06, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[10, 60, 170, 280, 340],
                 windSpeed=[14, 11, 19, 15, 6],
                 telescope=est_tel,
                 zenith=45,
                 logger=test_logger.logger)


# ============================================================
# CASE 2: MEDIUM-LOW seeing (r0 = 0.10)
# ============================================================

# Draw 1
atm = Atmosphere(r0=0.10,
                 L0=25,
                 fractionalR0=[0.50, 0.24, 0.14, 0.07, 0.05],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[20, 70, 120, 210, 300],
                 windSpeed=[8, 12, 16, 11, 7],
                 telescope=est_tel,
                 zenith=30,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0=0.10,
                 L0=25,
                 fractionalR0=[0.50, 0.24, 0.14, 0.07, 0.05],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[55, 110, 185, 250, 20],
                 windSpeed=[9, 14, 18, 10, 6],
                 telescope=est_tel,
                 zenith=30,
                 logger=test_logger.logger)

# Draw 3
atm = Atmosphere(r0=0.10,
                 L0=25,
                 fractionalR0=[0.50, 0.24, 0.14, 0.07, 0.05],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[350, 40, 150, 230, 290],
                 windSpeed=[11, 10, 15, 13, 8],
                 telescope=est_tel,
                 zenith=30,
                 logger=test_logger.logger)


# ============================================================
# CASE 3: MEDIUM-GOOD seeing (r0 = 0.18)
# ============================================================

# Draw 1
atm = Atmosphere(r0=0.18,
                 L0=25,
                 fractionalR0=[0.42, 0.18, 0.20, 0.12, 0.08],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[0, 45, 135, 225, 315],
                 windSpeed=[7, 11, 14, 13, 9],
                 telescope=est_tel,
                 zenith=60,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0=0.18,
                 L0=25,
                 fractionalR0=[0.42, 0.18, 0.20, 0.12, 0.08],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[30, 80, 160, 240, 330],
                 windSpeed=[6, 10, 16, 12, 8],
                 telescope=est_tel,
                 zenith=60,
                 logger=test_logger.logger)

# Draw 3
atm = Atmosphere(r0=0.18,
                 L0=25,
                 fractionalR0=[0.42, 0.18, 0.20, 0.12, 0.08],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[70, 120, 200, 280, 25],
                 windSpeed=[8, 9, 15, 11, 10],
                 telescope=est_tel,
                 zenith=60,
                 logger=test_logger.logger)


# ============================================================
# CASE 4: GOOD seeing (r0 = 0.30)
# ============================================================

# Draw 1
atm = Atmosphere(r0=0.30,
                 L0=25,
                 fractionalR0=[0.30, 0.18, 0.22, 0.18, 0.12],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[25, 90, 180, 270, 340],
                 windSpeed=[5, 8, 12, 10, 7],
                 telescope=est_tel,
                 zenith=15,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0=0.30,
                 L0=25,
                 fractionalR0=[0.30, 0.18, 0.22, 0.18, 0.12],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[60, 140, 220, 300, 20],
                 windSpeed=[4, 9, 11, 8, 6],
                 telescope=est_tel,
                 zenith=15,
                 logger=test_logger.logger)

# Draw 3
atm = Atmosphere(r0=0.30,
                 L0=25,
                 fractionalR0=[0.30, 0.18, 0.22, 0.18, 0.12],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[350, 40, 120, 210, 290],
                 windSpeed=[6, 7, 10, 9, 5],
                 telescope=est_tel,
                 zenith=15,
                 logger=test_logger.logger)
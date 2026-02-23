from SAOS.Atmosphere import Atmosphere

# Case 1: r0 10 cm, z = 90, [0.7, 0.3], [0, 9000]

# Draw 1

atm = Atmosphere(r0 = 0.10,
                 L0= 25,
                 fractionalR0=[0.7, 0.3],
                 altitude=[0, 9000],
                 windDirection=[0, 20],
                 windSpeed=[10, 25],
                 telescope=est_tel,
                 zenith = 0,
                 logger=test_logger.logger)

# Draw 2

atm = Atmosphere(r0 = 0.10,
                 L0= 25,
                 fractionalR0=[0.7, 0.3],
                 altitude=[0, 9000],
                 windDirection=[0, 130],
                 windSpeed=[10, 20],
                 telescope=est_tel,
                 zenith = 0,
                 logger=test_logger.logger)

## Case 2: r0 11 cm, z = 60, [0.46, 0.20, 0.21, 0.07, 0.06], p = 20

# Draw 1
atm = Atmosphere(r0 = 0.11,
                 L0= 25,
                 fractionalR0=[0.46, 0.20, 0.21, 0.07, 0.06],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[45, 90, 180, 67, 2],
                 windSpeed=[10, 19, 14, 25, 31],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0 = 0.11,
                 L0= 25,
                 fractionalR0=[0.46, 0.20, 0.21, 0.07, 0.06],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[70, 110, 40, 13,145],
                 windSpeed=[7, 19, 14, 25, 31],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

## Case 3: r0 09 cm, z = 15, [0.53, 0.37, 0.05, 0.03, 0.02], p = 50

# Draw 1
atm = Atmosphere(r0 = 0.09,
                 L0= 25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[45, 90, 180, 67, 2],
                 windSpeed=[12, 4, 45, 18, 8],
                 telescope=est_tel,
                 zenith = 15,
                 logger=test_logger.logger)

# Draw 2
atm = Atmosphere(r0 = 0.09,
                 L0= 25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[20, 223, 345, 0, 110],
                 windSpeed=[12, 4, 45, 18, 8],
                 telescope=est_tel,
                 zenith = 15,
                 logger=test_logger.logger)

## Case 4: r0 10 cm, z = 45, [0.57, 0.30, 0.07, 0.03, 0.03], p = 50

# Draw 1
atm = Atmosphere(r0 = 0.10,
                 L0= 25,
                 fractionalR0=[0.57, 0.30, 0.07, 0.03, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[87, 43, 150, 240, 350],
                 windSpeed=[15, 10, 24, 19, 29],
                 telescope=est_tel,
                 zenith = 45,
                 logger=test_logger.logger)

# Draw 2

atm = Atmosphere(r0 = 0.10,
                 L0= 25,
                 fractionalR0=[0.57, 0.30, 0.07, 0.03, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[200, 100, 300, 50, 70],
                 windSpeed=[15, 10, 24, 19, 29],
                 telescope=est_tel,
                 zenith = 45,
                 logger=test_logger.logger)

## Case 5: r0 40 cm, z = 60, [0.29, 0.19, 0.20, 0.19, 0.14], p = 80

# Draw 1
atm = Atmosphere(r0 = 0.40,
                 L0= 25,
                 fractionalR0=[0.29, 0.19, 0.20, 0.19, 0.14],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[0, 45, 225, 315, 135],
                 windSpeed=[8, 19, 20, 17, 23],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

# Draw 2

atm = Atmosphere(r0 = 0.40,
                 L0= 25,
                 fractionalR0=[0.29, 0.19, 0.20, 0.19, 0.14],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[90, 180, 270, 335, 60],
                 windSpeed=[8, 19, 20, 17, 23],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

## Case 6: r0 06 cm, z = 45, [0.54, 0.33, 0.08, 0.03, 0.02], p = 20

# Draw 1
atm = Atmosphere(r0 = 0.06,
                 L0= 25,
                 fractionalR0=[0.54, 0.33, 0.08, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[76, 45, 90, 270, 10],
                 windSpeed=[14, 20, 40, 10, 45],
                 telescope=est_tel,
                 zenith = 45,
                 logger=test_logger.logger)

# Draw 2

atm = Atmosphere(r0 = 0.06,
                 L0= 25,
                 fractionalR0=[0.54, 0.33, 0.08, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[320, 20, 100, 210, 0],
                 windSpeed=[14, 20, 30, 10, 45],
                 telescope=est_tel,
                 zenith = 45,
                 logger=test_logger.logger)

## Case 7: r0 21 cm, z = 60, [0.48, 0.11, 0.22, 0.11, 0.09], p = 50

# Draw 1
atm = Atmosphere(r0 = 0.21,
                 L0= 25,
                 fractionalR0=[0.48, 0.11, 0.22, 0.11, 0.09],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[35, 60, 210, 50, 200],
                 windSpeed=[12, 4, 45, 18, 8],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

# Draw 2

atm = Atmosphere(r0 = 0.21,
                 L0= 25,
                 fractionalR0=[0.48, 0.11, 0.22, 0.11, 0.09],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[60, 260, 135, 75, 345],
                 windSpeed=[12, 4, 45, 18, 8],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)
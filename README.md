# AMSC664
Final code for AMSC664

The project: The end goal of this study is to enhance the bandwidth and power of gyro-amplifiers by utilizing a spatially periodic waveguide.   

These files are intended to demonstrate some of the simulations I have run through this semester, but I have not included the current files I am working on or any of the MAGY code.  

To run this code you need Python 3.9.  I tried to upgrade half way through the semester to 3.10, however there were two instances were the code I had used was deprecated.  The required packages are numpy, cmath, scipy, numba, and matplotlib.  Everything else is provided.

Both files quartic_solver_smooth.py and quartic_solver_corrugated.py require manual changes if a different scenario is desired. They can both be run right now with scenario 17, g = 0.76 and cutoff freq of 1.89*10**10.

These programs were both used to choose a magnetic field strength resulting in the highest gain for the voltage amplitude of the wave. Comparing both plots, it can be seen that the results did not provide enhanced bandwidth and the gain remained the same.

The program fqs.py was found on github, thanks to NKrvavica.  Most of the original derivations were done by T. Antonsen and the MAGY paper authors, “MAGY: A Time-Dependent Code for Simulation of Slow and Fast Microwave Sources’ , M. Botton, Thomas M. Antonsen, Jr., Baruch Levush, Khanh T. Nguyen, and Alexander N. Vlasov. IEEE Transactions on Plasma Science, VOL. 26, NO. 3, JUNE 1998.


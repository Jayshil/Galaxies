import numpy as np
import astropy.constants as con
import astropy.units as u

name = np.array(['M81', 'M87', 'M101'])
distances = np.array([3.6, 16.4, 6.4])*u.Mpc
ang_eff_rad = np.array([0.473, 99.084, 0.056])*u.arcsec
ang_sc_rad = np.array([108.849, 0, 112.793])*u.arcsec

ang_eff_rad_radians = ang_eff_rad.to('', equivalencies=u.dimensionless_angles())
ang_sc_rad_radians = ang_sc_rad.to('', equivalencies=u.dimensionless_angles())

size_eff = (ang_eff_rad_radians*distances).to(u.kpc)
size_scale = (ang_sc_rad_radians*distances).to(u.kpc)

f1 = open('various_radii.dat', 'w')
f1.write('#Name \t Effective radius \t Scale Radius\n')
f1.write('#\t\t (in kpc)\t (in kpc)\n')

for i in range(len(name)):
	f1.write(name[i] + '\t' + str(size_eff[i]) + '\t' + str(size_scale[i]) + '\n')

f1.close()

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit as cft
from matplotlib import gridspec as gd

#----------------------------------
#----------Here's data-------------
#----------------------------------
filename = input('Enter the name of data file: ')
asc, sb, sbe = np.loadtxt(filename + '.dat', usecols=(0,1,2), unpack=True)

#----------Fake points
xx = np.linspace(np.min(asc), np.max(asc), 1000)

#-----------------------------------------------------------------
#
#--------------------Fitting using Scipy--------------------------
#
#-----------------------------------------------------------------

print('-----------------------------------------')
print('                                         ')
print('             Scipy Fitting               ')
print('                                         ')
print('-----------------------------------------')

def sersic(r_arc, A, B):
	ir = A*np.exp(-1*B*r_arc)
	return ir

popt, pcov = cft(sersic, asc, sb, sigma=sbe, absolute_sigma=True, p0 = [0,0], bounds = (-np.inf, np.inf))
print(popt)
perr = np.sqrt(np.diag(pcov))

residuals1 = sb - sersic(asc, *popt)

bb = popt[1]
bbe = perr[1]

rd_inv = np.random.normal(bb, bbe, 10000)
rd = rd_inv**-1

print('Effective radius: ' + str(np.median(rd)) + '+/-' + str(np.std(rd)))
print('                                         ')
fig2 = plt.figure(figsize = (8,10))
gs2 = gd.GridSpec(2, 1, height_ratios = [4,1])

ax2 = plt.subplot(gs2[0])

ax2.errorbar(asc, sb, yerr=sbe, color='orangered', fmt='.', alpha=0.5, label='Data')
ax2.plot(xx, sersic(xx, *popt), color='orangered', label='Best fit Sersic profile')
ax2.grid()
ax2.legend(loc='best')
plt.subplots_adjust(hspace = 0.15)
plt.ylabel('Surface brightness')
plt.title('Exponential profile for M101 Galaxy')


ax22 = plt.subplot(gs2[1], sharex = ax2)
ax22.errorbar(asc, residuals1, fmt='.', color='orangered', alpha=0.5)
ax22.grid()
plt.ylabel('Residuals')
plt.xlabel('Radius (in arcsec)')
#plt.show()
plt.savefig(filename + '_scipy.png')
plt.close(fig2)

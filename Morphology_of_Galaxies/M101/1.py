import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.functional_models import Sersic1D as src
from astropy.modeling import models, fitting
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
#--------------------Fitting using Astropy------------------------
#
#-----------------------------------------------------------------

print('-----------------------------------------')
print('                                         ')
print('            Astropy Fitting              ')
print('                                         ')
print('-----------------------------------------')
s1 = src()
fit_s1 = fitting.SimplexLSQFitter()
s1_fit = fit_s1(s1, x=asc, y=sb, maxiter=10000, weights=1/sbe)
print('                                         ')
print(s1_fit)
print('                                         ')

residuals = sb - s1_fit(asc)
#----------For figure
fig1 = plt.figure(figsize = (8,10))
gs1 = gd.GridSpec(2, 1, height_ratios = [4,1])

ax1 = plt.subplot(gs1[0])

ax1.errorbar(asc, sb, yerr=sbe, color='orangered', fmt='.', alpha=0.5, label='Data')
ax1.plot(xx, s1_fit(xx), color='orangered', label='Best fit Sersic profile')
ax1.grid()
plt.ylabel('Surface brightness')
plt.title('Sersic profile for M101 Galaxy')
plt.legend(loc='best')

ax11 = plt.subplot(gs1[1], sharex = ax1)
ax11.errorbar(asc, residuals, fmt='.', color='orangered', alpha=0.5)
ax11.grid()
plt.ylabel('Residuals')
plt.xlabel('Radius (in arcsec)')

plt.subplots_adjust(hspace = 0.15)

plt.savefig(filename + '_astropy.png')
plt.close(fig1)
#-----------Calculating BIC
rss_a = 0
for i in range(len(asc)):
	rss1 = (sb[i] - s1_fit(asc[i]))**2
	rss_a = rss_a + rss1

bic_a = len(asc)*np.log(rss_a)
print('BIC for astropy fitting is ' + str(bic_a))
print('                                         ')
print('-----------------------------------------')

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

def sersic(r_arc, ie, bn, re, n):
	r11 = (r_arc/re)**(1/n)
	b1 = -bn*(r11-1)
	ir = ie*np.exp(b1)
	return ir

popt, pcov = cft(sersic, asc, sb, sigma=sbe, absolute_sigma=True, bounds = (-np.inf, np.inf))
perr = np.sqrt(np.diag(pcov))
print('Effective intensity: ' + str(popt[0]) + '+/-' + str(perr[0]))
print('Effective radius: ' + str(popt[2]) + '+/-' + str(perr[2]))
print('Sersic index: ' + str(popt[-1]) + '+/-' + str(perr[-1]))
print('                                         ')

residuals1 = sb - sersic(asc, *popt)

#-------------For figure
fig2 = plt.figure(figsize = (8,10))
gs2 = gd.GridSpec(2, 1, height_ratios = [4,1])

ax2 = plt.subplot(gs2[0])

ax2.errorbar(asc, sb, yerr=sbe, color='orangered', fmt='.', alpha=0.5, label='Data')
ax2.plot(xx, sersic(xx, *popt), color='orangered', label='Best fit Sersic profile')
ax2.grid()
plt.ylabel('Surface brightness')
plt.title('Sersic profile for M101 Galaxy')
plt.subplots_adjust(hspace = 0.15)
plt.legend(loc='best')

ax22 = plt.subplot(gs2[1], sharex = ax2)
ax22.errorbar(asc, residuals1, fmt='.', color='orangered', alpha=0.5)
ax22.grid()
plt.ylabel('Residuals')
plt.xlabel('Radius (in arcsec)')


rss_s = 0
for i in range(len(asc)):
	rss1 = (sb[i] - sersic(asc[i], *popt))**2
	rss_s = rss_s + rss1

bic_s = len(asc)*np.log(rss_s)
print('BIC for scipy fitting is ' + str(bic_s))
print('                                         ')
print('-----------------------------------------')
plt.savefig(filename + '_scipy.png')
plt.close(fig2)

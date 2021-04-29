import numpy as np
import pickle
import sys
import pathlib
import galsim
dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
dir_binnedFit = dir_repo + '/BinnedFit'
sys.path.append(dir_KLens)
sys.path.append(dir_binnedFit)

from tfCube2 import gen_grid

from utils import spin_rotation, sini_rotation, PA_rotation

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy import constants
from astropy.io import fits
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

from spec2D import Spec2D

class ParametersTNG:
    lineLambda0 = {'OIIa': 372.7092, 'OIIb': 372.9875,
                   'OIIIa': 496.0295, 'OIIIb': 500.8240,
                   'Halpha': 656.461}  # [unit: nm]

    def __init__(self, **kwargs):
        '''The parameter object to store parameters to interact with TNGmock & Sky
            Kwargs:
                lambda_cen: real
                    defines the central value of lambdaGrid
                line_species: str
                    available values are : 'OII', 'OIII', 'Halpha'
                    If line_species is passed as a kwarg, self.fid['lambda_cen'] would be overwritten by the redshifted center of the given line_species. i.e. if line_species='Halpha', lambda_cen = (1+redshift)*656.461 .

                Other available keyward arguments are defined in self.base.keys() .
                See self.set_defaults() method for detail.

            Examples:
                eg0: simply use the default parameter values
                    >> Pars = ParametersTNG()
                eg1: pass kwargs to be changed w.r.t. the default parameter dict (self.par0).
                    >> Pars = ParametersTNG(sini=0.8, g1=0.05, g2=0.05)
                eg2: pass a dictionary of parameters 
                    >> par_in = {'sini':0.8, 'g1':0.05, 'g2':0.05}
                    >> Pars = ParametersTNG(**par_in)
        
        '''
        # 1. init the default parameters of TNGcube
        self.base = self.set_defaults()

        # 2. update the fidicual parameters based on the user input
        self.fid = {**self.base, **kwargs}

        if 'line_species' in kwargs: 
            if kwargs['line_species'] in ['OII', 'OIII']: # doubles
                line_doubles = [kwargs['line_species']+'a', kwargs['line_species']+'b']
                lambda0 = np.mean([ParametersTNG.lineLambda0[key] for key in line_doubles])
                self.fid['lambda_cen'] = (1.+self.fid['redshift']) * lambda0
            else: # singlets
                self.fid['lambda_cen'] = (1.+self.fid['redshift'])*ParametersTNG.lineLambda0[kwargs['line_species']]
        
        self.add_cosmoRedshift()
        self.define_grids()

    def set_defaults(self):
        '''Initiate default parameters'''

        base = {}
        base['redshift'] = 0.4
        base['sini'] = 0.5
        base['theta_int'] = 0.
        base['aspect'] = 0.2

        base['spinR'] = [0., 0., -1.]
        base['g1'] = 0.
        base['g2'] = 0.

        # grid parameters
        base['ngrid'] = 256
        base['image_size'] = 128
        base['pixScale'] = 0.1185

        base['nm_per_pixel'] = 0.033

        # set default lambda_cen = halpha at redshift
        base['lambda_cen'] = (1. + base['redshift'])*656.461

        # observation parameters
        base['sigma_thermal'] = 16.  # [unit: km/s]
        base['psfFWHM'] = 0.5
        base['psf_g1'] = 0.
        base['psf_g2'] = 0.
        base['Resolution'] = 5000.  # for Keck

        base['slitWidth'] = 0.12

        # line intensity
        base['expTime'] = 30.*60.              # [unit: sec]
        base['area'] = 3.14 * (1000./2.)**2    # telescope area [unit: cm2]
        base['throughput'] = 0.29
        # peak of the reference SDSS line intersity (for a given line_species)
        base['ref_SDSS_peakI'] = 3.*1e-17     # [unit: erg/s/Angstrom/cm2]

        base['read_noise'] = 3.0
        
        return base
    
    @property
    def integrated_peakI(self):
        '''integrated peak line intensity over telescope area and explosure time
                self.integrated_peakI : [unit: photons/nm]
        '''
        ref_peakI = self.fid['ref_SDSS_peakI'] * u.erg/u.second/u.Angstrom/u.cm**2
        ref_peakI = ref_peakI.to(u.erg/u.second/u.nm/u.cm**2)

        int_peakI = ref_peakI * (self.fid['area']*u.cm**2) * (self.fid['expTime']*u.second)

        energy_per_photon = constants.h*constants.c/(self.fid['lambda_cen']*u.nm) / u.photon
        energy_per_photon = energy_per_photon.to(u.erg/u.photon)

        return int_peakI/energy_per_photon
    
    def define_grids(self):

        # grid parameters
        self.extent = self.fid['image_size'] * self.fid['pixScale']
        self.subGridPixScale = self.extent/self.fid['ngrid']
        self.lambda_min = self.fid['lambda_cen'] - 2.
        self.lambda_max = self.fid['lambda_cen'] + 2.

        self.spaceGrid = gen_grid(cen=0., pixScale=self.subGridPixScale, Ngrid=self.fid['ngrid'])
        Ngrid_l = int((self.lambda_max-self.lambda_min)/self.fid['nm_per_pixel'])+1
        self.lambdaGrid = gen_grid(cen=self.fid['lambda_cen'], pixScale=self.fid['nm_per_pixel'], Ngrid=Ngrid_l)

        self.spaceGrid_edg = np.append(self.spaceGrid-self.subGridPixScale/2., self.spaceGrid[-1]+self.subGridPixScale/2.)
        self.lambdaGrid_edg = np.append(self.lambdaGrid-self.fid['nm_per_pixel']/2., self.lambdaGrid[-1]+self.fid['nm_per_pixel']/2.)
        
    def add_cosmoRedshift(self):

        # wavelength of lines being redshifted by cosmic expansion
        self.lineLambdaC = {key: (1.+self.fid['redshift'])*self.lineLambda0[key]
                            for key in self.lineLambda0.keys()}
        
        # comiving distance
        self.Dc = cosmo.comoving_distance(z=self.fid['redshift']).to(u.kpc).value * cosmo.h # [unit: ckpc/h]
    
    def __getitem__(self, key):
        '''allowing calling self[key] to access self.fid[key]'''
        return self.fid[key]
    
    def __getattr__(self, key):
        '''allowing calling self.key to access self.fid['key']'''
        return self.fid[key]
        

class Subhalo:
    def __init__(self, info, snap):
        '''
            Args: 
                subhaloInfo: overall properties of the subhalo
                    e.g. info = { 'snap': 75, 'id': 46,
                                  'mass': 5.62947, 'stellarphotometrics_r': -19.0875,
                                  'vmax': 94.5006, 'vmaxrad': 9.9715, 'mass_log_msun': 10.919622316768256,
                                  'cm': [8364.92, 24582.0, 21766.5], 'pos': [8364.78, 24583.6, 21768.1],
                                  'spin': [-96.1246, -11.3787, -158.723], 'vel': [-865.215, 17.2853, -195.896]}
        '''
        self.info = info
        self.snap = snap
    
    def vec3Dtransform(self, M, vec):
        '''Coordinate transformation on vector given the input matrix operator
            Args:
                M: ndarray 3x3
                    transformation matrix
                vec: ndarray Nx3
                    Vector coordinates before applying M
        '''
        vecM = np.dot(M, vec.T).T
        return vecM
    
    def rotation(self, R):
        '''Perform cooordination transformation on all vector quantities stored in the Subhalo obj. given the matrix operator R.
            Args:
                R: 3x3 array
                    total rotation operator : R = R_pa@R_sini@R_spin
        '''
        
        #for key in ['cm', 'pos', 'spin', 'vel']:
        #    self.info[key] = self.vec3Dtransform(M=R, vec=self.info[key])

        for ptlType in ['gas', 'stars']:
            for key in ['pos', 'vel']:
                self.snap[ptlType][key] = self.vec3Dtransform(M=R, vec=self.snap[ptlType][key])
    
    def shear(self, g1, g2):
        '''Add shear on vector quantities in the x, and y directions. (The L.O.S direction is not affected.)
            L: the lensing shear matrix
        '''
        L = np.array([[1.+g1, g2, 0.],[g2, 1.-g1, 0.], [0., 0., 1.]])
        
        #for key in ['cm', 'pos', 'spin', 'vel']:
        #    self.info[key]=self.vec3Dtransform(M=L, vec=self.info[key])

        for ptlType in ['gas', 'stars']:
            for key in ['pos', 'vel']:
                self.snap[ptlType][key]=self.vec3Dtransform(M=L, vec=self.snap[ptlType][key])

    def recenter_pos(self, dx):
        '''Minor position adjust to make the center of the subhalo more closer to [0, 0, 0].
            Args:
                dx: [unit: ckpc/h]
                    e.g [0.1, 0.1, 0.]
        '''
        for ptlType in ['gas', 'stars']:
            for j in range(3):
                self.snap[ptlType]['pos'][:, j] += dx[j]
    
    def recenter_vel(self, dv):
        '''Minor velocity adjust to make the center of the subhalo more closer to [0, 0, 0].
            Args:
                dv: [unit: km/s]
                    e.g [0., 0., 0.5]
        '''
        for ptlType in ['gas', 'stars']:
            for j in range(3):
                self.snap[ptlType]['vel'][:, j] += dv[j]

    
class TNGmock:

    def __init__(self, pars, subhalo, par_meta=None):

        if isinstance(pars, dict):
            self.Pars = ParametersTNG(**pars)
        elif isinstance(pars, ParametersTNG):
            self.Pars = pars
        else:
            raise TypeError("Argument, pars, needs to be a dictionary or an instance of the ParametersTNG class.")
        
        if par_meta is not None:
            self.par_meta = par_meta

        self.subhalo = subhalo
        self.reset_subhalo_coordinates()

        self.set_constants()

        self.line_species = self.find_lines_within_lambdaGrid()
        
        if len(self.line_species)==1 :
            self.lambda0 = self.Pars.lineLambda0[self.line_species[0]]
        else :
            self._lambda0s = [self.Pars.lineLambda0[self.line_species[i]] for i in range(len(self.line_species))]
            self.lambda0 = np.mean(self._lambda0s)
        
        self.z = self.Pars.fid['redshift']
    
    def set_constants(self):   
        self.c_kms = 2.99792458e5
        self.radian2arcsec = 206264.806247096

    def find_lines_within_lambdaGrid(self):
        '''Find line_species that are within the range covered in lambdaGrid'''
        
        def is_between(lambda_in):
            return self.Pars.lambdaGrid[0] < lambda_in < self.Pars.lambdaGrid[-1]
        
        line_species = [key for key, val in filter(lambda item: is_between(item[1]), self.Pars.lineLambdaC.items())]

        return line_species   
  
    def vLOS_to_lambda(self, v_z, lineType):
        '''computed the redshifted lambda given v_z
            Args:
               v_z: 1d array
                    L.O.S. velocity in km/s
                    v_z's sign is defined by the right-hand rule. For a face-on cooridinate system x, y viewd by an observer, 
                    v_z > 0 is the out-paper direction. -> blue-shifted, smaller lambda
                    v_z < 0 is the in-paper diection. -> red-shifted, larger lambda

               lineType: string
                    e.g. lineType = 'Halpha'
            Returns:
                lambdaLOS: 1d array
                    redshifted wavelength along the L.O.S.
        '''

        lambdaLOS = (1.-v_z/self.c_kms)*self.Pars.lineLambdaC[lineType]

        return lambdaLOS

    def _massCube_i(self, ptlType, lineType, weights='mass'):
        '''Generate the massCube for given ptlType, lineType

            Returns:
                massCube: 3D array (x, y, lambda) [unit: Msun/h / pix^3]
            
            Note:
                When calling np.histogramdd to make 3D histgram, input ptl cooridnates need to be: (y, x, lambda), 
                in order to produce consistent mesh definition as tfCube. 
                This way, tfCube2.modelCube and massCube can adopt the same plotting routing 
                i.e. imshow(np.sum(modelCube, axis = 2), origin='lower').

                If the input ptl cooridnates are ordered by (x, y, lambda), then the ploting routings for both 3D cubes would differ:
                    imshow(np.sum(modelCube , axis = 2), origin='lower')    # for tfCube.modelCube
                    imshow(np.sum(massArray.T, axis = 2), origin='lower')    # for TNGCube.massCube
        '''
        
        x_arcsec = self.subhalo.snap[ptlType]['pos'][:, 0]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]
        y_arcsec = self.subhalo.snap[ptlType]['pos'][:, 1]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]
        lambdaLOS = self.vLOS_to_lambda(self.subhalo.snap[ptlType]['vel'][:, 2], lineType=lineType) #[unit: nm]

        if weights=='mass':
            mass = self.subhalo.snap[ptlType]['mass']*1.e10  # [unit: Msun/h]
            massCube, _ = np.histogramdd((y_arcsec, x_arcsec, lambdaLOS), 
                                         bins=(self.Pars.spaceGrid_edg, self.Pars.spaceGrid_edg, self.Pars.lambdaGrid_edg), weights=mass)
        elif weights == 'SFR':
            massCube, _ = np.histogramdd((y_arcsec, x_arcsec, lambdaLOS),
                                         bins=(self.Pars.spaceGrid_edg, self.Pars.spaceGrid_edg, self.Pars.lambdaGrid_edg), weights=self.subhalo.snap['gas']['SFR'])
        else:
            raise ValueError('weights needs to be either \'mass\' or \'SFR\'')

        return massCube
    
    def gen_massCube(self, ptlTypes, lineTypes, weights='mass'):
        '''Generate the the sum of massCube for all input ptlTypes and lineTypes

            Args:
                ptlTypes: list
                    e.g. ptlTypes = ['gas', 'stars']
                lineTypes: list, line species that fall within the range of lambdaGrid
                    lineTypes = self.line_species
                weights :  'mass' or 'SFR'
                    the weighting factor to pass into np.histogramdd
                    for 'SFR' weights, ptlType can only be ['gas']

            Returns:
                massCube: 3D array (x, y, lambda)
                    [unit: Msun/h /x_grid/y_grid/lambda_grid
        '''

        massCube = np.zeros([self.Pars.spaceGrid.size, self.Pars.spaceGrid.size, self.Pars.lambdaGrid.size])

        for lineType in lineTypes:
            for ptlType in ptlTypes:
                massCube += self._massCube_i(ptlType, lineType, weights)

        return massCube
    
    def gen_imageArray(self, band='r', weights='photometry'):
        '''Generate galaxy image array (2D)
            Args:
                band : str
                    available bands : U, B, V, K, g, r, i, z
                weights : 'photometry' or 'mass'
                    if weights == 'mass', use ptl mass to build 2D image histogram
                    if weights == 'photometry', use the stellar ptl photometry (in given band) as weights to build 2D image.
        '''
        x_arcsec = self.subhalo.snap['stars']['pos'][:, 0]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]
        y_arcsec = self.subhalo.snap['stars']['pos'][:, 1]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]

        if weights == 'mass':
            imageArr, _ = np.histogramdd((y_arcsec, x_arcsec),
                            bins=(self.Pars.spaceGrid_edg, self.Pars.spaceGrid_edg), 
                            weights=self.subhalo.snap['stars']['mass']*1.e10, density=True)

        if weights == 'photometry':
            IDband = 'UBVKgriz'.index(band)
            mAB = self.subhalo.snap['stars']['GFM_StellarPhotometrics'][:, IDband] # [unit: AB magnitude]
            intensity = 10**((mAB+48.60)/(-2.5))                                   # [unit: erg/s 1/Hz 1/cm^2]
            imageArr, _ = np.histogramdd((y_arcsec, x_arcsec),
                            bins=(self.Pars.spaceGrid_edg, self.Pars.spaceGrid_edg), 
                            weights=intensity, density=True)
        return imageArr
        
    def mass_to_light(self, massCube, MLratio=4.e-6):
        '''turn the unit of massCube (Msun/h /pix^3) to photonCube wiht unit Nphotons/pix^3.
            Args:
                MLratio: real
                    mass to light ratio
            
            Note:
                Currently simply set the default MLratio roughly at 4.e-6 by letting the photonCube have a right order.
                    np.sum(TF.modelCube) = 56673.481175424145
                    np.sum(massCube) = 13366719585.1875
                    MLratio = 56673.481175424145/13366719585.1875 ~ 4.0e-06
                In the future this function can be use to acount for optical transparency. 
        '''

        photonCube = massCube * MLratio
        specCube = SpecCube(photonCube, self.Pars.spaceGrid, self.Pars.lambdaGrid)

        return specCube
        
    def flux_renorm(self, specCube):
        '''Perform flux re-normalization for photonCube such that the integrated fiber spectrum is consistent with the given SDSS fiber spectrum set in self.Pars['ref_SDSS_peakI']*expTime*area
        '''
        spec1D = Fiber(specCube).get_spectrum(fiberR=1.5)  # SDSS fiber Radius=1.5 arcsec
        Nphoton_peak = spec1D.max()*u.photon/u.nm
        renorm_factor = self.Pars.integrated_peakI/Nphoton_peak
        specCube.array *= renorm_factor.value

        return specCube
    
    def add_sky_noise(self, specCube, sky):

        for k in range(self.Pars.lambdaGrid.size):
            thisIm = galsim.Image(np.ascontiguousarray(specCube.array[:, :, k]), scale=specCube.pixScale)
            noise = galsim.CCDNoise(sky_level = sky.spec1D_arr[k], read_noise = self.Pars.read_noise)
            noiseImage = thisIm.copy()
            noiseImage.addNoise(noise)

            specCube.array[:, :, k] = noiseImage.array
        
        return specCube
    
    def cal_sigma_thermal_nm(self, sigma_thermal_kms):
        '''Compute sigma_thermal in unit the same as lambdaGrid, given sigma_thermal in [km/s]'''
        return self.Pars.lambda_cen*sigma_thermal_kms/self.c_kms

    def reset_subhalo_coordinates(self):
        ''' Rotate and shear coordinates of subhalo particles'''
        # 1. compute total rotation matrix, Rtot
        R_spin = spin_rotation(spin0=self.subhalo.info['spin'], spinR=self.Pars.spinR)
        R_sini = sini_rotation(sini=self.Pars.sini)
        R_pa = PA_rotation(theta=self.Pars.theta_int)
        Rtot = R_pa@R_sini@R_spin

        # 2. Perform rotation to subhalo
        self.subhalo.rotation(Rtot)

        # 2.1 Perform additional adjustment to subhalo (if par_meta is set)
        if self.par_meta is not None:
            if self.par_meta['theta'] is not None:
                Rth = PA_rotation(theta=self.par_meta['theta'])
                self.subhalo.rotation(Rth)
            if self.par_meta['dx'] is not None:
                self.subhalo.recenter_pos(dx=self.par_meta['dx'])
            if self.par_meta['dv'] is not None:
                self.subhalo.recenter_vel(dv=self.par_meta['dv'])
        
        # 2.2 add Shear to subhalo
        self.subhalo.shear(g1=self.Pars.g1, g2=self.Pars.g2)

    def gen_mock_image(self, weights='photometry', band='r', noise_mode=0):
        '''Generate mock image
            Three options to generate mock image:
            - Option1 : image based on stacking specCube along lambdaGrid direction
                >> self.gen_mock_image(weight='line')
            - Option2 : image based on stellar particle photometry
                >> self.gen_mock_image(weight='photometry', band='r')
            - Option3 : image based on stellar particle mass
                >> self.gen_mock_image(weight='mass')

            Args:
                weights : weights for each image pixel
                    weights = 'line', 'photometry', 'mass' 

            develop log:
                The computation when weights == 'line' is slow... it requires generating noiseless specCube first.
                specCube0 is likely to be computed twice throughout the operation of this code...
                Better reorganization of the code would be needed.
        '''

        if weights == 'photometry':
            imageArr = self.gen_imageArray(band=band, weights='photometry')
            self.image = Image(imageArr, self.Pars.spaceGrid)
        elif weights == 'mass':
            imageArr = self.gen_imageArray(weights='mass')
            self.image = Image(imageArr, self.Pars.spaceGrid)
        elif weights == 'line':
            specCube0 = self.gen_mock_specCube(noise_mode=0)
            self.image = Image(specCube0)
        else:
            raise ValueError("Invalid weights argument. weights = \'photometry\', \'mass\', \'intensity\' ")

        # add psf to image
        self.image.add_psf(psfFWHM=self.Pars.psfFWHM,
                           psf_g1=self.Pars.psf_g1, psf_g2=self.Pars.psf_g2)
        
        # compute noise given SNR or add noise to self.image.array
        if noise_mode == 1:
            self.image.array_var = self.image.gen_image_variance(signal_to_noise=100., add_noise=True)
        else:
            self.image.array_var = self.image.gen_image_variance(signal_to_noise=100., add_noise=False)

        return self.image
        
    def gen_mock_specCube(self, noise_mode=0):

        # 1. generate specCube
        #massCube = self.gen_massCube(ptlTypes=['gas', 'stars'], lineTypes=self.line_species, weights='mass')
        massCube = self.gen_massCube(ptlTypes=['gas'], lineTypes=self.line_species, weights='SFR')
        self.specCube = self.mass_to_light(massCube)

        # 2. add psf for each plan at lambdaGrid[i]
        self.specCube.add_psf(psfFWHM=self.Pars.psfFWHM,
                              psf_g1=self.Pars.psf_g1, psf_g2=self.Pars.psf_g2)

        # 3 smooth spectrum
        # thermal part
        self.sigma_thermal_nm = self.cal_sigma_thermal_nm(
            sigma_thermal_kms=self.Pars.sigma_thermal)
        # spectral resoultion part
        self.sigma_resolution_nm = self.Pars.lambda_cen / self.Pars.Resolution
        self.sigma_tot = np.sqrt(self.sigma_thermal_nm**2 + self.sigma_resolution_nm**2)

        # smoothing with quick approximated way
        self.specCube.add_spec_sigma_approx(sigma=self.sigma_tot)
        #self.specCube.add_spec_sigma(resolution=self.Pars.Resolution, sigma_thermal_nm=self.sigma_resolution_nm) # smoothing with detailed way

        # 4. flux renorm
        self.specCube = self.flux_renorm(self.specCube)

        # 5. compute sky noise
        self.sky = Sky(self.Pars)

        # 6. add sky noise to specCube if noise_mode is 1.
        if noise_mode == 1:
            self.specCube = self.add_sky_noise(self.specCube, self.sky)

        return self.specCube

    
    def gen_mock_data(self, noise_mode=0):
        '''generate mock data_info dict.
            noise_mode = 1 : return noise data
            noise_mode = 0 : return noiseless data
        '''

        self.specCube = self.gen_mock_specCube(noise_mode=noise_mode)
        self.image = self.gen_mock_image(weights='photometry', band='r', noise_mode=noise_mode)

        spectra = Slit(self.specCube, slitWidth=self.Pars.slitWidth).get_spectra(
            slitAngles=self.Pars.slitAngles)

        dataInfo = {    'spec_variance': self.sky.spec2D_arr,
                        'image_variance': self.image.gen_image_variance(signal_to_noise=100),
                        'par_fid': self.Pars.fid,
                        'flux_norm': np.sum(self.image.array)   }
        
        if len(self.line_species)==1 :  # singlet
            dataInfo['line_species'] = self.line_species[0]
        else:                           # doublets
            dataInfo['line_species'] = self.line_species[0][:-1]
        
        dataInfo['image'] = self.image
        dataInfo['spec'] = spectra

        for j in range(len(dataInfo['spec'])):
            dataInfo['spec'][j] = Spec2D(array=dataInfo['spec'][j], array_var=dataInfo['spec_variance'],spaceGrid=self.specCube.spaceGrid, lambdaGrid=self.specCube.lambdaGrid, line_species=dataInfo['line_species'], z=self.z, auto_cut=False)

        dataInfo['par_fid']['vcirc'] = self.subhalo.info['vmax']
        dataInfo['par_fid']['r_hl_image'] = 0.5
        dataInfo['par_fid']['vscale'] = 0.5
        dataInfo['par_fid']['r_0'] = 0.0
        dataInfo['par_fid']['v_0'] = 0.0
        dataInfo['par_fid']['flux'] = dataInfo['flux_norm']
        dataInfo['par_fid']['subGridPixScale'] = self.specCube.pixScale
        dataInfo['par_fid']['ngrid'] = self.image.ngrid
        dataInfo['lambdaGrid'] = self.specCube.lambdaGrid
        dataInfo['spaceGrid'] = self.specCube.spaceGrid
        dataInfo['lambda0'] = self.lambda0

        return dataInfo


class Sky:
    def __init__(self, pars, skyfile=dir_KLens+'/data/Simulation/skytable.fits'):

        if isinstance(pars, dict):
            self.Pars = ParametersTNG(**pars)
        elif isinstance(pars, ParametersTNG):
            self.Pars = pars
        else:
            raise TypeError("Argument, pars, needs to be a dictionary or an instance of the ParametersTNG class.")
        
        self.skyTemplate = fits.getdata(skyfile)

    @property
    def spec1D_arr(self):
        '''
            raw sky flux in the file is: photon/s/m2/micron/arcsec2
            convert skySpec1D unit to photon/s/cm2/nm/arcsec2
            and to photons (/1 lambda_pixel/1 x_pixel/1 y_pixel), after integration
        '''
        spec = np.interp(self.Pars.lambdaGrid, self.skyTemplate['lam']*1000., self.skyTemplate['flux'])
        spec /= 1.0e7  # 1/1000 for micron <-> nm ; 1/10000 for m2 <-> cm2
        spec *= self.Pars.expTime*self.Pars.area*self.Pars.throughput * \
            self.Pars.subGridPixScale**2*self.Pars.nm_per_pixel

        return spec
    
    @property
    def skyCube(self):

        skyArray = np.empty([self.Pars.ngrid, self.Pars.ngrid, self.Pars.lambdaGrid.size])
        skyArray[:, :, :] = self.spec1D_arr[None, None, :]

        return SpecCube(skyArray, self.Pars.spaceGrid, self.Pars.lambdaGrid)
    
    @property
    def spec2D_arr(self):
        spec = Slit(self.skyCube, slitWidth=self.Pars.slitWidth).get_spectra(
            slitAngles=[0.])[0]
        return spec

        
class Image:
    '''The image data class
        Two ways to construct an Image:
        > Image(array2D, spaceGrid)
        > Image(SpecCube)

        kwargs:
            array_var : image_variance for the image array
            signal_to_noise : 
                When this keyword is set, the code would call galsim.addNoiseSNR to generate the corresponding array_var for the image. 
    '''
    def __init__(self, *args, **kwargs):

        if len(args) == 1:
            if isinstance(args[0], SpecCube):
                self.array = np.sum(args[0].array, axis=2)
                self.spaceGrid = args[0].spaceGrid
            else:
                raise TypeError('Input arguemnt needs to be a SpecCube obj (if only 1 argument is passed).')
        elif len(args) == 2:
            if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                self.array = args[0]
                self.spaceGrid = args[1]
            else:
                raise TypeError('Input arguemnts need to be a 2D and a 1D np.array (when 2 arguments are passed).')
        
        if 'array_var' in kwargs:
            self.array_var = kwargs.pop('array_var')
        
        if 'signal_to_noise' in kwargs:
            self.signal_to_noise = kwargs.pop('signal_to_noise')
            self.array_var = self.gen_image_variance(signal_to_noise=self.signal_to_noise)
    
    @property
    def pixScale(self):
        return self.spaceGrid[2]-self.spaceGrid[1]

    @property
    def ngrid(self):
        return len(self.spaceGrid)
    
    def cutout(self, xlim=[-2.5, 2.5]):
        '''return a smaller subImage given the xlim range'''
        id_x = np.where((self.spaceGrid >= xlim[0]) & (self.spaceGrid <= xlim[1]))[0]

        return Image(self.array[id_x, :][:, id_x], self.spaceGrid[id_x])
    
    def rebin(self, shape, operation='sum'):
        '''rebin self.array into the given shape'''
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        
        sh = shape[0], self.array.shape[0]//shape[0], shape[1], self.array.shape[1]//shape[1]

        if operation == 'sum':
            new_image = self.array.reshape(sh).sum(3).sum(1)
        else:
            new_image = self.array.reshape(sh).mean(3).mean(1)
        
        new_spaceGrid = self.spaceGrid.reshape(sh[0], sh[1]).mean(1)

        return Image(new_image, new_spaceGrid)
    
    def gen_image_variance(self, signal_to_noise, add_noise=False):
        gsImg = galsim.Image(np.ascontiguousarray(self.array.copy()), scale=self.pixScale)
        variance = gsImg.addNoiseSNR(galsim.GaussianNoise(), signal_to_noise, preserve_flux=True)
        if add_noise:  # replace self.array with the noise version
            self.array = gsImg.array
        return variance

    def add_psf(self, psfFWHM, psf_g1, psf_g2):
        psf = galsim.Gaussian(fwhm=psfFWHM)
        psf = psf.shear(g1=psf_g1, g2=psf_g2)

        thisIm = galsim.Image(np.ascontiguousarray(self.array), scale=self.pixScale)
        galobj = galsim.InterpolatedImage(image=thisIm)
        galC = galsim.Convolution([galobj, psf])
        newImage = galC.drawImage(image=galsim.Image(self.ngrid, self.ngrid, scale=self.pixScale))
        self.array = newImage.array
    
    def _get_mesh(self, mode='corner'):
        '''generate coordiante mesh
            Args: 
                mode: 'corner' or 'center'
                    corner mode: mesh coordinate refers to the bottom left corner of each pixel grid
                    center mode: mesh coordinate refers to the center of each pixel grid
        '''
        if mode == 'corner':
            spaceGrid_plt = self.spaceGrid - self.pixScale/2.
            Xmesh, Ymesh = np.meshgrid(spaceGrid_plt, spaceGrid_plt)
        else:
            Xmesh, Ymesh = np.meshgrid(self.spaceGrid, self.spaceGrid)
        return Xmesh, Ymesh
    
    def display(self, xlim=None, filename=None, title='image', mark_cen=True, model=None):
        '''display the 2D image array'''

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
        plt.rc('font', size=14)

        if model is None:
            Xmesh, Ymesh = self._get_mesh(mode='corner')
            gal = ax.pcolormesh(Xmesh, Ymesh, self.array)
        else:
            Xmesh, Ymesh = self._get_mesh(mode='center')
            gal = ax.contourf(Xmesh, Ymesh, self.array)
            mod = ax.contour(Xmesh, Ymesh, model, levels=gal.levels, colors='yellow')
            ax.clabel(mod, inline=1, fontsize=10)
        
        if mark_cen:
            ax.axvline(x=0., ls='--', color='lightgray', alpha=0.7)
            ax.axhline(y=0., ls='--', color='lightgray', alpha=0.7)

        ax.set_xlabel('x [arcsec]', fontsize=14)
        ax.set_ylabel('y [arcsec]', fontsize=14)
        ax.tick_params(labelsize=14)

        ax.set_title(title, fontsize=14)

        cbr = fig.colorbar(gal, ax=ax)
        cbr.ax.tick_params(labelsize=13)

        if xlim is not None:
            ax.set_xlim((xlim[0], xlim[1]))
            ax.set_ylim((xlim[0], xlim[1]))
        else:
            ax.set_xlim((self.spaceGrid.min(), self.spaceGrid.max()))
            ax.set_ylim((self.spaceGrid.min(), self.spaceGrid.max()))
        
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.tight_layout()
            fig.show()
        
        return fig, ax



class Slit:
    def __init__(self, specCube, slitWidth):

        self.specCube = specCube
        self.slit_mask = self.gen_mask(slitWidth=slitWidth)

    def gen_mask(self, slitWidth):

        ngrid = self.specCube.ngrid
        X, Y = np.meshgrid(self.specCube.spaceGrid, self.specCube.spaceGrid)
        mask = np.ones((ngrid, ngrid))
        mask[np.abs(Y) > slitWidth/2.] = 0.

        return mask

    def get_spectra(self, slitAngles):
        spectra = []

        for this_slit_angle in slitAngles:
            this_data = rotate(self.specCube.array, this_slit_angle * (180./np.pi), reshape=False)
            spectra.append(np.sum(this_data*self.slit_mask[:, :, np.newaxis], axis=0))

        return spectra


class Fiber:
    def __init__(self, specCube):

        self.specCube = specCube
        self.ngrid = specCube.ngrid
        X, Y = np.meshgrid(self.specCube.spaceGrid, self.specCube.spaceGrid)
        self.R = np.sqrt(X**2+Y**2)

    def gen_mask(self, fiberR):

        mask = np.ones((self.ngrid, self.ngrid))
        ID_out_R = np.where(self.R > fiberR)
        mask[ID_out_R] = 0.0

        return mask

    def get_spectrum(self, fiberR, expTime=None, area=None):
        '''
            Args:
                fiberR : fiber radius [unit: arcsec]
        '''

        mask = self.gen_mask(fiberR)
        maskCube = np.repeat(mask[:, :, np.newaxis], self.specCube.array.shape[2], axis=2)
        spectrum = np.sum(np.sum(self.specCube.array*maskCube, axis=0), axis=0)

        if (expTime is not None) and (area is not None):
            # if both expTime and telescope area information is given, 
            # return spectrum in the default unit of SDSS
            return Fiber.specPhoton_2_specSDSS(spectrum, expTime, area) # [unit: u.erg/u.Angstrom/u.s/u.cm**2]
        else:
            return spectrum  # [unit: u.photon/u.nm]
    
    @staticmethod
    def specPhoton_2_specSDSS(specPhoton, expTime, area):
        '''Perform unit transformation for a input spec1D [photons/nm] to the standard SDSS fiber spec unit [erg/Angstrom/s/cm^2], given the expTime, and telecscope area.

            Args:
                specPhoton : 1D array, spectrum in unit: photons/nm
                expTime: real, unit: sec
                area: real, telescope area, unit: cm2
            Returns:
                specSDSS : 1D array, spectrum in unit [erg/Angstrom/s/cm^2]
        '''
        specSDSS = (specPhoton*u.photon/u.nm)/(expTime*u.second)/(area*u.cm**2)

        energy_per_photon = constants.h*constants.c/(self.specCube.lambdaGrid*u.nm) / u.photon
        energy_per_photon = energy_per_photon.to(u.erg/u.photon)

        specSDSS = specSDSS*energy_per_photon

        return specSDSS.to(u.erg/u.Angstrom/u.s/u.cm**2).value


class SpecCube:
    def __init__(self, array3D, spaceGrid, lambdaGrid):
        self.array = array3D
        self.spaceGrid = spaceGrid
        self.lambdaGrid = lambdaGrid
    
    @property
    def pixScale(self):
        return self.spaceGrid[2]-self.spaceGrid[1]
    
    @property
    def ngrid(self):
        return len(self.spaceGrid)
    
    @property
    def nm_per_pixel(self):
        return self.lambdaGrid[2]-self.lambdaGrid[1]

    @property    
    def id_LOSwithEmitssion(self):
        '''LOS ids that have non-zero line emission signal'''
        return [k for k in range(len(self.lambdaGrid)) if np.any(self.array[:, :, k])]

    def cutout(self, xlim=[-2.5, 2.5], id_LOS=None):
        '''return a subCube given the xlim, and id_LOS
            Args:
                id_LOS: array of LOS ids to take out from the original 3D array
                    default: self.id_LOSwithEmitssion
        '''

        if id_LOS is None:
            id_LOS = self.id_LOSwithEmitssion

        id_x = np.where((self.spaceGrid >= xlim[0]) & (self.spaceGrid <= xlim[1]))[0]

        return SpecCube(self.array[id_x, :, :][:, id_x, :][:, :, id_LOS], self.spaceGrid[id_x], self.lambdaGrid[id_LOS])
    
    def rebin(self, shape, operation='sum'):
        ''' rebin self.array to the input shape
            Args:
                operation: 'sum' or 'mean'
        '''
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")

        sh = []
        for i in range(3):
            sh += [shape[i], self.array.shape[i]//shape[i]]
        
        #print(sh)

        if operation == 'sum':
            new_dataCube = self.array.reshape(sh).sum(5).sum(3).sum(1)
        else:
            new_dataCube = self.array.reshape(sh).mean(5).mean(3).mean(1)
        
        new_spaceGrid = self.spaceGrid.reshape(sh[0], sh[1]).mean(1)
        new_lambdaGrid = self.lambdaGrid.reshape(sh[4], sh[5]).mean(1)

        return SpecCube(new_dataCube, new_spaceGrid, new_lambdaGrid)
    
    def add_psf(self, psfFWHM, psf_g1, psf_g2):
        psf = galsim.Gaussian(fwhm=psfFWHM)
        psf = psf.shear(g1=psf_g1, g2=psf_g2)

        for k in self.id_LOSwithEmitssion:
            thisIm = galsim.Image(np.ascontiguousarray(self.array[:,:,k]), scale=self.pixScale)
            galobj = galsim.InterpolatedImage(image=thisIm)
            galC = galsim.Convolution([galobj, psf])
            newImage = galC.drawImage(image=galsim.Image(self.ngrid, self.ngrid, scale=self.pixScale))
            self.array[:,:,k] = newImage.array

    def _kernel_at_k(self, k, sigma2Grid):
        #return 1./np.sqrt(2*np.pi*self.sigma2Grid[k]) * np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*self.sigma2Grid[k]))
        weight = np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*sigma2Grid[k]))
        weight /= weight.sum()
        return weight
        
    def _smooth_spec11D(self, spec1D, sigma2Grid):

        smoothed_spec1D = np.zeros(len(spec1D))
        for k in range(len(self.lambdaGrid)):
            weightGird = self._kernel_at_k(k, sigma2Grid)
            smoothed_spec1D[k] = np.sum(weightGird*spec1D)
        return smoothed_spec1D
    
    def add_spec_sigma(self, resolution, sigma_thermal_nm=None):
        '''Smooth along the lambdaGrid for photonCube given spectragraph resolution, sigma_thermal
            Args:
                resolution: spectral resolution
                    Keck resolution = 5000
                sigma_thermal_nm: thermal contribution of velocity dispersion in unit: nm
        '''

        if sigma_thermal_nm is not None:
            sigma2Grid = (self.lambdaGrid/resolution)**2 + sigma_thermal_nm**2
        else:
            sigma2Grid = (self.lambdaGrid/resolution)**2

        for i, x in enumerate(self.spaceGrid):
            for j, y in enumerate(self.spaceGrid):
                self.array[j, i, :] = self._smooth_spec11D(spec1D=self.array[j, i, :], sigma2Grid=sigma2Grid)

    def add_spec_sigma_approx(self, sigma):
        '''Perform gaussian smooth in lambdaGrid direction, given sigma. 
           This mehtod is a quick approximation of self.add_spec_sigma
            Args: 
                sigma : real, in the same unit as lambdaGrid [unit: nm]
                    sigma = np.sqrt((lambda_cen/resolution)**2 + sigma_thermal_nm**2)
        '''
        sigma_pix = sigma/self.nm_per_pixel  # sigma in [unit: pix]
        self.array = gaussian_filter1d(self.array, sigma_pix, axis=2)
        
if __name__ == '__main__':
    Pars = ParametersTNG(sini=0.8, g1=0.05, g2=0.05, line_species='OII', redshift=0.)

    par_in = {'sini': 0.8, 'g1': 0.05}
    ParsIn = ParametersTNG(**par_in)

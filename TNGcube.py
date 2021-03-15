import numpy as np
import pickle
from rotations.rotations3d import rotation_matrices_from_vectors
import sys
import pathlib
import galsim
dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)

from tfCube2 import gen_grid

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy import constants
from astropy.io import fits
from scipy.ndimage.interpolation import rotate


class ParametersTNG:
    lineLambda0 = {'OIIa': 372.7092, 'OIIb': 372.9875,
                   'OIIIa': 496.0295, 'OIIIb': 500.8240,
                   'Halpha': 656.461}  # [unit: nm]

    def __init__(self, par_in=None):

        # 1. set default parameters of TNGcube
        self._pars0 = self._init_pars()

        # 2. define parameter dictionary, pars, based on user input, par_in
        #    for parameters thart are not defined, use their default value as par0
        self.fid = self._pars0.copy()
        if par_in is not None:
            for key in par_in.keys():
                self.fid[key] = par_in[key]
        
        self.define_grids(pars=self.fid)
        self.add_cosmoRedshift(redshift=self.fid['redshift'])

    def _init_pars(self):
        '''Initiate default parameters'''

        pars0 = {}
        pars0['redshift'] = 0.4
        pars0['sini'] = 0.5
        pars0['theta_int'] = 0.

        pars0['spinR'] = [0., 0., -1.]
        pars0['g1'] = 0.
        pars0['g2'] = 0.

        # grid parameters
        pars0['ngrid'] = 256
        pars0['image_size'] = 128
        pars0['pixScale'] = 0.1185

        pars0['nm_per_pixel'] = 0.033

        pars0['lambda_cen'] = (1. + pars0['redshift'])*656.461 # set default lambda_cen = halpha at redshift

        # observation parameters
        pars0['psfFWHM'] = 0.5
        pars0['psf_g1'] = 0.
        pars0['psf_g2'] = 0.
        pars0['Resolution'] = 5000. # for Keck

        pars0['slitWidth'] = 0.06

        # line intensity
        pars0['expTime'] = 30.*60.              # [unit: sec]
        pars0['area'] = 3.14 * (1000./2.)**2    # telescope area [unit: cm2]
        pars0['throughput'] = 0.29
        # peak of the reference SDSS line intersity at lambda_cen
        pars0['ref_SDSS_peakI'] = 3.*1e-17     # [unit: erg/s/Angstrom/cm2]

        pars0['read_noise'] = 3.0
        
        return pars0
    
    @property
    def int_SDSS_peakI(self):
        ref_peakI = self.fid['ref_SDSS_peakI'] * u.erg/u.second/u.Angstrom/u.cm**2
        ref_peakI = ref_peakI.to(u.erg/u.second/u.nm/u.cm**2)

        int_peakI = ref_peakI * (self.fid['area']*u.cm**2) * (self.fid['expTime']*u.second)

        energy_per_photon = constants.h*constants.c/(self.fid['lambda_cen']*u.nm) / u.photon
        energy_per_photon = energy_per_photon.to(u.erg/u.photon)

        return int_peakI/energy_per_photon
    
    def define_grids(self, pars):

        # grid parameters
        self.extent = pars['image_size'] * pars['pixScale']
        self.subGridPixScale = self.extent/pars['ngrid']
        self.lambda_min = pars['lambda_cen'] - 2.
        self.lambda_max = pars['lambda_cen'] + 2.

        self.spaceGrid = gen_grid(cen=0., pixScale=self.subGridPixScale, Ngrid=pars['ngrid'])
        Ngrid_l = int((self.lambda_max-self.lambda_min)/pars['nm_per_pixel'])+1
        self.lambdaGrid = gen_grid(cen=pars['lambda_cen'], pixScale=pars['nm_per_pixel'], Ngrid=Ngrid_l)

        self.spaceGrid_edg = np.append(self.spaceGrid-self.subGridPixScale/2., self.spaceGrid[-1]+self.subGridPixScale/2.)
        self.lambdaGrid_edg = np.append(self.lambdaGrid-pars['nm_per_pixel']/2., self.lambdaGrid[-1]+pars['nm_per_pixel']/2.)
    
        return pars
    
    def add_cosmoRedshift(self, redshift):

        # wavelength of lines being redshifted by cosmic expansion
        self.lineLambdaC = {key: (1.+redshift)*self.lineLambda0[key]
                            for key in self.lineLambda0.keys()}
        
        # comiving distance
        self.Dc = cosmo.comoving_distance(z=redshift).to(u.kpc).value * cosmo.h # [unit: ckpc/h]

        

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
        
        for key in ['cm', 'pos', 'spin', 'vel']:
            self.info[key] = self.vec3Dtransform(M=R, vec=self.info[key])

        for ptlType in ['gas', 'stars']:
            for key in ['pos', 'vel']:
                self.snap[ptlType][key] = self.vec3Dtransform(M=R, vec=self.snap[ptlType][key])
    
    def shear(self, g1, g2):
        '''Add shear on vector quantities in the x, and y directions. (The L.O.S direction is not affected.)
            L: the lensing shear matrix
        '''
        L = np.array([[1.+g1, g2, 0.],[g2, 1.-g1, 0.], [0., 0., 1.]])
        
        for key in ['cm', 'pos', 'spin', 'vel']:
            self.info[key]=self.vec3Dtransform(M=L, vec=self.info[key])

        for ptlType in ['gas', 'stars']:
            for key in ['pos', 'vel']:
                self.snap[ptlType][key]=self.vec3Dtransform(M=L, vec=self.snap[ptlType][key])
    


class TNGmock:

    def __init__(self, pars, subhalo):

        if isinstance(pars, dict):
            self.Pars = ParametersTNG(par_in=pars)
        elif isinstance(pars, ParametersTNG):
            self.Pars = pars
        else:
            raise TypeError("Argument, pars, needs to be a dictionary or an instance of the ParametersTNG class.")


        self.subhalo = subhalo
        self._init_constants()

        self.line_species = self._lines_within_lambdaGrid()
    
    def _init_constants(self):   
        self.c_kms = 2.99792458e5
        self.radian2arcsec = 206264.806247096

    def _lines_within_lambdaGrid(self):
        '''Find line_species that are within the range covered in lambdaGrid'''
        
        def is_between(lambda_in):
            return self.Pars.lambdaGrid[0] < lambda_in < self.Pars.lambdaGrid[-1]
        
        line_species = [key for key, val in filter(lambda item: is_between(item[1]), self.Pars.lineLambdaC.items())]

        return line_species

    def spin_rotation(self, spin0, spinR=[0.,0., -1.]):
        '''Spin Rotation Operator
            Compute the rotation matrix R_spin such that after applying R_spin on patricles, the new spin vector is aligned with the targeted spin direction.
            Args:
                spin0: [spin_x, spin_y, spin_z]
                    the original spin axis vector
                spinR: default = [0., 0., -1.]
                    targeted spin dirction after applying the rotation operator
            Return:
                R_spin: Rotation matrix to aling spin0 to spinR.
        '''
        R_spin = rotation_matrices_from_vectors(spin0, spinR)[0]
        return R_spin
    
    def sini_rotation(self, sini):
        '''Inclination Operator
            Args:
                sini: real
                    sin(inclination angle)
            Returns: 
                R_sini: Rotation matrix along the x-axis to incline a face-on disk defined in the x-y plane.
        '''
        cosi = np.sqrt(1.-sini**2)
        R_sini = np.array([[1., 0., 0.], [0., cosi, -sini], [0., sini, cosi]])
        return R_sini
    
    def PA_rotation(self, theta_int):
        '''Position angle Operator
                theta_int: real [unit: radian]
                    P.A. of a disk
            Returns:
                R_pa: Rotation matrix to rotate a disk along the z-axis by theta_int.
        '''
        sina = np.sin(theta_int)
        cosa = np.cos(theta_int)
        R_pa = np.array([[cosa, -sina, 0.],[sina, cosa, 0.],[0., 0., 1.]])

        return R_pa    
  
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

    def _massCube_i(self, ptlType, lineType, subhalo):
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
        
        x_arcsec = subhalo.snap[ptlType]['pos'][:, 0]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]
        y_arcsec = subhalo.snap[ptlType]['pos'][:, 1]/self.Pars.Dc * self.radian2arcsec #[unit: arcsec]
        lambdaLOS = self.vLOS_to_lambda(subhalo.snap[ptlType]['vel'][:, 2], lineType=lineType) #[unit: nm]
        mass = subhalo.snap[ptlType]['mass']*1.e10  # [unit: Msun/h]

        #massCube, _ = np.histogramdd((x_arcsec, y_arcsec, lambdaLOS), bins=(self.spaceGrid_edg, self.spaceGrid_edg, self.lambdaGrid_edg), weights=mass) 
        massCube, _ = np.histogramdd((y_arcsec, x_arcsec, lambdaLOS), 
                        bins=(self.Pars.spaceGrid_edg, self.Pars.spaceGrid_edg, self.Pars.lambdaGrid_edg), weights=mass)

        return massCube
    
    def gen_massCube(self, ptlTypes, lineTypes, subhalo):
        '''Generate the the sum of massCube for all input ptlTypes and lineTypes
            Args:
                ptlTypes: list
                    e.g. ptlTypes = ['gas', 'stars']
                lineTypes: list, line species that fall within the range of lambdaGrid
                    lineTypes = self.line_species
            Returns:
                massCube: 3D array (x, y, lambda)
                    [unit: Msun/h /x_grid/y_grid/lambda_grid
        '''

        massCube = np.zeros([self.Pars.spaceGrid.size, self.Pars.spaceGrid.size, self.Pars.lambdaGrid.size])

        for lineType in lineTypes:
            for ptlType in ptlTypes:
                massCube += self._massCube_i(ptlType, lineType, subhalo)

        return massCube
    
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
        '''Perform flux re-normalization for photonCube such that the integrated fiber spectrum is consistent with the given SDSS fiber spectrum set in self.Pars.fid['ref_SDSS_peakI']*expTime*area
        '''
        fiberObj = FiberSpec(specCube)
        spec1D = fiberObj.get_spectrum(fiberR=1.5)  # SDSS fiber Radius=1.5 arcsec
        Nphoton_peak = spec1D.max()*u.photon/u.nm
        renorm_factor = self.Pars.int_SDSS_peakI/Nphoton_peak
        specCube.array *= renorm_factor.value

        return specCube
    
    def add_sky_noise(self, specCube):
        
        self.sky = Sky(self.Pars)

        for k in range(self.Pars.lambdaGrid.size):
            thisIm = galsim.Image(np.ascontiguousarray(specCube.array[:, :, k]), scale=specCube.gridPixScale)
            noise = galsim.CCDNoise(sky_level = self.sky.spec1D[k], read_noise = self.Pars.fid['read_noise'])
            noiseImage = thisIm.copy()
            noiseImage.addNoise(noise)

            specCube.array[:, :, k] = noiseImage.array
        
        return specCube


class Sky:
    def __init__(self, pars, skyfile=dir_KLens+'/data/Simulation/skytable.fits'):

        if isinstance(pars, dict):
            self.Pars = ParametersTNG(par_in=pars)
        elif isinstance(pars, ParametersTNG):
            self.Pars = pars
        else:
            raise TypeError("Argument, pars, needs to be a dictionary or an instance of the ParametersTNG class.")
        
        self.skyTemplate = fits.getdata(skyfile)

    @property
    def spec1D(self):
        '''
            raw sky flux in the file is: photon/s/m2/micron/arcsec2
            convert skySpec1D unit to photon/s/cm2/nm/arcsec2
            and to photons (/1 lambda_pixel/1 x_pixel/1 y_pixel), after integration
        '''
        spec = np.interp(self.Pars.lambdaGrid, self.skyTemplate['lam']*1000., self.skyTemplate['flux'])
        spec /= 1.0e7  # 1/1000 for micron <-> nm ; 1/10000 for m2 <-> cm2
        spec *= self.Pars.fid['expTime']*self.Pars.fid['area']*self.Pars.fid['throughput'] * \
            self.Pars.subGridPixScale**2*self.Pars.fid['nm_per_pixel']

        return spec
    
    @property
    def skyCube(self):

        skyArray = np.empty([self.Pars.fid['ngrid'], self.Pars.fid['ngrid'], self.Pars.lambdaGrid.size])
        skyArray[:, :, :] = self.spec1D[None, None, :]

        return SpecCube(skyArray, self.Pars.spaceGrid, self.Pars.lambdaGrid)
    
    @property
    def spec2D(self):
        slitObj = SlitSpec(self.skyCube, slitWidth=self.Pars.fid['slitWidth'])
        return slitObj.get_spectra(slitAngles=[0.])[0]

        
    




class SlitSpec:
    def __init__(self, specCube, slitWidth):

        self.specCube = specCube
        self.slit_weight = self.weight_mask(slitWidth=slitWidth)

    def weight_mask(self, slitWidth):

        ngrid = len(self.specCube.spaceGrid)
        X, Y = np.meshgrid(self.specCube.spaceGrid, self.specCube.spaceGrid)
        weight = np.ones((ngrid, ngrid))
        weight[np.abs(Y) > slitWidth/2.] = 0.

        return weight

    def get_spectra(self, slitAngles):
        spectra = []

        for this_slit_angle in slitAngles:
            this_data = rotate(self.specCube.array, this_slit_angle * (180./np.pi), reshape=False)
            spectra.append(np.sum(this_data*self.slit_weight[:, :, np.newaxis], axis=0))

        return spectra


class FiberSpec:
    def __init__(self, specCube):

        self.specCube = specCube
        self.ngrid = len(self.specCube.spaceGrid)
        X, Y = np.meshgrid(self.specCube.spaceGrid, self.specCube.spaceGrid)
        self.R = np.sqrt(X**2+Y**2)

    def weight_mask(self, fiberR):

        weight = np.ones((self.ngrid, self.ngrid))
        ID_out_R = np.where(self.R > fiberR)
        weight[ID_out_R] = 0.0

        return weight

    def get_spectrum(self, fiberR, expTime=None, area=None):
        '''
            Args:
                fiberR : fiber radius [unit: arcsec]
        '''

        weight = self.weight_mask(fiberR)
        weightCube = np.repeat(weight[:, :, np.newaxis], self.specCube.array.shape[2], axis=2)
        spectrum = np.sum(np.sum(self.specCube.array*weightCube, axis=0), axis=0)

        if (expTime is not None) and (area is not None):
            # if both expTime and telescope area information is given, 
            # return spectrum in the default unit of SDSS
            return self._specPhoton_2_specSDSS(spectrum, expTime, area) # [unit: u.erg/u.Angstrom/u.s/u.cm**2]
        else:
            return spectrum  # [unit: u.photon/u.nm]
    
    def _specPhoton_2_specSDSS(self, specPhoton, expTime, area):
        '''
            Args:
                specPhoton : 1D array, spectrum in unit: photons/nm
                expTime: real, unit: sec
                area: real, telescope area, unit: cm2
        '''
        specSDSS = (specPhoton*u.photon/u.nm)/(expTime*u.second)/(area*u.cm**2)

        energy_per_photon = constants.h*constants.c/(self.specCube.lambdaGrid*u.nm) / u.photon
        energy_per_photon = energy_per_photon.to(u.erg/u.photon)

        specSDSS = specSDSS*energy_per_photon

        return specSDSS.to(u.erg/u.Angstrom/u.s/u.cm**2).value


class SpecCube:
    def __init__(self, array, spaceGrid, lambdaGrid):
        self.array = array
        self.spaceGrid = spaceGrid
        self.lambdaGrid = lambdaGrid
    
    @property
    def gridPixScale(self):
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

        if operation == 'sum':
            new_dataCube = self.array.reshape(sh).sum(5).sum(3).sum(1)
        else:
            new_dataCube = self.array.reshape(sh).mean(5).mean(3).mean(1)
        
        new_spaceGrid = self.spaceGrid.reshape(sh[0], self.array.shape[0]//shape[0]).mean(1)
        new_lambdaGrid = self.lambdaGrid.reshape(sh[2], self.array.shape[2]//shape[2]).mean(1)

        return SpecCube(new_dataCube, new_spaceGrid, new_lambdaGrid)
    
    def add_psf(self, psfFWHM, psf_g1, psf_g2):
        psf = galsim.Gaussian(fwhm=psfFWHM)
        psf = psf.shear(g1=psf_g1, g2=psf_g2)

        for k in self.id_LOSwithEmitssion:
            thisIm = galsim.Image(np.ascontiguousarray(self.array[:,:,k]), scale=self.gridPixScale)
            galobj = galsim.InterpolatedImage(image=thisIm)
            galC = galsim.Convolution([galobj, psf])
            newImage = galC.drawImage(image=galsim.Image(self.ngrid, self.ngrid, scale=self.gridPixScale))
            self.array[:,:,k] = newImage.array

    def kernel_at_k(self, k, sigma2Grid):
        #return 1./np.sqrt(2*np.pi*self.sigma2Grid[k]) * np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*self.sigma2Grid[k]))
        weight = np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*sigma2Grid[k]))
        weight /= weight.sum()
        return weight
        
    def _smooth_spec11D(self, spec1D, sigma2Grid):

        smoothed_spec1D = np.zeros(len(spec1D))
        for k in range(len(self.lambdaGrid)):
            weightGird = self.kernel_at_k(k, sigma2Grid)
            smoothed_spec1D[k] = np.sum(weightGird*spec1D)
        return smoothed_spec1D
    
    def add_spec_resolution(self, resolution):
        '''Smooth along lambdaGrid for photonCube for spectragraph resolution.
            Note: Don't need to smooth if sigma = np.sqrt(sigma2Grid) is << nm_per_pixel
        '''
        sigma2Grid = self.lambdaGrid/resolution**2

        for i, x in enumerate(self.spaceGrid):
            for j, y in enumerate(self.spaceGrid):
                self.array[j, i, :] = self._smooth_spec11D(spec1D=self.array[j, i, :], sigma2Grid=sigma2Grid)

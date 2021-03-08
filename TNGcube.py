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
from KLtool import getSlitSpectra, getFiberSpec

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.ndimage.interpolation import rotate


def vec3Dtransform(M, vec):
    '''Coordinate transformation on vector given the input matrix operator
        Args:
            M: ndarray 3x3
                transformation matrix
            vec: ndarray Nx3
                Vector coordinates before applying M
    '''
    vec = np.array(vec)
    vecM = np.dot(M, vec.T).T
    return vecM


class ParametersTNG():
    def __init__(self, par_in=None):

        # 1. set default parameters of TNGcube
        self.pars0 = self.init_pars()

        # 2. define parameter dictionary, pars, based on user input, par_in
        #    for parameters thart are not defined, use their default value as par0
        self.fid = self.pars0.copy()
        if par_in is not None:
            for key in par_in.keys():
                self.fid[key] = par_in[key]
        self.fid = self.update_derived_par(pars=self.fid)


    def init_pars(self):
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
        
        pars0 = self.update_derived_par(pars=pars0)

        return pars0
    
    def update_derived_par(self, pars):

        # grid parameters
        pars['extent'] = pars['image_size'] * pars['pixScale']
        pars['subGridPixScale'] = pars['extent']/pars['ngrid']
        
        pars['lambda_min'] = pars['lambda_cen'] - 2.
        pars['lambda_max'] = pars['lambda_cen'] + 2.

        return pars
    
    def gen_par_dict(self, active_par, active_par_key, par_ref):

        pars = par_ref.copy()

        for j, item in enumerate(active_par_key):
            pars[item] = active_par[j]

        return pars
        

class TNGcube():
    def __init__(self, pars, subhaloInfo):
        
        self.Pars = ParametersTNG(par_in=pars)
        self.subhaloInfo = subhaloInfo
        self._init_frequent_pars(redshift=self.Pars.fid['redshift'])
    
    def _init_frequent_pars(self, redshift):
        self.c_kms = 2.99792458e5
        self.radian2arcsec = 206264.806247096

        # comiving distance
        self.Dc = cosmo.comoving_distance(z=self.Pars.fid['redshift']).to(u.kpc).value * cosmo.h # [unit: ckpc/h]

        self.lineLambda0 = {'OIIa': 372.7092, 'OIIb': 372.9875, \
                            'OIIIa': 496.0295, 'OIIIb': 500.8240, \
                            'Halpha': 656.461 } # [unit: nm]

        # wavelength of lines being redshifted by cosmic expansion
        self.lineLambdaC = { key: (1.+redshift)*self.lineLambda0[key] 
                             for key in self.lineLambda0.keys() }
        
        # init grid coordinates
        self.spaceGrid = gen_grid(cen=0., pixScale=self.Pars.fid['subGridPixScale'], Ngrid=self.Pars.fid['ngrid'])
        Ngrid_l = int((self.Pars.fid['lambda_max']-self.Pars.fid['lambda_min'])/self.Pars.fid['nm_per_pixel'])+1
        self.lambdaGrid = gen_grid(cen=self.Pars.fid['lambda_cen'], pixScale=self.Pars.fid['nm_per_pixel'], Ngrid=Ngrid_l)

        self.spaceGrid_edg = np.append(self.spaceGrid-self.Pars.fid['subGridPixScale']/2., self.spaceGrid[-1]+self.Pars.fid['subGridPixScale']/2.)
        self.lambdaGrid_edg = np.append(self.lambdaGrid-self.Pars.fid['nm_per_pixel']/2., self.lambdaGrid[-1]+self.Pars.fid['nm_per_pixel']/2.)

        self.line_species = self._lines_within_lambdaGrid()
        self.sigma2Grid = self.lambdaGrid/self.Pars.fid['Resolution']**2

        self.psf = self.galsimPSF()
        
        self.X, self.Y = np.meshgrid(self.spaceGrid, self.spaceGrid)
        self.slit_weight = np.ones((self.Pars.fid['ngrid'], self.Pars.fid['ngrid']))
        self.slit_weight[np.abs(self.Y) > self.Pars.fid['slitWidth']/2.] = 0.        


    def _lines_within_lambdaGrid(self):
        '''Find line_species that are within the range covered in lambdaGrid'''
        
        def is_between(lambda_in):
            return self.lambdaGrid[0] < lambda_in < self.lambdaGrid[-1]
        
        line_species = [key for key, val in filter(lambda item: is_between(item[1]), self.lineLambdaC.items())]

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
    
    def subhalo_rotation(self, R, subhaloInfo0):
        '''Perform cooordination transformation on all vector quantities of subhaloInfo given the matrix operator R.
            Args:
                R: 3x3 array
                    total rotation operator : R = R_pa@R_sini@R_spin
        '''

        subhaloInfoR = {    key : vec3Dtransform(M=R, vec=subhaloInfo0[key]) 
                            for key in ['cm', 'pos', 'spin', 'vel']            }

        subhaloInfoR['snapInfo'] = {    ptlType: {  key: vec3Dtransform(M=R, vec=subhaloInfo0['snapInfo'][ptlType][key]) 
                                                    for key in ['pos', 'vel']   }
                                        for ptlType in ['gas', 'stars']                 }
        
        return subhaloInfoR
    
    def add_shear(self, g1, g2, subhaloInfoR):
        '''Add shear on vector quantities in the x, and y directions. (The L.O.S direction is not affected.)
            L: the lensing shear matrix
        '''
        L = np.array([[1.+g1, g2, 0.],[g2, 1.-g1, 0.], [0., 0., 1.]])
        
        subhaloInfoL = {    key : vec3Dtransform(M=L, vec=subhaloInfoR[key]) 
                            for key in ['cm', 'pos', 'spin', 'vel']            }

        subhaloInfoL['snapInfo'] = {    ptlType: {  key: vec3Dtransform(M=L, vec=subhaloInfoR['snapInfo'][ptlType][key]) 
                                                    for key in ['pos', 'vel']   }
                                        for ptlType in ['gas', 'stars']                 }
        return subhaloInfoL
    
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

        lambdaLOS = (1.-v_z/self.c_kms)*self.lineLambdaC[lineType]

        return lambdaLOS

    def _massCube_i(self, ptlType, lineType, subhaloInfoL):
        '''Generate the massCube for given ptlType, lineType
            Returns:
                massCube: 3D array (x, y, lambda) [unit: Msun/h / pix^3]
            
            Note:
                When calling np.histogramdd to make 3D histgram, input ptl cooridnates need to be: (y, x, lambda), 
                in order to produce consistent mesh definition as tfCube. 
                This way, tfCube2.modelCube and TNGCube.massCube can adopt the same plotting routing 
                i.e. imshow(np.sum(massCube, axis = 2), origin='lower').

                If the input ptl cooridnates are ordered by (x, y, lambda), then the ploting routings for both 3D cubes would differ:
                    imshow(np.sum(modelCube , axis = 2), origin='lower')    # for tfCube.modelCube
                    imshow(np.sum(massCube.T, axis = 2), origin='lower')    # for TNGCube.massCube
        '''
        
        x_arcsec = subhaloInfoL['snapInfo'][ptlType]['pos'][:, 0]/self.Dc * self.radian2arcsec #[unit: arcsec]
        y_arcsec = subhaloInfoL['snapInfo'][ptlType]['pos'][:, 1]/self.Dc * self.radian2arcsec #[unit: arcsec]
        lambdaLOS = self.vLOS_to_lambda(subhaloInfoL['snapInfo'][ptlType]['vel'][:, 2], lineType=lineType) #[unit: nm]
        mass = self.subhaloInfo['snapInfo'][ptlType]['mass']*1.e10  #[unit: Msun/h]

        #massCube, _ = np.histogramdd((x_arcsec, y_arcsec, lambdaLOS), bins=(self.spaceGrid_edg, self.spaceGrid_edg, self.lambdaGrid_edg), weights=mass) 
        massCube, _ = np.histogramdd((y_arcsec, x_arcsec, lambdaLOS), bins=(
            self.spaceGrid_edg, self.spaceGrid_edg, self.lambdaGrid_edg), weights=mass)

        return massCube #, x_arcsec, y_arcsec, mass, lambdaLOS
    
    def gen_massCube(self, ptlTypes, lineTypes, subhaloInfoL):
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

        massCube = np.zeros([self.spaceGrid.size, self.spaceGrid.size, self.lambdaGrid.size])

        for lineType in lineTypes:
            for ptlType in ptlTypes:
                massCube += self._massCube_i(ptlType, lineType, subhaloInfoL)

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

        return photonCube
    
    def galsimPSF(self):
        psf = galsim.Gaussian(fwhm=self.Pars.fid['psfFWHM'])
        psf = psf.shear(g1=self.Pars.fid['psf_g1'], g2=self.Pars.fid['psf_g2'])
        return psf

    def add_psf(self, photonCube):

        for k in range(len(self.lambdaGrid)):

            is_flux_image = np.any(photonCube[:,:,k])

            if is_flux_image:
                thisIm = galsim.Image(np.ascontiguousarray(photonCube[:,:,k]), scale=self.Pars.fid['subGridPixScale'])
                galobj = galsim.InterpolatedImage(image=thisIm)
                galC = galsim.Convolution([galobj, self.psf])
                newImage = galC.drawImage(image=galsim.Image(self.Pars.fid['ngrid'], self.Pars.fid['ngrid'], scale=self.Pars.fid['subGridPixScale']))
                photonCube[:,:,k] = newImage.array

        return photonCube

    def kernel_at_k(self, k):
        #return 1./np.sqrt(2*np.pi*self.sigma2Grid[k]) * np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*self.sigma2Grid[k]))
        weight = np.exp(- (self.lambdaGrid[k]-self.lambdaGrid)**2 / (2.*self.sigma2Grid[k]))
        return weight/weight.sum()

    def smooth_spec11D(self, spec1D):
        
        smoothed_spec1D = np.zeros(len(spec1D))
        for k in range(len(self.lambdaGrid)):
            weightGird = self.kernel_at_k(k)
            smoothed_spec1D[k] = np.sum(weightGird*spec1D)
        return smoothed_spec1D
   
    def add_spec_resolution(self, photonCube):
        '''Smooth along lambdaGrid for photonCube for spectragraph resolution.
            Note: Don't need to smooth if sigma = np.sqrt(self.sigma2Grid) is << nm_per_pixel
        '''
        
        for i, x in enumerate(self.spaceGrid):
            for j, y in enumerate(self.spaceGrid):
                photonCube[j, i, :] = self.smooth_spec11D(spec1D=photonCube[j, i, :])

        return photonCube
    
    def flux_renorm(self, photonCube):
        '''Perform flux re-normalization for photonCube such that the integrated fiber spectrum is consistent with given SDSS fiber spectrum
        '''
        pass
    
    def spec2D(self, photonCube):
        spectra = getSlitSpectra(photonCube, slitAngles=self.Pars.fid['slitAngles'], slit_weight=self.slit_weight)
        return spectra


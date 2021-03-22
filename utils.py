import numpy as np
from rotations.rotations3d import rotation_matrices_from_vectors

def spin_rotation(spin0, spinR=[0., 0., -1.]):
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

def sini_rotation(sini):
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

def PA_rotation(theta):
    '''Position angle Operator
            theta_int: real [unit: radian]
                P.A. of a disk
        Returns:
            R_pa: Rotation matrix to rotate a disk along the z-axis by theta.
    '''
    sina = np.sin(theta)
    cosa = np.cos(theta)
    R_pa = np.array([[cosa, -sina, 0.], [sina, cosa, 0.], [0., 0., 1.]])

    return R_pa

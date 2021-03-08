import requests
import os
import numpy as np
import h5py
import pickle

baseURL = 'http://www.tng-project.org/api/'
headers = {"api-key": "b703779c1f099efed6f47b91607b1bb1"}


def available_simNames(baseURL=baseURL, headers=headers):
    rInfo = get(baseURL)
    simNames = [sim['name'] for sim in rInfo['simulations']]
    return simNames


def get_snapNum(redshift, simName='TNG50-1', baseURL=baseURL):
    '''Find the Snapshat number that are closest to the input redshift, given simName.'''
    simSnapsInfo = get(baseURL+simName+'/snapshots/')
    snaps_z_list = [simSnapsInfo[i]['redshift'] for i in range(len(simSnapsInfo))]
    snapNum = min(range(len(snaps_z_list)), key=lambda i: abs(snaps_z_list[i]-redshift))
    return snapNum


class QueryTNG():
    def __init__(self, simName, snapNum, baseURL=baseURL, headers=headers):

        self.simURL = baseURL+simName+'/'
        self.snapURL = self.simURL+f'snapshots/{int(snapNum)}/'
        self.subhalosURL = self.snapURL + 'subhalos/'

        self.simInfo = get(self.simURL)
        self.snapInfo = get(self.snapURL)

        self.h = self.simInfo['hubble'] # hubble constant of TNG simulations
        self.redshift = self.snapInfo['redshift']
        self.a = 1./(1.+self.redshift)

        self.simName = simName
        self.snapNum = snapNum
    
    def query_subhaloCat(self, mass_min=1e8, vmax_min=50., limit=10, pageID=0):
        '''Query a catalog of available subhalo IDs whose masses and rotation curves are above the input thresholds. 
            Args:
                mass_min : double [unit: Msun/h]
                vmax_min : double [unit: km/s]
                    query subhalos with vmax > vmax_min.
                    vmax is the maximum value of the spherically-averaged rotation curve
                limit : int
                    number of subhalos to query for each pageID
                pageID : int
                    e.g. if set limie=10:
                        pageID = 0, return the first 10 subhalo IDs indexed from 0~9
                        pageID = 1, return the next 10 subhalos IDs indexed from 10~19
            Returns:
                suhaloIDs : list
                    a list of suhaloIDs satisfying the query condition.
        '''
        mass_min_1e10Msun_h = mass_min/1e10*self.h  # mass_min_1e10Msun_h [unit: 1e10 Msun/h]

        search_query = f'?limit={limit}&offset={int(pageID*limit)}&mass__gt={mass_min_1e10Msun_h}&vmax__gt={vmax_min}'

        searchurl = self.subhalosURL + search_query
        self.subhaloCat = get(searchurl)
        subhaloIDs = [self.subhaloCat['results'][i]['id'] for i in range(limit)]
        return subhaloIDs

    def query_subhaloInfo(self, subhaloID):

        subhaloURL = self.subhalosURL + f'{int(subhaloID)}/'

        subhaloInfo_all = get(subhaloURL)

        keys_takeout = ['snap', 'id', 'mass', 'stellarphotometrics_r', 'vmax', 'vmaxrad', 'mass_log_msun']

        subhaloInfo = {key: subhaloInfo_all[key] for key in keys_takeout}

        key_vec = ['cm', 'pos', 'spin', 'vel']
        _xyz = ['_x', '_y', '_z']

        for key in key_vec:
            subhaloInfo[key] = [subhaloInfo_all[key+i] for i in _xyz]

        return subhaloInfo
    
    def load_subhaloCutout(self, subhaloID):
        '''Download the snapshot particle information in the given subhalo field.
            Output file: 
                e.g. if subhaloID=46, output a file at './cutout_46.hdf5'
        '''

        cutout_request = {'gas'  :'Coordinates,Masses,Velocities', \
                          'stars': 'Coordinates,Masses,Velocities'} 
                        # 'dm'   :'Coordinates,Velocities'

        subhaloURL = self.subhalosURL + f'{int(subhaloID)}/'
        print(subhaloURL+'cutout.hdf5')
        fname_cutout = get(subhaloURL+'cutout.hdf5', cutout_request)
        return fname_cutout
    
    def _preprocess_snap_arrs(self, f_hdf5, ptl_type, subhaloInfo):
        '''
            Returns:
                snap['pos']  : ndarray Nptlx3 [unit: ckpc/h]
                    particle position w.r.t. the subhalo central of mass coordinate
                snap['mass'] : 1d array       [unit: 10^10 Msun/h]
                    particle mass 
                snap['vel']  : ndarray Nptlx3 [unit: km/s]
                    particle velocities w.r.t. the overall systematic velocity of the subhalo
                    (Note: the sqrt(a) factor in the original unit is taken away after processing.)
        '''

        snap = {    'pos': f_hdf5[ptl_type]['Coordinates'][:, :], 
                    'vel': f_hdf5[ptl_type]['Velocities'][:, :]*np.sqrt(self.a), 
                        # change the unit of velocity from [sqrt(a) km/s] to [km/s]
                    'mass': f_hdf5[ptl_type]['Masses'][:]   }

        # modify the ptl. position and velocity vectors such that they are w.r.t. the subhalo c.m. and system velocity
        for j in range(3):
            snap['pos'][:, j] -= subhaloInfo['cm'][j]
            snap['vel'][:, j] -= subhaloInfo['vel'][j]

        return snap
    
    def download_subhalos(self, subhaloIDs):
        '''Download and preprocess the queried subhalo information, and save it as a dictionary that can work with TNGcube.
            Output file:
                e.g. subhaloIDs = [46, 47]
                save the subhalo info. as './cutout_46.pkl' and './cutout_47.pkl'
                to load subhalo info. back: 
                    subhaloInfo = pickle.load(open('cutout_46.pkl', 'rb'))
        '''

        subhaloInfos = {}
        for ID in subhaloIDs:
            subhaloInfo = self.query_subhaloInfo(ID)
            fhdf5_cutout = self.load_subhaloCutout(ID)
            
            with h5py.File(fhdf5_cutout, 'r') as f_hdf5:
                snapInfo = {    'gas' :  self._preprocess_snap_arrs(f_hdf5, 'PartType0', subhaloInfo),
                                'stars' : self._preprocess_snap_arrs(f_hdf5, 'PartType4', subhaloInfo)  }

            subhaloInfo['snapInfo'] = snapInfo

            os.remove(fhdf5_cutout)
            fpkl_cutout = f'cutout_{ID}.pkl'
            pickle.dump(subhaloInfo, open(fpkl_cutout, 'wb'))
            subhaloInfos[ID] = subhaloInfo

        return subhaloInfos
    


def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json()  # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename  # return the filename string

    return r


if __name__ == '__main__':
    Q_TNG = QueryTNG(simName='TNG50-1', snapNum=75)

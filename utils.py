import numpy as np

def rebin2D(arr, shape, operation='sum'):
    '''
        Args:
            operation: 'sum' or 'mean'
        
        Example
        -------
        > a = np.arange(24).reshape((4,6))
        > a_sh = rebin2D(a, shape=[2,3], operation='sum')
        
    '''
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")

    sh = shape[0], arr.shape[0]//shape[0], shape[1], arr.shape[1]//shape[1]

    if operation == 'sum':
        return arr.reshape(sh).sum(3).sum(1)
    else:
        return arr.reshape(sh).mean(3).mean(1)

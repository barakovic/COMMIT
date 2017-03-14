#!python
# cython: boundscheck=False, wraparound=False, profile=False

import cython
import numpy as np
cimport numpy as np
import nibabel
from os.path import join, exists
from os import makedirs
from tempfile import TemporaryFile
import time
import scipy
from scipy import ndimage

# Interface to actual C code
cdef extern from "spline2dictionary_c.cpp":
   int spline2dictionary_IC( char* strTRKfilename, int Nx, int Ny, int Nz, float Px, float Py, float Pz, int n_count, int n_scalars, int n_properties, float fiber_shift, int points_to_skip, int c, int splinePts, float* ptr_CP_old, float* ptr_CP_new, float* ptrMASK, float* ptrGMASK,float* ptrTDI, float* ptrTRK, char* path_out ) nogil

# Interface to actual C code
cdef extern from "spline2dictionary_c.cpp":
   int spline2dictionary_EC( float* ptrPEAKS, int Np, float vf_THR, int ECix, int ECiy, int ECiz, float* ptrTDI ) nogil

# Interface to actual C code
cdef extern from "spline2dictionary_c.cpp":
   int simulated_annealing( float normL2COMMIT, float num_random, float* ptr_CP_old, float* ptr_CP_new, float* ptrTRK, float* ptrMASK, float* ptrGMASK, float* ptrEe, float* ptrEm, float* ptrX ) nogil

# Interface to actual C code
cdef extern from "spline2dictionary_c.cpp":
   int create_trk( float* ptr_CP_old, float* ptrTRK ) nogil

cpdef dictionary(filename_trk, path_out, filename_peaks, filename_mask, filename_gmask, n_control_points, do_intersect = True, fiber_shift = 0.5, points_to_skip = 0, vf_THR = 0.1, flip_peaks = [False,False,False]):

    '''
    do_intersect : boolean
        If True then fiber segments that intersect voxel boundaries are splitted (default).
        If False then the centroid of the segment is used as its voxel position.

    fiber_shift : float
        If necessary, apply a translation to fiber coordinates (default : 0) to account
        for differences between the reference system of the tracking algorithm and COMMIT.
        The value is specified in voxel units, eg 0.5 translates by half voxel.

    filename_mask : string
        Path to a binary mask to use to restrict the analysis to specific areas.

    points_to_skip : integer
        If necessary, discard first points at beginning/end of a fiber (default : 0).
    '''

    print '\n-> Creating the dictionary from spline representation (given a .trk file):'
    print '\t* Segment position = %s' % ( 'COMPUTE INTERSECTIONS' if do_intersect else 'CENTROID' )
    print '\t* Fiber shift      = %.3f (voxel-size units)' % fiber_shift
    print '\t* Points to skip   = %d' % points_to_skip

    print '\t* Loading data:'

    # fiber-tracts from .trk
    print '\t\t* tractogram'
    try :
        fib, trk_hdr = nibabel.trackvis.read( filename_trk, as_generator=True )
    except :
        raise IOError( 'Track file not found' )
    Nx = trk_hdr['dim'][0]
    Ny = trk_hdr['dim'][1]
    Nz = trk_hdr['dim'][2]
    Px = trk_hdr['voxel_size'][0]
    Py = trk_hdr['voxel_size'][1]
    Pz = trk_hdr['voxel_size'][2]
    print '\t\t\t- %d x %d x %d' % ( Nx, Ny, Nz )
    print '\t\t\t- %.4f x %.4f x %.4f' % ( Px, Py, Pz )
    print '\t\t\t- %d fibers' % trk_hdr['n_count']

    # white-matter mask
    cdef float* ptrMASK
    cdef float [:, :, ::1] niiMASK_img
    if filename_mask is not None :
        print '\t\t* filtering mask'
        niiMASK = nibabel.load( filename_mask )
        print '\t\t\t- %d x %d x %d' % ( niiMASK.shape[0], niiMASK.shape[1], niiMASK.shape[2] )
        print '\t\t\t- %.4f x %.4f x %.4f' % ( niiMASK.get_header()['pixdim'][1], niiMASK.get_header()['pixdim'][2], niiMASK.get_header()['pixdim'][3] )
        if ( Nx!=niiMASK.shape[0] or Ny!=niiMASK.shape[1] or Nz!=niiMASK.shape[2] or
             abs(Px-niiMASK.get_header()['pixdim'][1])>1e-3 or abs(Py-niiMASK.get_header()['pixdim'][2])>1e-3 or abs(Pz-niiMASK.get_header()['pixdim'][3])>1e-3 ) :
            print '\t\t  [WARNING] WM dataset does not have the same geometry as the tractogram'
        niiMASK_img = np.ascontiguousarray( niiMASK.get_data().astype(np.float32) )
        ptrMASK  = &niiMASK_img[0,0,0]
    else :
        print '\t\t* no mask specified to filter IC compartments'
        ptrMASK = NULL

    # gray-matter mask
    cdef float* ptrGMASK
    cdef float [:, :, ::1] niiGMASK_img
    if filename_gmask is not None :
        print '\t\t* filtering gray mask'
        niiGMASK = nibabel.load( filename_gmask )
        niiGMASK_img = np.ascontiguousarray( niiGMASK.get_data().astype(np.float32) )
        ptrGMASK  = &niiGMASK_img[0,0,0]
    else :
        print '\t\t* no mask specified to filter IC compartments'
        ptrGMASK = NULL

    # peaks file for EC contributions
    cdef float* ptrPEAKS
    cdef float [:, :, :, ::1] niiPEAKS_img
    cdef int Np
    if filename_peaks is not None :
        print '\t\t* EC orientations'
        niiPEAKS = nibabel.load( filename_peaks )
        print '\t\t\t- %d x %d x %d x %d' % ( niiPEAKS.shape[0], niiPEAKS.shape[1], niiPEAKS.shape[2], niiPEAKS.shape[3] )
        print '\t\t\t- %.4f x %.4f x %.4f' % ( niiPEAKS.get_header()['pixdim'][1], niiPEAKS.get_header()['pixdim'][2], niiPEAKS.get_header()['pixdim'][3] )
        print '\t\t\t- ignoring peaks < %.2f * MaxPeak' % vf_THR
        print '\t\t\t- flipping axes : [ x=%s, y=%s, z=%s ]' % ( flip_peaks[0], flip_peaks[1], flip_peaks[2] )
        
        if niiPEAKS.shape[3] % 3 :
            raise RuntimeError( 'PEAKS dataset must have 3*k volumes' )
        if vf_THR < 0 or vf_THR > 1 :
            raise RuntimeError( 'vf_THR must be between 0 and 1' )
        niiPEAKS_img = np.ascontiguousarray( niiPEAKS.get_data().astype(np.float32) )
        ptrPEAKS = &niiPEAKS_img[0,0,0,0]
        Np = niiPEAKS.shape[3]/3
    else :
        print '\t\t* no dataset specified for EC compartments'
        Np = 0
        ptrPEAKS = NULL

    # output path
    print '\t\t* output written to "%s"' % path_out
    if not exists( path_out ):
        makedirs( path_out )

    # create TDI mask
    cdef float [:, :, ::1] niiTDI_img = np.ascontiguousarray( np.zeros((Nx,Ny,Nz),dtype=np.float32) )
    cdef float* ptrTDI  = &niiTDI_img[0,0,0]

    # create the structure for the control points 
    cdef float* ptr_CP_old
    cdef float [:, ::1] control_points_old
    control_points_old = np.ascontiguousarray( np.zeros((trk_hdr['n_count'],n_control_points*3)).astype(np.float32) )
    ptr_CP_old = &control_points_old[0,0]

    # create the structure for the control points 
    cdef float* ptr_CP_new
    cdef float [:, ::1] control_points_new
    control_points_new = np.ascontiguousarray( np.zeros((trk_hdr['n_count'],n_control_points*3)).astype(np.float32) )
    ptr_CP_new = &control_points_new[0,0]


    #create the structure for the control points
    cdef float* ptrTRK
    cdef float [:, ::1] points_trk
    points_trk = np.ascontiguousarray( np.zeros((trk_hdr['n_count'],1000)).astype(np.float32) )
    ptrTRK = &points_trk[0,0]

    #exporting IC compartments
    spline2dictionary_IC( filename_trk, Nx, Ny, Nz, Px, Py, Pz, trk_hdr['n_count'], trk_hdr['n_scalars'], trk_hdr['n_properties'], fiber_shift, points_to_skip, 1 if do_intersect else 0, n_control_points, ptr_CP_old, ptr_CP_new, ptrMASK, ptrGMASK, ptrTDI, ptrTRK, path_out )

    #save trk file from spline
    size = 0;
    fib = []
    for i in range(trk_hdr['n_count']):
        for j in range(1000):
            if points_trk[i,j] == 0:
                size = (j / 3)
                break
        f = np.zeros((size, 3))
        for k in range(size):      
            f[k,:] = points_trk[i,k*3:k*3+3]
        fib.append((f,None,None))

    nibabel.trackvis.write( join(path_out,'fiber.trk'), fib, trk_hdr )

    #exporting EC compartments
    spline2dictionary_EC( ptrPEAKS, Np, vf_THR, -1 if flip_peaks[0] else 1, -1 if flip_peaks[1] else 1, -1 if flip_peaks[2] else 1, ptrTDI )

    #save control points
    np.save(join(path_out,'control_points_old'), control_points_old)
    np.save(join(path_out,'control_points_new'), control_points_new)

    #save TDI map with dilation
    niiTDI = nibabel.Nifti1Image( niiTDI_img, niiPEAKS.affine )
    nibabel.save( niiTDI, join(path_out,'dictionary_tdi.nii.gz') )


cpdef SA( normL2COMMIT, num_random, filename_trk, x, filename_peaks, filename_mask, filename_gmask, Ee_COMMIT, Em_COMMIT, path_out ):

    # fiber-tracts from .trk
    try :
        fib, trk_hdr = nibabel.trackvis.read( filename_trk, as_generator=True )
    except :
        raise IOError( 'Track file not found' )
    Nx = trk_hdr['dim'][0]
    Ny = trk_hdr['dim'][1]
    Nz = trk_hdr['dim'][2]

    # white-matter mask
    cdef float* ptrMASK
    cdef float [:, :, ::1] niiMASK_img
    if filename_mask is not None :
        niiMASK = nibabel.load( filename_mask )
        niiMASK_img = np.ascontiguousarray( niiMASK.get_data().astype(np.float32) )
        ptrMASK  = &niiMASK_img[0,0,0]
    else :
        print '\t\t* no mask specified to filter IC compartments'
        ptrMASK = NULL

    # gray-matter mask
    cdef float* ptrGMASK
    cdef float [:, :, ::1] niiGMASK_img
    if filename_gmask is not None :
        niiGMASK = nibabel.load( filename_gmask )
        niiGMASK_img = np.ascontiguousarray( niiGMASK.get_data().astype(np.float32) )
        ptrGMASK  = &niiGMASK_img[0,0,0]
    else :
        print '\t\t* no mask specified to filter IC compartments'
        ptrGMASK = NULL

    # x
    cdef float* ptrX
    cdef float [::1] X
    X =  x.astype(np.float32)   
    ptrX = &X[0]

    # Ee
    cdef float* ptrEe
    cdef float [:, :, :, ::1] Ee
    Ee =  np.ascontiguousarray( Ee_COMMIT.get_data().astype(np.float32) )
    ptrEe = &Ee[0,0,0,0]    

    # Em
    cdef float* ptrEm
    cdef float [:, :, :, ::1] Em
    Em =  np.ascontiguousarray( Em_COMMIT.get_data().astype(np.float32) )
    ptrEm = &Em[0,0,0,0] 

    # peaks file for EC contributions
    niiPEAKS = nibabel.load( filename_peaks )

    # create the structure for the control points
    cdef float* ptr_CP_old
    cdef float [:, ::1] control_points_trk_old
    control_points_trk_old = np.ascontiguousarray( np.load(join(path_out,'control_points_old.npy')).astype(np.float32) )
    ptr_CP_old = &control_points_trk_old[0,0]

    # create the structure for the control points
    cdef float* ptr_CP_new
    cdef float [:, ::1] control_points_trk_new
    control_points_trk_new = np.ascontiguousarray( np.load(join(path_out,'control_points_new.npy')).astype(np.float32) )
    ptr_CP_new = &control_points_trk_new[0,0]


    #create the structure for the control points
    cdef float* ptrTRK
    cdef float [:, ::1] points_trk
    points_trk = np.ascontiguousarray( np.zeros((trk_hdr['n_count'],5000)).astype(np.float32) )
    ptrTRK = &points_trk[0,0]

    simulated_annealing( normL2COMMIT, num_random, ptr_CP_old, ptr_CP_new, ptrTRK, ptrMASK, ptrGMASK, ptrEe, ptrEm, ptrX )

    # save control points
    np.save(join(path_out,'control_points_old'), control_points_trk_old)    
    np.save(join(path_out,'control_points_new'), control_points_trk_new)

    #save trk file from spline
    size = 0;
    fib = []
    for i in range(trk_hdr['n_count']):
        for j in range(5000):
            if points_trk[i,j] == 0:
                size = (j / 3)
                break
        f = np.zeros((size, 3))
        for k in range(size):      
            f[k,:] = points_trk[i,k*3:k*3+3]
        print ptrX[i]
        if ptrX[i] != 0:
            fib.append((f,None,None))
    #nibabel.trackvis.write( join(path_out,'fiberMOVE.trk'), fib, trk_hdr )


cpdef last_TRK( filename_trk, x_re_normalized, path_out ):

    # fiber-tracts from .trk
    print '\t\t* tractogram'
    try :
        _, trk_hdr = nibabel.trackvis.read( filename_trk, as_generator=True )
    except :
        raise IOError( 'Track file not found' )
    Nx = trk_hdr['dim'][0]
    Ny = trk_hdr['dim'][1]
    Nz = trk_hdr['dim'][2]

    # create the structure for the control points
    cdef float* ptr_CP_old
    cdef float [:, ::1] control_points_trk_old
    control_points_trk_old = np.ascontiguousarray( np.load(join(path_out,'control_points_old.npy')).astype(np.float32) )
    ptr_CP_old = &control_points_trk_old[0,0]

    #create the structure for the control points
    cdef float* ptrTRK
    cdef float [:, ::1] points_trk
    points_trk = np.ascontiguousarray( np.zeros((trk_hdr['n_count'],5000)).astype(np.float32) )
    ptrTRK = &points_trk[0,0]

    create_trk( ptr_CP_old, ptrTRK )

    #save trk file from spline
    size = 0;
    fib = []
    new_hdr = trk_hdr.copy()
    for i in range(trk_hdr['n_count']):
        for j in range(5000):
            if points_trk[i,j] == 0:
                size = (j / 3)
                break
        f = np.zeros((size, 3))
        for k in range(size):      
            f[k,:] = points_trk[i,k*3:k*3+3]
        print x_re_normalized[i]
        fib.append((f, np.ones((size,1))*x_re_normalized[i], None))
    new_hdr['scalar_name'][0] = 'weight_X'
    nibabel.trackvis.write( join(path_out,'fiberEND.trk'), fib, new_hdr )

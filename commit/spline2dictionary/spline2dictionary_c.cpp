#include "parameters.cpp"
#include "supportfuncs.cpp"

// =========================
// Function called by CYTHON
// =========================
int spline2dictionary_IC(char* strTRK, int Nx, int Ny, int Nz, float Px, float Py, float Pz, int n_fibTRK, int n_scalars, int n_properties, float fiber_shift, int points_to_skip, int c, int n_CP, float* ptr_CP_old, float* ptr_CP_new, float* _ptrMASK, float* _ptrGMASK, float* ptrTDI, float* ptrTRK, char* path_out ){

	float			fiber[3][MAX_FIB_LEN];
	float           fiberLen;
	unsigned int   	N, totICSegments = 0, totFibers = 0;
    std::map<segKey,float>::iterator it;

	//variables control_points
	std::vector<float> polyline;
	std::vector<float> control_points;

	std::vector<Vector<float> > PointFiber;	
	float		fiber_controlled[3][MAX_FIB_LEN];
	int index = 0;
	int numpy_ptr = 0, numpy_ptr2 = 0;
	int numpy_trk = 0;

	bool pen = false;

	Catmull		FIBER;

	// set global variables
    dim.Set( Nx, Ny, Nz );
    pixdim.Set( Px, Py, Pz );
    nPointsToSkip = points_to_skip;
    fiberShiftXmm = fiber_shift * pixdim.x; // shift in mm for the coordinates
    fiberShiftYmm = fiber_shift * pixdim.y;
    fiberShiftZmm = fiber_shift * pixdim.z;
    ptrMASK       = _ptrMASK;
    ptrGMASK       = _ptrGMASK;
    doIntersect   = c > 0;

	OUTPUT_path = path_out;
	strTRKfilename = strTRK;
	n_count = n_fibTRK;
	splinePts = n_CP;

	printf( "\t* Exporting IC compartments:\n" );
	
	FILE* fpTRK = fopen(strTRKfilename,"r+b");
    	if (fpTRK == NULL) return 0;
    	fseek(fpTRK,1000,SEEK_SET);		
	
	
	
	// open files
    	filename = OUTPUT_path+"/dictionary_IC_trkLen.dict";   FILE* pDict_IC_trkLen = fopen(filename.c_str(),"w");
    	if ( !pDict_IC_trkLen )
    	{
        	printf( "\n[trk2dictionary] Unable to create output files" );
        	return 0;
    	}
    	filename = OUTPUT_path+"/dictionary_IC_f.dict";        FILE* pDict_IC_f      = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_vx.dict";       FILE* pDict_IC_vx     = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_vy.dict";       FILE* pDict_IC_vy     = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_vz.dict";       FILE* pDict_IC_vz     = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_ox.dict";       FILE* pDict_IC_ox     = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_oy.dict";       FILE* pDict_IC_oy     = fopen(filename.c_str(),"w");
    	filename = OUTPUT_path+"/dictionary_IC_len.dict";      FILE* pDict_IC_len    = fopen(filename.c_str(),"w");

	// iterate over fibers
    	PROGRESS.reset( n_count );
    	PROGRESS.setPrefix("\t  ");
    	for(int f=0; f<n_count ;f++){
    	
		pen = false;		
   
		PROGRESS.inc();
        	N = read_fiber( fpTRK, fiber, n_scalars, n_properties );
 
		polyline.clear();
		control_points.clear();
 
		polyline.resize(3* N);
		index = 0;	
		for( int j = 0; j < N; j++) {			
			for(int k = 0; k< 3;k++) {
				polyline[index] = fiber[k][j];							
				index++;
			}
		}
		psimpl::simplify_douglas_peucker_n <3> (polyline.begin(), polyline.end(),splinePts, std::back_inserter (control_points));
		PointFiber.resize(control_points.size()/3);
  		
		index = 0;
		for( int j = 0; j < control_points.size(); j = j+3) {
			PointFiber[index].Set(control_points[j], control_points[j+1], control_points[j+2]);
			ptr_CP_old[numpy_ptr] = control_points[j];
			numpy_ptr++;
			ptr_CP_old[numpy_ptr] = control_points[j+1];
			numpy_ptr++;
			ptr_CP_old[numpy_ptr] = control_points[j+2];
			numpy_ptr++;		
			index++;
		}		
 
		index = 0;
		for( int j = 0; j < control_points.size(); j = j+3) {
			ptr_CP_new[numpy_ptr2] = control_points[j];
			numpy_ptr2++;
			ptr_CP_new[numpy_ptr2] = control_points[j+1];
			numpy_ptr2++;
			ptr_CP_new[numpy_ptr2] = control_points[j+2];
			numpy_ptr2++;		
			index++;
		}
 
		FIBER.set(PointFiber);
		FIBER.eval(FIBER.L/SEGMENT_len);
		FIBER.arcLengthReparametrization(SEGMENT_len);
 
		pen = CheckGM(FIBER);
		if (pen == false){
			std::cout<< "The fiber "<< f << " is outside the WM mask or the end points are otside the GM mask "<< std::endl;
		}
		for( int j=0; j<FIBER.P.size(); j++ ){
			fiber_controlled[0][j] = FIBER.P[j].x;
			fiber_controlled[1][j] = FIBER.P[j].y;
			fiber_controlled[2][j] = FIBER.P[j].z;
		}

		index = 0;
		for( int j = 0; j < 1000; j++) {
			if( j < FIBER.P.size() ){
				ptrTRK[numpy_trk] = FIBER.P[j].x;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].y;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].z;
				numpy_trk++;
			} else {
				ptrTRK[numpy_trk] = 0;
				numpy_trk++;
				
				if ( numpy_trk % 1000 == 0 )
					break;
			}
		}

		// create segments
		fiberForwardModel_SD( fiber_controlled, FIBER.P.size() );
		if ( FiberSegments_SD.size() > 0 ){
			   		
	    		// store data to files
            		fiberLen = 0;
            		for (it=FiberSegments_SD.begin(); it!=FiberSegments_SD.end(); it++)
            		{
                		fwrite( &totFibers,      4, 1, pDict_IC_f );
                		fwrite( &(it->first.x),  1, 1, pDict_IC_vx );
                		fwrite( &(it->first.y),  1, 1, pDict_IC_vy );
                		fwrite( &(it->first.z),  1, 1, pDict_IC_vz );
                		fwrite( &(it->first.ox), 1, 1, pDict_IC_ox );
                		fwrite( &(it->first.oy), 1, 1, pDict_IC_oy );
                		fwrite( &(it->second),   4, 1, pDict_IC_len );
                		ptrTDI[ it->first.z + dim.z * ( it->first.y + dim.y * it->first.x ) ] += it->second;
                		fiberLen += it->second;
            		}
            		fwrite( &fiberLen,  1, 4, pDict_IC_trkLen );
            		totICSegments += FiberSegments_SD.size();
            		totFibers++;
		}
	}
    	PROGRESS.close();

	FiberSegments_SD.clear();

    	fclose( fpTRK );
    	fclose( pDict_IC_trkLen );
    	fclose( pDict_IC_f );
    	fclose( pDict_IC_vx );
    	fclose( pDict_IC_vy );
    	fclose( pDict_IC_vz );
    	fclose( pDict_IC_ox );
    	fclose( pDict_IC_oy );
    	fclose( pDict_IC_len );

    	printf("\t  [ %d fibers, %d segments ]\n", totFibers, totICSegments );
}


// =========================
// Function called by CYTHON
// =========================
int spline2dictionary_EC( float* ptrPEAKS, int Np, float vf_THR, int ECix, int ECiy, int ECiz, float* ptrTDI ){
	/*=========================*/
	/*     EC compartments     */
	/*=========================*/
	
    	PROGRESS.reset( dim.z );

	unsigned int totECSegments = 0, totECVoxels = 0;

	printf( "\t* Exporting EC compartments:\n" );

	filename = OUTPUT_path+"/dictionary_EC_vx.dict";       FILE* pDict_EC_vx  = fopen( filename.c_str(),   "w" );
	filename = OUTPUT_path+"/dictionary_EC_vy.dict";       FILE* pDict_EC_vy  = fopen( filename.c_str(),   "w" );
	filename = OUTPUT_path+"/dictionary_EC_vz.dict";       FILE* pDict_EC_vz  = fopen( filename.c_str(),   "w" );
	filename = OUTPUT_path+"/dictionary_EC_ox.dict";       FILE* pDict_EC_ox  = fopen( filename.c_str(),   "w" );
	filename = OUTPUT_path+"/dictionary_EC_oy.dict";       FILE* pDict_EC_oy  = fopen( filename.c_str(),   "w" );

	if ( ptrPEAKS != NULL )
	{
	Vector<double> dir;
	double         longitude, colatitude;
	segKey         ec_seg;
	int            ix, iy, iz, id, atLeastOne;
	float          peakMax;
	float          norms[ Np ];
	float          *ptr;

	for(iz=0; iz<dim.z ;iz++)
	{
	    PROGRESS.inc();
	    for(iy=0; iy<dim.y ;iy++)
	    for(ix=0; ix<dim.x ;ix++)
	    {
		// check if in mask previously computed from IC segments
		if ( ptrTDI[ iz + dim.z * ( iy + dim.y * ix ) ] == 0 ) continue;

		peakMax = -1;
		for(id=0; id<Np ;id++)
		{
		    ptr = ptrPEAKS + 3*(id + Np * ( iz + dim.z * ( iy + dim.y * ix ) ));
		    dir.x = ptr[0];
		    dir.y = ptr[1];
		    dir.z = ptr[2];
		    norms[id] = dir.norm();
		    if ( norms[id] > peakMax )
			peakMax = norms[id];
		}

		if ( peakMax > 0 )
		{
		    ec_seg.x  = ix;
		    ec_seg.y  = iy;
		    ec_seg.z  = iz;
		    atLeastOne = 0;
		    for(id=0; id<Np ;id++)
		    {
			if ( norms[id]==0 || norms[id] < vf_THR*peakMax ) continue; // peak too small, don't consider it

			// store this orientation (invert axes if needed)
			ptr = ptrPEAKS + 3*(id + Np * ( iz + dim.z * ( iy + dim.y * ix ) ));
			dir.x = ECix * ptr[0];
			dir.y = ECiy * ptr[1];
			dir.z = ECiz * ptr[2];
			if ( dir.y < 0 )
			{
			    // ensure to be in the right hemisphere (the one where kernels were pre-computed)
			    dir.x = -dir.x;
			    dir.y = -dir.y;
			    dir.z = -dir.z;
			}
			colatitude = atan2( sqrt(dir.x*dir.x + dir.y*dir.y), dir.z );
			longitude  = atan2( dir.y, dir.x );
			ec_seg.ox = (int)round(colatitude/M_PI*180.0);
			ec_seg.oy = (int)round(longitude/M_PI*180.0);
			fwrite( &ec_seg.x,   1, 1, pDict_EC_vx );
			fwrite( &ec_seg.y,   1, 1, pDict_EC_vy );
			fwrite( &ec_seg.z,   1, 1, pDict_EC_vz );
			fwrite( &ec_seg.ox,  1, 1, pDict_EC_ox );
			fwrite( &ec_seg.oy,  1, 1, pDict_EC_oy );
			totECSegments++;
			atLeastOne = 1;
		    }
		    if ( atLeastOne>0 )
			totECVoxels++;
		}
	    }
	}
	PROGRESS.close();
	}

	fclose( pDict_EC_vx );
	fclose( pDict_EC_vy );
	fclose( pDict_EC_vz );
	fclose( pDict_EC_ox );
	fclose( pDict_EC_oy );

	printf("\t  [ %d voxels, %d segments ]\n", totECVoxels, totECSegments );
	
	return 1;
	
}

// =========================
// Function called by CYTHON
// =========================

int simulated_annealing( float normL2COMMIT, float num_random, float* ptr_CP_old, float* ptr_CP_new, float* ptrTRK, float* _ptrMASK, float* _ptrGMASK, float* ptrEe, float* ptrEm, float* ptrX ){

	float		fiber[3][MAX_FIB_LEN];
	float           fiberLen;
	unsigned int   	N, totICSegments = 0, totFibers = 0;
    	std::map<segKey,float>::iterator it;

	ptrMASK       = _ptrMASK;
	ptrGMASK       = _ptrGMASK;

	std::vector<Vector<float> > PointFiber;	
	float	fiber_controlled[3][MAX_FIB_LEN];

	Catmull			FIBER;

	int numpy_trk = 0;

	if ( normL2 == -1 )
		normL2 = normL2COMMIT;

	filename = OUTPUT_path+"/L2_compare.txt";        		FILE* L2_file            = fopen(filename.c_str(),"a");
	fprintf( L2_file,  "%f\t%f\n", normL2COMMIT,normL2 );	
	fclose( L2_file );

	if ( normL2COMMIT <= normL2 ){
		for (int i=0; i< n_count*splinePts*3; i++)
			ptr_CP_old[i] = ptr_CP_new[i];
		normL2 = normL2COMMIT;	
	}

	printf( "-> Moving fibers\n" );
	Move_fiber( ptr_CP_old, ptr_CP_new, num_random );	
	
	totICSegments = 0;
	totFibers = 0;

	// open files
    filename = OUTPUT_path+"/dictionary_IC_trkLen.dict";   FILE* pDict_IC_trkLen = fopen(filename.c_str(),"w");
    if ( !pDict_IC_trkLen ){
        	printf( "\n[trk2dictionary] Unable to create output files" );
        	return 0;
    }

    filename = OUTPUT_path+"/dictionary_IC_f.dict";         FILE* pDict_IC_f      = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vx.dict";        FILE* pDict_IC_vx     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vy.dict";        FILE* pDict_IC_vy     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vz.dict";        FILE* pDict_IC_vz     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_ox.dict";        FILE* pDict_IC_ox     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_oy.dict";        FILE* pDict_IC_oy     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_len.dict";       FILE* pDict_IC_len    = fopen(filename.c_str(),"w");

	// iterate over fibers
    PROGRESS.reset( n_count );
    PROGRESS.setPrefix("\t  ");

	PointFiber.resize(splinePts);

    for(int f=0; f<n_count ;f++){
		        	
		PROGRESS.inc();	
		for( int j = 0; j < splinePts*3; j = j+3) {
			PointFiber[j/3].Set(ptr_CP_new[f*splinePts*3+j], ptr_CP_new[f*splinePts*3+j+1], ptr_CP_new[f*splinePts*3+j+2]);		
		}		
				
		FIBER.set(PointFiber);
		FIBER.eval(FIBER.L/SEGMENT_len);
		FIBER.arcLengthReparametrization(SEGMENT_len);
		
		for( int j = 0; j < 5000; j++) {
			if( j < FIBER.P.size() ){
				ptrTRK[numpy_trk] = FIBER.P[j].x;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].y;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].z;
				numpy_trk++;
			} 
			else {
				ptrTRK[numpy_trk] = 0;
				numpy_trk++;
				
				if ( numpy_trk % 5000 == 0 )
					break;
			}
		}	

		for( int j=0; j<FIBER.P.size(); j++ ){
			fiber_controlled[0][j] = FIBER.P[j].x;
			fiber_controlled[1][j] = FIBER.P[j].y;
			fiber_controlled[2][j] = FIBER.P[j].z;
		}

		// create segments
		fiberForwardModel_D( fiber_controlled, FIBER.P.size() );
		if ( FiberSegments_D.size() > 0 ){
			   		
	    	fiberLen = 0;
            for (it=FiberSegments_D.begin(); it!=FiberSegments_D.end(); it++){
                fwrite( &totFibers,      4, 1, pDict_IC_f );
                fwrite( &(it->first.x),  1, 1, pDict_IC_vx );
                fwrite( &(it->first.y),  1, 1, pDict_IC_vy );
                fwrite( &(it->first.z),  1, 1, pDict_IC_vz );
                fwrite( &(it->first.ox), 1, 1, pDict_IC_ox );
                fwrite( &(it->first.oy), 1, 1, pDict_IC_oy );
                fwrite( &(it->second),   4, 1, pDict_IC_len );
                fiberLen += it->second;
            }
            fwrite( &fiberLen,  1, 4, pDict_IC_trkLen );
            totICSegments += FiberSegments_D.size();
            totFibers++;
		}
	}
    
	PROGRESS.close();
	FiberSegments_D.clear();

   	fclose( pDict_IC_trkLen );
   	fclose( pDict_IC_f );
   	fclose( pDict_IC_vx );
   	fclose( pDict_IC_vy );
   	fclose( pDict_IC_vz );
   	fclose( pDict_IC_ox );
   	fclose( pDict_IC_oy );
   	fclose( pDict_IC_len );

	return 1;
}

// =========================
// Function called by CYTHON
// =========================

int create_trk( float* ptr_CP_old, float* ptrTRK ){

	float		fiber[3][MAX_FIB_LEN];
	float           fiberLen;
	unsigned int   	N, totICSegments = 0, totFibers = 0;
   	std::map<segKey,float>::iterator it;

	std::vector<Vector<float> > PointFiber;	
	float	fiber_controlled[3][MAX_FIB_LEN];

	Catmull			FIBER;

	int numpy_trk = 0;

	totICSegments = 0;
	totFibers = 0;

	// open files
   	filename = OUTPUT_path+"/dictionary_IC_trkLen.dict";   FILE* pDict_IC_trkLen = fopen(filename.c_str(),"w");
   	if ( !pDict_IC_trkLen ){
        printf( "\n[trk2dictionary] Unable to create output files" );
        return 0;
    }

    filename = OUTPUT_path+"/dictionary_IC_f.dict";         FILE* pDict_IC_f      = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vx.dict";        FILE* pDict_IC_vx     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vy.dict";        FILE* pDict_IC_vy     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_vz.dict";        FILE* pDict_IC_vz     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_ox.dict";        FILE* pDict_IC_ox     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_oy.dict";        FILE* pDict_IC_oy     = fopen(filename.c_str(),"w");
    filename = OUTPUT_path+"/dictionary_IC_len.dict";       FILE* pDict_IC_len    = fopen(filename.c_str(),"w");

	printf( "\t* Exporting IC compartments:\n" );

	// iterate over fibers
   	PROGRESS.reset( n_count );
   	PROGRESS.setPrefix("\t  ");

	PointFiber.resize(splinePts);

   	for(int f=0; f<n_count ;f++){
		        	
		PROGRESS.inc();
 		
		for( int j = 0; j < splinePts*3; j = j+3) {
			PointFiber[j/3].Set(ptr_CP_old[f*splinePts*3+j], ptr_CP_old[f*splinePts*3+j+1], ptr_CP_old[f*splinePts*3+j+2]);		
		}		
				
		FIBER.set(PointFiber);
		FIBER.eval(FIBER.L/SEGMENT_len);
		FIBER.arcLengthReparametrization(SEGMENT_len);
		
		for( int j = 0; j < 5000; j++) {
			if( j < FIBER.P.size() ){
				ptrTRK[numpy_trk] = FIBER.P[j].x;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].y;
				numpy_trk++;
				ptrTRK[numpy_trk] = FIBER.P[j].z;
				numpy_trk++;
			} else {
				ptrTRK[numpy_trk] = 0;
				numpy_trk++;
				
				if ( numpy_trk % 5000 == 0 )
					break;
			}
		}	

		for( int j=0; j<FIBER.P.size(); j++ ){
			fiber_controlled[0][j] = FIBER.P[j].x;
			fiber_controlled[1][j] = FIBER.P[j].y;
			fiber_controlled[2][j] = FIBER.P[j].z;
		}

		// create segments
		fiberForwardModel_D( fiber_controlled, FIBER.P.size() );
		if ( FiberSegments_D.size() > 0 ){
			   		
	    	fiberLen = 0;
            for (it=FiberSegments_D.begin(); it!=FiberSegments_D.end(); it++){
                fwrite( &totFibers,      4, 1, pDict_IC_f );
                fwrite( &(it->first.x),  1, 1, pDict_IC_vx );
                fwrite( &(it->first.y),  1, 1, pDict_IC_vy );
                fwrite( &(it->first.z),  1, 1, pDict_IC_vz );
                fwrite( &(it->first.ox), 1, 1, pDict_IC_ox );
                fwrite( &(it->first.oy), 1, 1, pDict_IC_oy );
                fwrite( &(it->second),   4, 1, pDict_IC_len );
                fiberLen += it->second;
            }
            fwrite( &fiberLen,  1, 4, pDict_IC_trkLen );
            totICSegments += FiberSegments_D.size();
            totFibers++;
		}
	}

  	PROGRESS.close();
	FiberSegments_D.clear();

   	fclose( pDict_IC_trkLen );
   	fclose( pDict_IC_f );
   	fclose( pDict_IC_vx );
   	fclose( pDict_IC_vy );
   	fclose( pDict_IC_vz );
   	fclose( pDict_IC_ox );
   	fclose( pDict_IC_oy );
   	fclose( pDict_IC_len );

   	printf("\t  [ %d fibers, %d segments ]\n", totFibers, totICSegments );

	return 1;
	
}

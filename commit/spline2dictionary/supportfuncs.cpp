// CLASS to store the segments of one fiber
class segKey
{
    public:
    unsigned char x, y, z, ox, oy;
    segKey(){}

    void set(unsigned char _x, unsigned char _y, unsigned char _z, unsigned char _ox, unsigned char _oy){
        x  = _x;
        y  = _y;
        z  = _z;
        ox = _ox;
        oy = _oy;
    }

    bool const operator <(const segKey& o) const{
        return oy<o.oy || (oy==o.oy && ox<o.ox) || (oy==o.oy && ox==o.ox && z<o.z) || (oy==o.oy && ox==o.ox && z==o.z && y<o.y) || (oy==o.oy && ox==o.ox && z==o.z && y==o.y && x<o.x);
    }

    bool const operator ==(const segKey& o) const{
        return oy<o.oy && ox==o.ox && z==o.z && y==o.y && x==o.x;
    }
};


// global variables (to avoid passing them at each call)
std::map<segKey,float> 	FiberSegments_SD;
std::map<segKey,float> 	FiberSegments_D;

Vector<int>     	dim;
Vector<float>   	pixdim;
float*          	ptrMASK;
float*          	ptrGMASK;
unsigned int    	nPointsToSkip;
float           	fiberShiftXmm, fiberShiftYmm, fiberShiftZmm;
bool            	doIntersect;


void fiberForwardModel_SD( float fiber[3][MAX_FIB_LEN], unsigned int pts );
void segmentForwardModel_SD( const Vector<double>& P1, const Vector<double>& P2 );
void fiberForwardModel_D( float fiber[3][MAX_FIB_LEN], unsigned int pts );
void segmentForwardModel_D( const Vector<double>& P1, const Vector<double>& P2 );

bool rayBoxIntersection( Vector<double>& origin, Vector<double>& direction, Vector<double>& vmin, Vector<double>& vmax, double & t);
unsigned int read_fiber( FILE* fp, float fiber[3][MAX_FIB_LEN], int ns, int np );


void Move_fiber( float* ptr_CP_old, float* ptr_CP_new, float num_random );
inline bool Alter_fiber( Catmull&, Catmull& );
inline bool CheckWM( const Catmull& Fiber );
inline bool CheckGM( const Catmull& Fiber );

/********************************************************************************************************************/
/*                                                 fiberForwardModel_SD                                                */
/********************************************************************************************************************/
void fiberForwardModel_SD( float fiber[3][MAX_FIB_LEN], unsigned int pts ){

   	static Vector<double> S1, S2, S1m, S2m, P;
   	static Vector<double> vox, vmin, vmax, dir;
   	static double         len, t;
   	static int            i, j, k;

   	FiberSegments_SD.clear();
   	for(i=nPointsToSkip; i<pts-1-nPointsToSkip ;i++){
		
		// original segment to be processed
		S1.Set( fiber[0][i]   + fiberShiftXmm, fiber[1][i]   + fiberShiftYmm, fiber[2][i]   + fiberShiftZmm );
        S2.Set( fiber[0][i+1] + fiberShiftXmm, fiber[1][i+1] + fiberShiftYmm, fiber[2][i+1] + fiberShiftZmm );
	
        // get a normal to the vector to move
        dir.x = S2.x-S1.x;
        dir.y = S2.y-S1.y;
        dir.z = S2.z-S1.z;
        dir.Normalize();
		
        if ( doIntersect==false ){          		
			segmentForwardModel_SD( S1, S2 );
		}        	
		else{
    		while( 1 ){
				len = sqrt( pow(S2.x-S1.x,2) + pow(S2.y-S1.y,2) + pow(S2.z-S1.z,2) ); // in mm
			       		
				if ( len < 1e-3 )
            		break;

        		// compute AABB of the first point (in mm)
        		vmin.x = floor( (S1.x + 1e-6*dir.x)/pixdim.x ) * pixdim.x;
        		vmin.y = floor( (S1.y + 1e-6*dir.y)/pixdim.y ) * pixdim.y;
        		vmin.z = floor( (S1.z + 1e-6*dir.z)/pixdim.z ) * pixdim.z;
        		vmax.x = vmin.x + pixdim.x;
        		vmax.y = vmin.y + pixdim.y;
        		vmax.z = vmin.z + pixdim.z;
        		
				if ( rayBoxIntersection( S1, dir, vmin, vmax, t ) && t>0 && t<len ){

            		// add the portion S1P, and then reiterate
            		P.Set( S1.x + t*dir.x, S1.y + t*dir.y, S1.z + t*dir.z );

            		segmentForwardModel_SD( S1, P );
            		S1.Set( P.x, P.y, P.z );
        		}
        		else{
			
            		// add the segment S1S2 and stop iterating
            		segmentForwardModel_SD( S1, S2 );
            			break;
        		}
    		}
		}
    }
}


/********************************************************************************************************************/
/*                                                segmentForwardModel_SD                                              */
/********************************************************************************************************************/
void segmentForwardModel_SD( const Vector<double>& P1, const Vector<double>& P2 ){

    static Vector<int>    vox;
    static Vector<double> dir, dirTrue;
    static double         longitude, colatitude, len;
    static segKey         key;


    // direction of the segment
    dir.y = P2.y-P1.y;
    if ( dir.y >= 0 ){
        dir.x = P2.x-P1.x;
        dir.z = P2.z-P1.z;
    }
    else{
        dir.x = P1.x-P2.x;
        dir.y = P1.y-P2.y;
        dir.z = P1.z-P2.z;
    }

    // length of segment
    len = dir.norm();
    if ( len<1e-4 ) 
		return; // in mm
    dir.Normalize();

    // voxel of the segment is the centroid
    vox.x = floor( 0.5 * (P1.x + P2.x) / pixdim.x );
    vox.y = floor( 0.5 * (P1.y + P2.y) / pixdim.y );
    vox.z = floor( 0.5 * (P1.z + P2.z) / pixdim.z );
    
    if ( vox.x>=dim.x || vox.x<0 || vox.y>=dim.y || vox.y<0 || vox.z>=dim.z || vox.z<0 )
        return;

    if ( ptrMASK && ptrMASK[ vox.z + dim.z * ( vox.y + dim.y * vox.x ) ]==0 )
        return;

    // add the segment to the data structure
    longitude  = atan2(dir.y, dir.x);
    colatitude = atan2( sqrt(dir.x*dir.x + dir.y*dir.y), dir.z );
    key.set( vox.x, vox.y, vox.z, (int)round(colatitude/M_PI*180.0), (int)round(longitude/M_PI*180.0) );
    FiberSegments_SD[key] += len;
}

/********************************************************************************************************************/
/*                                                 fiberForwardModel                                                */
/********************************************************************************************************************/
void fiberForwardModel_D( float fiber[3][MAX_FIB_LEN], unsigned int pts ){

	static Vector<double> S1, S2, S1m, S2m, P;
  	static Vector<double> vox, vmin, vmax, dir;
   	static double         len, t;
   	static int            i, j, k;
	
   	FiberSegments_D.clear();
   	for(i=nPointsToSkip; i<pts-1-nPointsToSkip ;i++){
        	
		// original segment to be processed
        S1.Set( fiber[0][i]   + fiberShiftXmm, fiber[1][i]   + fiberShiftYmm, fiber[2][i]   + fiberShiftZmm );
        S2.Set( fiber[0][i+1] + fiberShiftXmm, fiber[1][i+1] + fiberShiftYmm, fiber[2][i+1] + fiberShiftZmm );
	
       	// get a normal to the vector to move
       	dir.x = S2.x-S1.x;
       	dir.y = S2.y-S1.y;
       	dir.z = S2.z-S1.z;
       	dir.Normalize();
		
       	if ( doIntersect==false ){          		
			segmentForwardModel_D( S1, S2 );
		}        	
		else{
    		while( 1 ){
				len = sqrt( pow(S2.x-S1.x,2) + pow(S2.y-S1.y,2) + pow(S2.z-S1.z,2) ); // in mm
			       		
				if ( len < 1e-3 )
            		break;

        		// compute AABB of the first point (in mm)
        		vmin.x = floor( (S1.x + 1e-6*dir.x)/pixdim.x ) * pixdim.x;
        		vmin.y = floor( (S1.y + 1e-6*dir.y)/pixdim.y ) * pixdim.y;
        		vmin.z = floor( (S1.z + 1e-6*dir.z)/pixdim.z ) * pixdim.z;
        		vmax.x = vmin.x + pixdim.x;
        		vmax.y = vmin.y + pixdim.y;
        		vmax.z = vmin.z + pixdim.z;
        		
				if ( rayBoxIntersection( S1, dir, vmin, vmax, t ) && t>0 && t<len ){
            		// add the portion S1P, and then reiterate
            		P.Set( S1.x + t*dir.x, S1.y + t*dir.y, S1.z + t*dir.z );

            		segmentForwardModel_D( S1, P );
            		S1.Set( P.x, P.y, P.z );
        		}
        		else{
			        // add the segment S1S2 and stop iterating
            		segmentForwardModel_D( S1, S2 );
            		break;
        		}
			}
    	}
    }
}


/********************************************************************************************************************/
/*                                                segmentForwardModel                                               */
/********************************************************************************************************************/
void segmentForwardModel_D( const Vector<double>& P1, const Vector<double>& P2 ){

    static Vector<int>    vox;
    static Vector<double> dir, dirTrue;
    static double         longitude, colatitude, len;
    static segKey         key;


    // direction of the segment
    dir.y = P2.y-P1.y;
    if ( dir.y >= 0 ){
        dir.x = P2.x-P1.x;
        dir.z = P2.z-P1.z;
    }
    else{
        dir.x = P1.x-P2.x;
        dir.y = P1.y-P2.y;
        dir.z = P1.z-P2.z;
    }

    // length of segment
    len = dir.norm();
    if ( len<1e-4 ) 
		return; // in mm
    dir.Normalize();

    // voxel of the segment is the centroid
    vox.x = floor( 0.5 * (P1.x + P2.x) / pixdim.x );
    vox.y = floor( 0.5 * (P1.y + P2.y) / pixdim.y );
    vox.z = floor( 0.5 * (P1.z + P2.z) / pixdim.z );
    
    if ( vox.x>=dim.x || vox.x<0 || vox.y>=dim.y || vox.y<0 || vox.z>=dim.z || vox.z<0 )
        return;

    if ( ptrMASK && ptrMASK[ vox.z + dim.z * ( vox.y + dim.y * vox.x ) ]==0 )
        return;

    // add the segment to the data structure
    longitude  = atan2(dir.y, dir.x);
    colatitude = atan2( sqrt(dir.x*dir.x + dir.y*dir.y), dir.z );
    key.set( vox.x, vox.y, vox.z, (int)round(colatitude/M_PI*180.0), (int)round(longitude/M_PI*180.0) );
    FiberSegments_D[key] += len;
}

/********************************************************************************************************************/
/*                                                rayBoxIntersection                                                */
/********************************************************************************************************************/
bool rayBoxIntersection( Vector<double>& origin, Vector<double>& direction, Vector<double>& vmin, Vector<double>& vmax, double & t){

    static double tmin, tmax, tymin, tymax, tzmin, tzmax;
    static Vector<double> invrd;

    // inverse direction to catch float problems
    invrd.x = 1.0 / direction.x;
    invrd.y = 1.0 / direction.y;
    invrd.z = 1.0 / direction.z;


    if (invrd.x >= 0){
      tmin = (vmin.x - origin.x) * invrd.x;
      tmax = (vmax.x - origin.x) * invrd.x;
    }
    else{
      tmin = (vmax.x - origin.x) * invrd.x;
      tmax = (vmin.x - origin.x) * invrd.x;
    }

    if (invrd.y >= 0){
      tymin = (vmin.y - origin.y) * invrd.y;
      tymax = (vmax.y - origin.y) * invrd.y;
    }
    else{
      tymin = (vmax.y - origin.y) * invrd.y;
      tymax = (vmin.y - origin.y) * invrd.y;
    }

    if ( (tmin > tymax) || (tymin > tmax) ) 
		return false;
    if ( tymin > tmin) 
		tmin = tymin;
    if ( tymax < tmax) 
		tmax = tymax;

    if (invrd.z >= 0){
      	tzmin = (vmin.z - origin.z) * invrd.z;
      	tzmax = (vmax.z - origin.z) * invrd.z;
    }
	else{
      	tzmin = (vmax.z - origin.z) * invrd.z;
      	tzmax = (vmin.z - origin.z) * invrd.z;
    }

    if ( (tmin > tzmax) || (tzmin > tmax) ) 
		return false;
    if ( tzmin > tmin) 
		tmin = tzmin;
    if ( tzmax < tmax) 
		tmax = tzmax;

    // check if values are valid
    t = tmin;
    if (t <= 0) 
		t = tmax;

    return true;
}


/********************************************************************************************************************/
/*                      Read a fiber from file and return the number of points                                      */
/********************************************************************************************************************/ 
unsigned int read_fiber( FILE* fp, float fiber[3][MAX_FIB_LEN], int ns, int np )
{
    int N;
    fread((char*)&N, 1, 4, fp);

    if ( N >= MAX_FIB_LEN || N <= 0 )
        return 0;

    float tmp[3];
    for(int i=0; i<N; i++){
        fread((char*)tmp, 1, 12, fp);
        fiber[0][i] = tmp[0];
        fiber[1][i] = tmp[1];
        fiber[2][i] = tmp[2];
        fseek(fp,4*ns,SEEK_CUR);
    }
    fseek(fp,4*np,SEEK_CUR);

    return N;			
}
	

/********************************************************************************************************************/
/*                                               simulated annealing                                                */
/********************************************************************************************************************/ 

void Move_fiber( float* ptr_CP_old, float* ptr_CP_new, float num_random ){
	
	int		iF;
	int 	countKNOTs;
	bool    tryAlter = false;			
	
	std::vector<Vector<float> > PointFiber;
	Catmull			FIBER, FIBER_copy;

	while(	tryAlter == false ){
		//Randomly select a fiber to alter  
		iF = floor( n_count*num_random);	
		std::cout << "iF : " << iF << std::endl;

		PointFiber.resize(splinePts);
		
		for( int j = 0; j < splinePts*3; j = j+3) {
			PointFiber[j/3].Set(ptr_CP_old[iF*splinePts*3+j], ptr_CP_old[iF*splinePts*3+j+1], ptr_CP_old[iF*splinePts*3+j+2]);		
		}
		
		FIBER.set(PointFiber);
		FIBER.eval(FIBER.L/SEGMENT_len);
		FIBER.arcLengthReparametrization(SEGMENT_len);
		
		FIBER_copy = FIBER;	
	
		tryAlter = Alter_fiber( FIBER, FIBER_copy );		
	}	

	countKNOTs = 0;
	for( int j = iF*splinePts*3; j < iF*splinePts*3+splinePts*3; j++) {
		if (j % 3 == 0)
			ptr_CP_new[j] =  FIBER.KNOTs[countKNOTs+1].x;
		if (j % 3 == 1)
			ptr_CP_new[j] =  FIBER.KNOTs[countKNOTs+1].y;
		if (j % 3 == 2){
			ptr_CP_new[j] =  FIBER.KNOTs[countKNOTs+1].z;
			countKNOTs++;
		}
	}
}


/********************************************************************************************************************/
/*                                                  alter a fiber                                                   */
/********************************************************************************************************************/ 
inline bool Alter_fiber( Catmull& s, Catmull& o ) {	
	// This function is called when we want to alter a fiber. 
	// CUrrently three types of movement: 1)translate a spline, 2) move ALL controlpoints 3) move one controlpoint

	ranlib::Uniform<float>		uniformGen;
	ranlib::NormalUnit<float>	normalGen;

	// Generate random numbers from 0 to 1
	float proposal;

	Vector<float> delta;
	float magnitude;

	bool pen = false;
	int count = 0;

	while ( pen == false && count < 100 ) {

		count++;
		proposal = uniformGen.random();
		
		delta.Set( MOVE_sigma*normalGen.random(), MOVE_sigma*normalGen.random(), MOVE_sigma*normalGen.random() );
		magnitude = MOVE_sigma*normalGen.random();

		if(proposal < .33) { 
			s.Translate(0, s.KNOTs.size()-1, delta, 1);	
		}
		else if ( proposal >= .33 && proposal <=1) {
			for( int i = 0; i < s.KNOTs.size();i++) {
				if( i == 0 || i == s.KNOTs.size()-2) {
					s.MoveKnot(i, delta, magnitude);
					s.MoveKnot(i+1, delta, magnitude);
					i++;
				} else {
					s.MoveKnot(i, delta, magnitude);
				}
			}
		}else {
			int knot = floor((static_cast <float> (rand()) / static_cast <float> (RAND_MAX))*(s.KNOTs.size()-2)+1);
			if( knot == 1) {
					s.MoveKnot(knot, delta, magnitude);
					s.MoveKnot(knot-1, delta, magnitude);
				} else if (knot = s.KNOTs.size()-2){
					s.MoveKnot(knot, delta, magnitude);
					s.MoveKnot(knot+1, delta, magnitude);
				} else
					s.MoveKnot(knot, delta, magnitude);
		}
		s.eval( s.L / SEGMENT_len );						
		s.arcLengthReparametrization( SEGMENT_len );

		pen = CheckGM(s);
		if (pen == false) {	// To make sure that we dont stray away from correct area
			s = o;				
		}
	}
	return pen;
}

/********************************************************************************************************************/
/*                                                  Check WM mask                                                   */
/********************************************************************************************************************/
inline bool CheckWM( const Catmull& Fiber ){

	int Vx, Vy, Vz;
	bool Pe = true;
	bool fiberStatus = false;	//control that when the fiber start to be white matter it should be to the first half of the fiber

	for( int i=1; i<Fiber.P.size()-1;i++ ) {
		Vx = floor( Fiber.P[i].x / pixdim.x );				
        Vy = floor( Fiber.P[i].y / pixdim.y );
        Vz = floor( Fiber.P[i].z / pixdim.z );

		if ( Vx < 0 || Vy < 0 || Vz < 0 )		// Penalize when outside the WM
			Pe = false;
		else
			if( i < Fiber.P.size()/2 )
				if ( ptrMASK[ Vz + dim.z * ( Vy + dim.y * Vx ) ] != 1 )
					if( fiberStatus == true )
						Pe = false;
				else
					if( fiberStatus == false )
						fiberStatus == true;
			else
				if ( ptrMASK[ Vz + dim.z * ( Vy + dim.y * Vx ) ] != 1 )
					if( fiberStatus == false )
						fiberStatus = true;
				else
					if( fiberStatus == true )
						Pe = false;
	}
	return Pe;
}


/********************************************************************************************************************/
/*                                                  Check GM mask                                                   */
/********************************************************************************************************************/
inline bool CheckGM( const Catmull& Fiber ){
	
	int Vx1, Vy1, Vz1, Vx2, Vy2, Vz2;

	Vx1 = floor( Fiber.KNOTs[0].x / pixdim.x );
   	Vy1 = floor( Fiber.KNOTs[0].y / pixdim.y );
	Vz1 = floor( Fiber.KNOTs[0].z / pixdim.z );
   	Vx2 = floor( Fiber.KNOTs.back().x / pixdim.x );
   	Vy2 = floor( Fiber.KNOTs.back().y / pixdim.y );
   	Vz2 = floor( Fiber.KNOTs.back().z / pixdim.z );

   	if( Vx1 == Vx2 && Vy1 == Vy2 && Vz1 == Vz2 )    		
		return false;

	if ( ptrGMASK[ Vz1 + dim.z * ( Vy1 + dim.y * Vx1 ) ] == 0 )
		return false;

	if ( ptrGMASK[ Vz2 + dim.z * ( Vy2 + dim.y * Vx2 ) ] == 0 )
		return false;

	return CheckWM( Fiber );
}

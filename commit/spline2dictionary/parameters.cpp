#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <string>
#include <map>
#include <random/uniform.h>
#include <random/normal.h>
#include "Catmull.h"
#include "psimpl.h"
#include "Vector.h"
#include "ProgressBar.h"
#define MAX_FIB_LEN 10000
#define SEGMENT_len 0.1		// desired length for each segment 


float			start_weight=0.005;

// alter a fiber
float                 	MOVE_sigma = 1;        		// move knots as N(0,s^2)  -- 3
float 			moveprob = 1;
float 			addprob	= 0; //0.25
float 			killprob = 0; //0.05

int 			Ns = 91;

char* 			strTRKfilename;
int 			n_count;
int			splinePts;

std::string    		filename;
std::string    		OUTPUT_path;
ProgressBar 		PROGRESS( 0 );

float normL2 = -1;


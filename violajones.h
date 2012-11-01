/*
##############################################################################
## THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
## OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, 
## ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, 
## INNOVATIVE OR RELEVANT NATURE, FITNESS FOR A PARTICULAR PURPOSE OR 
## COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.
## In the event of publication, the following notice is applicable:
##
##              (C) COPYRIGHT 2010 THALES RESEARCH & TECHNOLOGY
##                            ALL RIGHTS RESERVED
##              (C) COPYRIGHT 2012 Universitat Politècnica de Catalunya
##                            ALL RIGHTS RESERVED
##
## The entire notice above must be reproduced on all authorized copies.
##
##
## Title:             violajones.h
##
## File:              header file
## Authors:           Teodora Petrisor  <claudia-teodora.petrisor@thalesgroup.com>
##                    Matina Maria Trompouki  <mtrompou@ac.upc.edu>
## Description:       source file
##
## Modification:
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
###############################################################################
*/

/* -------------------------- Global constant parameters ------------------ */
#ifndef __VIOLAJONES_H__
#define __VIOLAJONES_H__

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#define N_BITS_MAX         8
#define N_CHANNELS_MAX     3
#define N_RECTANGLES_MAX   4    // max number of rectangles in Haar feature
#define N_MAX_STAGES       30   // only a test example, actual number much higher
#define N_MAX_CLASSIFIERS  250  // only a test example, actual number may be much higher

#define NB_MAX_DETECTION  100                    /* Maximum number of detections */
#define NB_MAX_POINTS     3*NB_MAX_DETECTION     /* Maximum number of detection parameters (3 points/detection) */

#define ERROR_CHECK { cudaError_t err; \
if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

/* ------------------------------- Data types ----------------------------- */


typedef struct Rectangle
{
    int x0;            // upper left column index
    int y0;            // upper left row index
    int width;
    int height;
}
Rectangle;

typedef struct CvHaarFeature
{
    int tilted; /* 0 means up-right feature, 1 means 45--rotated feature */
    /* 2-3 rectangles with weights of opposite signs and
    with absolute values inversely proportional to the areas of the rectangles.
    if rect[2].weight !=0, then
    the feature consists of 4 rectangles, otherwise it consists 
    of 2 or 3 rectangles depending on the weight ratio: 
    rect[0].weight/rect[1].weight = 0.5 => 2-rectangle,
    rect[0].weight/rect[1].weight = 1/3 => 3-rectangle */
    struct
    {
        Rectangle r;
        float weight; 
    }
    rect[N_RECTANGLES_MAX];
}
CvHaarFeature;


/* a single tree classifier (stump in the simplest case) that returns the response for the feature
   at the particular image location (i.e. pixel sum over subrectangles of the window) and gives out
   a value depending on the response */
typedef struct CvHaarClassifier
{
    int count;  /* number of nodes in the decision tree */
    CvHaarFeature* haarFeature;
    float threshold;
    float left;
    float right;
}
CvHaarClassifier;


/* a boosted battery of classifiers(=stage classifier):
   the stage classifier returns 1
   if the sum of the classifiers' responces
   is greater than threshold and 0 otherwise */
typedef struct CvHaarStageClassifier
{
    int  count;  /* number of classifiers in the battery */
    float threshold; /* threshold for the boosted classifier */
    CvHaarClassifier* classifier; /* array of classifiers up to N_MAX_CLASSIFIERS */
}
CvHaarStageClassifier;


/* cascade or tree of stage classifiers */
typedef struct CvHaarClassifierCascade
{
    int  flags; /* signature */
    int  count; /* number of stages */
    int orig_window_sizeR; /* original object size (the cascade is trained for) */
    int orig_window_sizeC;
    /* these two parameters are set by cvSetImagesForHaarClassifierCascade */
    int real_window_size; /* current object size */
    float scale; /* current scale */
    CvHaarStageClassifier* stageClassifier; /* array of stage classifiers up to N_MAX_STAGES*/
}
CvHaarClassifierCascade;

struct Lock{
        int *mutex;
        Lock(void){
                int state = 0;
		mutex=NULL;
                CUDA_SAFE_CALL(cudaMalloc((void**)&mutex, sizeof(int)));
                ERROR_CHECK

		if(mutex == NULL)
		{
			printf("Cannot allocate memory for mutex\n");
			exit(-1);
		}
                CUDA_SAFE_CALL(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
                ERROR_CHECK
        }

        ~Lock(void){
                //cudaFree(mutex);
        }

        __device__ void lock(void){
                while(atomicCAS(mutex, 0, 1) != 0);
        }

        __device__ void unlock(void){
                atomicExch(mutex, 0);
        }
};

__host__ __device__
float computeVariance(float *img, float *imgSq, int irow, int icol, int height, int width, int real_height, int real_width);

__host__ __device__
void computeFeature(float *img, float *imgSq, CvHaarFeature *f, float *featVal, int irow, int icol, int height, int width, float scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width);

__host__ __device__ 
void raster_rectangle(uint32_t *img, int x0, int y0, int radius, int real_width);


#endif


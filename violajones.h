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
##
## The entire notice above must be reproduced on all authorized copies.
##
##
## Title:             violajones.h
##
## File:              header file
## Author:            Teodora Petrisor  <claudia-teodora.petrisor@thalesgroup.com>
## Description:       source file
##
## Modification:
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
###############################################################################
*/

/* -------------------------- Global constant parameters ------------------ */
#define N_BITS_MAX         8
#define N_CHANNELS_MAX     3
#define N_RECTANGLES_MAX   4    // max number of rectangles in Haar feature
#define N_MAX_STAGES       80   // only a test example, actual number much higher
#define N_MAX_CLASSIFIERS  250  // only a test example, actual number may be much higher


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
    double scale; /* current scale */
    CvHaarStageClassifier* stageClassifier; /* array of stage classifiers up to N_MAX_STAGES*/
}
CvHaarClassifierCascade;


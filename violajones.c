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
## Title:             violajones.c
##
## File:              C file
## Author:            Teodora Petrisor <claudia-teodora.petrisor@thalesgroup.com>
## Description:       C source file
##
## Modification:
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
###############################################################################
*/

/* ************************************************************************* 
* Pedestrian detection application (adapted from OpenCV)
*        - classification based on Viola&Jones 2001 algorithm 
* 			(Haar-like features, AdaBoost algorithm)
*		 - learning data transcripted from OpenCV generated file
*
* authors:  Teodora Petrisor
* Modifications: Paul Brelet
* 
* ************************************************************************* */

/******INCLUDE********/
/*********DECLARATION*****/
/* Global Library */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#include <assert.h>
/* Static Library */
#include "violajones.h"

/******MACROS ********/
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#define INFO      1

#define MAX_BUFFERSIZE    256                    /* Maximum name file */
#define MAX_IMAGESIZE     1024                   /* Maximum Image Size */
#define MAX_BRIGHTNESS    255                    /* Maximum gray level */
#define NB_MAX_DETECTION  100                    /* Maximum number of detections */
#define NB_MAX_POINTS     3*NB_MAX_DETECTION     /* Maximum number of detection parameters (3 points/detection) */


#if INFO
#define TRACE_INFO(x) printf x
#else
#define TRACE_INFO(x)
#endif


/* ********************************** FUNCTIONS ********************************** */

/*** Read pgm file, only P2 or P5 type image ***/
void load_image_check(uint32_t *img, char *imgName, int width, int height)
{
	char buffer[MAX_BUFFERSIZE] = {0};   /* Get Image information */
	FILE *fp = NULL;                     /* File pointer */
	int x_size1 = 0, y_size1 = 0;        /* width & height of image1*/
	int max_gray = 0;                    /* Maximum gray level */
	int x = 0, y = 0;                    /* Loop variable */
	int pixel_in = 0;                    /* Get the pixel value */
	int error = 0;                       /* Check if errors */

	/* Input file open */
	
	fp = fopen(imgName, "rb");
	if (NULL == fp)
	{
		TRACE_INFO(("     The file doesn't exist!\n\n"));
		exit(1);
	}
	/* Check of file-type ---P2 or P5 */
	fgets(buffer, MAX_BUFFERSIZE, fp);

	if(buffer[0] == 'P' && buffer[1] == '2')
	{
		/* input of x_size1, y_size1 */
		x_size1 = 0;
		y_size1 = 0;
		while (x_size1 == 0 || y_size1 == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d %d", &x_size1, &y_size1);
			}
		}
		/* input of max_gray */
		max_gray = 0;
		while (max_gray == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d", &max_gray);
			}
		}
		/* Display parameters */
		if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE)
		{
			TRACE_INFO(("     Image size exceeds %d x %d\n\n", MAX_IMAGESIZE, MAX_IMAGESIZE));
			TRACE_INFO(("     Please use smaller images!\n\n"));
			exit(1);
		}
		if (max_gray != MAX_BRIGHTNESS)
		{
			TRACE_INFO(("     Invalid value of maximum gray level!\n\n"));
			exit(1);
		}
		/* Input of image data*/
		for(y=0; y < y_size1; y++)
		{
			for(x=0; x < x_size1; x++)
			{
        // read PGM pixel and check input stream state 
				error = fscanf(fp, "%d", &pixel_in);
				if (error <= 0) 
				{
					if (feof(fp))
					{
						TRACE_INFO(("PGM file, premature EOF !\n"));
					}
					else if (ferror(fp))
					{
						TRACE_INFO(("PGM file format error !\n"));
					}
					else
					{
						TRACE_INFO(("PGM file, fatal error during read !\n"));
						exit(1);
					}
				}
				img[y*x_size1+x] = pixel_in;
			}
		}
	}
	else if(buffer[0] == 'P' && buffer[1] == '5') 
	{
		/* Input of x_size1, y_size1 */
		x_size1 = 0;
		y_size1 = 0;
		while (x_size1 == 0 || y_size1 == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d %d", &x_size1, &y_size1);
			}
		}
		/* Input of max_gray */
		max_gray = 0;
		while (max_gray == 0)
		{
			fgets(buffer, MAX_BUFFERSIZE, fp);
			if (buffer[0] != '#')
			{
				sscanf(buffer, "%d", &max_gray);
			}
		}
		/* Display parameters */
		TRACE_INFO(("\n   Image width = %d, Image height = %d\n", x_size1, y_size1));
		TRACE_INFO(("     Maximum gray level = %d\n\n", max_gray));
		if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE)
		{
			TRACE_INFO(("     Image size exceeds %d x %d\n\n", MAX_IMAGESIZE, MAX_IMAGESIZE));
			TRACE_INFO(("     Please use smaller images!\n\n"));
			exit(1);
		}
		if (max_gray != MAX_BRIGHTNESS)
		{
			TRACE_INFO(("     Invalid value of maximum gray level!\n\n"));
			exit(1);
		}
		/* Input of image data*/
		for (y = 0; y < y_size1; y++)
		{
			for (x = 0; x < x_size1; x++)
			{
				img[y*x_size1+x] = (uint32_t) fgetc(fp);
			}
		}
	}
	else
	{
		TRACE_INFO(("    Wrong file format, only P2/P5 allowed!\n\n"));
		exit(1);
	}
	fclose(fp);
}
//end function: load_image_check ***********************************************

/*** Get the MAX pixel value from image ****/
int maxImage(uint32_t *img, int height, int width)
{
	int maximg = 0;
	int irow = 0;

	for(irow = 0; irow < height*width; irow++)
	{
		if (img[irow]> maximg )
		{
			maximg = img[irow];
		}
	}
	return maximg;
}
//end function: maxImage *******************************************************

/*** Get image dimensions from pgm file ****/
void getImgDims(char *imgName, int *width, int *height)
{
	FILE *pgmfile = NULL;
	char filename[MAX_BUFFERSIZE]={0};
	char buff1[MAX_BUFFERSIZE]={0};
	char buff2[MAX_BUFFERSIZE]={0};

	sprintf(filename, imgName);
	pgmfile = fopen(filename,"r");

	if (pgmfile == NULL)
	{
		TRACE_INFO(("\nPGM file \"%s\" cannot be opened !\n",filename));
		exit(1);
	} 
	else
	{        
		fgets(buff1, MAX_BUFFERSIZE, pgmfile);
		fgets(buff2, MAX_BUFFERSIZE, pgmfile);
		fscanf(pgmfile, "%d %d",width, height);
	}
}
//end function: getImgDims *****************************************************

/*** Write the result image ***/
void imgWrite(uint32_t *imgIn, char img_out_name[MAX_BUFFERSIZE], int height, int width)
{
	FILE *pgmfile_out = NULL;

	int irow = 0;
	int icol = 0;
	int maxval = 0;

	pgmfile_out = fopen(img_out_name, "wt");

	if (pgmfile_out == NULL) 
	{
		TRACE_INFO(("\nPGM file \"%s\" cannot be opened !\n", img_out_name));
		exit(1);
	}
	maxval = maxImage((uint32_t*)imgIn, height, width);
	if (maxval>MAX_BRIGHTNESS)
	{
		fprintf(pgmfile_out, "P2\n# CREATOR: smartcamera.c\n%d %d\n%d\n", width, height, maxval);
	}
	else
	{
		fprintf(pgmfile_out, "P2\n# CREATOR: smartcamera.c\n%d %d\n%d\n", width, height, MAX_BRIGHTNESS);
	}
	for (irow = 0; irow < height; irow++)
	{
		for (icol = 0; icol < width; icol++) 
		{
			fprintf(pgmfile_out, "%d\n", imgIn[irow*width+icol]);
		}
	}
	fclose(pgmfile_out);
}
//end function: imgWrite *******************************************************

/*** Allocation function for Classifier Cascade ***/
CvHaarClassifierCascade* allocCascade()
{
	int i = 0;
	int j = 0;
	int k = 0;

	CvHaarClassifierCascade *cc;

	cc = malloc(sizeof(CvHaarClassifierCascade));
	cc->stageClassifier = calloc(N_MAX_STAGES,sizeof(CvHaarStageClassifier));
	for (i = 0; i < N_MAX_STAGES; i++)
	{
		cc->stageClassifier[i].classifier = calloc(N_MAX_CLASSIFIERS,sizeof(CvHaarClassifier));
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			cc->stageClassifier[i].classifier[j].haarFeature = malloc(sizeof(CvHaarFeature));
			for (k = 0; k<N_RECTANGLES_MAX; k++)
			{
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.x0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.y0 = 0;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.width = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].r.height = 1;
				cc->stageClassifier[i].classifier[j].haarFeature->rect[k].weight = 1.0;
			}
			cc->stageClassifier[i].classifier[j].threshold = 0.0;        
			cc->stageClassifier[i].classifier[j].left = 1.0;
			cc->stageClassifier[i].classifier[j].right = 1.0;
		}
		cc->stageClassifier[i].count = 1;
		cc->stageClassifier[i].threshold = 0.0;
	}
	return cc;
}
//end function: allocCascade ***************************************************

/*** Deallocation function for the whole Cascade ***/
void releaseCascade(CvHaarClassifierCascade *cc)
{
	int i = 0;
	int j = 0;

	for (i=0; i<N_MAX_STAGES; i++)
	{
		for (j=0; j<N_MAX_CLASSIFIERS; j++)
		{
			free(cc->stageClassifier[i].classifier[j].haarFeature);
		}
		free(cc->stageClassifier[i].classifier);
	}
	free(cc->stageClassifier);
	free(cc);
}
//end function: releaseCascade *************************************************

/*** Read classifier cascade file and build cascade structure***/
void readClassifCascade(char *haarFileName, CvHaarClassifierCascade *cascade, int *nRows, int *nCols, int *nStages)
{
	FILE *haarfile = NULL;

	char line[MAX_BUFFERSIZE] = {0};
	char linetag = 0;

	int x0 = 0, y0 = 0, wR = 0, hR = 0;    // rectangle coordinates
	int iStage = 0;
	int iNode=0;
	int nRectangles = 0;        
	int isRect = 0;

	float thresh = 0.0;
	float featThresh = 0.0;                        
	float weight = 0.0;
	float a = 0.0;
	float b = 0.0;

	haarfile = fopen(haarFileName, "r");
	if (haarfile == NULL) 
	{        
		TRACE_INFO(("\nFile \"%s\" cannot be opened !\n", haarFileName));
		exit(1);
	}
	else
	{
		fscanf(haarfile,"%d %d", nCols, nRows); 
		while (!feof(haarfile))
		{
			fgets(line, MAX_BUFFERSIZE, haarfile);
			linetag = line[0];
			if (isRect) 
			{
				nRectangles++;
			}
			else 
			{
				nRectangles = 1;
			}
			switch (linetag)
			{
				case 'S':
				{
//                Stage number index
					sscanf(line,"%*s %d", &iStage);
					isRect = 0;
					break;
				}
				case 'T':
				{
//                Stage threshold
					sscanf(line,"%*s %*d %f", &thresh);
					isRect = 0;
					assert(iStage<N_MAX_STAGES);
					cascade->stageClassifier[iStage].count = iNode+1;
					cascade->stageClassifier[iStage].threshold = thresh;
					break;
				}
				case 'N':
				{
//                Feature (node) index
					sscanf(line,"%*s %d",&iNode);
					break;
				}
				case 'R':
//                Rectangle feature; encoded as (left corner) column row width height weight
//                weight indicates the type of rectangle (sign(weight)<0 <=> white rectangle, else <=> black rectangle)
				{
					isRect = 1;
					sscanf(line,"%*s %d %d %d %d %f", &x0, &y0, &wR, &hR, &weight);
					assert(iNode<N_MAX_CLASSIFIERS);
				        assert(nRectangles-1<N_RECTANGLES_MAX);
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.x0 = x0;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.y0 = y0;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.width = wR;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].r.height = hR;
					cascade->stageClassifier[iStage].classifier[iNode].haarFeature->rect[nRectangles-1].weight = weight;
					break;
				}
				case 'a':
				{
					sscanf(line,"%*s %f", &a);
					assert(iStage<N_MAX_STAGES);
					assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].left = a;
					break;
				}                                
				case 'b':
				{
					sscanf(line,"%*s %f", &b);
					assert(iStage<N_MAX_STAGES);
                                        assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].right = b;
					break;
				}
				default:
				{
					isRect = 0;
					sscanf(line,"%f",&featThresh);
					assert(iStage<N_MAX_STAGES);
                                        assert(iNode<N_MAX_CLASSIFIERS);
					cascade->stageClassifier[iStage].classifier[iNode].threshold = featThresh;
				}
			}
		}        
		*nStages = iStage+1;
		assert(*nStages<N_MAX_STAGES);
		cascade->count = *nStages;
		cascade->orig_window_sizeR = *nRows;
		cascade->orig_window_sizeC = *nCols;
	}
	fclose(haarfile);
}
//end function: readClassifCascade *********************************************

/*** Pixel-wise square image ****/
void imgDotSquare(uint32_t *imgIn, uint32_t *imgOut, int height, int width)
{
	int irow = 0, icol = 0;

	for (irow = 0; irow < height; irow++)
	{
		for (icol = 0; icol < width; icol++)
		{
			imgOut[irow*width+icol] = imgIn[irow*width+icol] * imgIn[irow*width+icol];
		}
	}
}
//end function: imgDotSquare ***************************************************

/*** Compute variance-normalized image ****/
void imgNormalize(uint32_t *imgIn, double *imgOut, double normFact, int height, int width)
{
	int irow = 0, icol = 0;
	int dim = 0;
	double meanImg = 0.0;

	dim = width*height;

	if(dim != 0)
	{
		meanImg = (imgIn[(height-1)*width+(width-1)])/dim;
	}

	if(normFact != 0)
	{
		for (irow = 0; irow < height; irow++)
		{
			for (icol = 0; icol < width; icol++)
			{
				imgOut[irow*width+icol] = (imgIn[irow*width+icol]- meanImg)/sqrt(normFact);
			}
		}
	}
}
//end function: imgNormalize ***************************************************

/*** Cast int image as double ****/
void imgCopy(uint32_t *imgIn, double *imgOut, int height, int width)
{
	int irow = 0, icol = 0;

	for (irow = 0; irow < height; irow++)
	{
		for (icol = 0; icol < width; icol++)
		{
			imgOut[irow*width+icol] = (double)imgIn[irow*width+icol];
		}
	}
}
//end function: imgCopy *******************************************************

/*** Copy one haarFeature into another ****/
void featCopy(CvHaarFeature *featSource, CvHaarFeature *featDest)
{
	int i = 0;

	for (i = 0; i < 3; i++)
	{
		featDest->rect[i].r.x0 = featSource->rect[i].r.x0;
		featDest->rect[i].r.y0 = featSource->rect[i].r.y0;
		featDest->rect[i].r.width = featSource->rect[i].r.width;
		featDest->rect[i].r.height = featSource->rect[i].r.height;        
		featDest->rect[i].weight = featSource->rect[i].weight;        
	}
}
//end function: featCopy *******************************************************

/*** Compute integral image ****/
void computeIntegralImg(uint32_t *imgIn, uint32_t *imgOut, int height, int width)
{
	int irow = 0, icol = 0;
	uint32_t row_sum = 0;

	for (irow = 0 ; irow < height; irow++)
	{
		row_sum = 0;
		for (icol = 0 ; icol < width; icol++)
		{
			row_sum += imgIn[irow*width+icol];
			if (irow > 0)
			{
				imgOut[irow*width+icol] = imgOut[(irow-1)*width+icol] + row_sum;
			}
			else
			{
				imgOut[irow*width+icol] = row_sum;
			}
		}
	}
}
//end function: computeIntegralImg *********************************************

/*** Recover any pixel in the image by using the integral image ****/
double getImgIntPixel(double *img, int row, int col, int real_height, int real_width)
{
	double pval = 0.0;

	if ((row == 0) && (col == 0))
	{
		pval = img[row*real_width+col];
		return pval;
	}
	if ((row > 0) && (col > 0))
	{
		pval = img[(row-1)*real_width+(col-1)] - img[(row-1)*real_width+col] - img[row*real_width+(col-1)] + img[row*real_width+col];
	}
	else
	{
		if (row == 0)
		{
			pval = img[col] - img[col-1];
		}
		else
		{
			if (col == 0)
			{
				pval = img[row*real_width] - img[(row-1)*real_width];
			}
		}
	}
	return pval;
}
//end function: getImgIntPixel *************************************************

/*** Compute any rectangle sum from integral image ****/
double computeArea(double *img, int row, int col, int height, int width, int real_height, int real_width)
{
	double sum = 0.0;
	int cornerComb = 0;

  // rectangle = upper-left corner pixel of the image
	if ((row == 0) && (col == 0) && (width == 1) && (height == 1))
	{
		sum = img[0];
		return sum;
	}
  // rectangle = pixel anywhere in the image
	else
	{
		if ((width == 1) && (height == 1))
		{
			sum = getImgIntPixel((double *)img, row, col, real_height, real_width);
			return sum;        
		}
    // map upper-left corner of rectangle possible combinations        
		if ((row == 0) && (col == 0))
		{
			cornerComb = 1;
		}
		if ((row == 0) && (col > 0))
		{
			cornerComb = 2;
		}
		if ((row > 0) && (col == 0))
		{
			cornerComb = 3;
		}
		if ((row > 0) && (col > 0))
		{
			cornerComb = 4;
		}

		switch (cornerComb)
		{
			case 1:
			{
        // row = 0, col = 0
				sum = img[(row+height-1)*real_width+(col+width-1)];
				break;
			}
			case 2:
			{
        // row = 0, col > 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row+height-1)*real_width+(col-1)]);
				break;
			}
			case 3:
			{
        // row > 0, col = 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row-1)*real_width+(col+width-1)]);
				break;
			}
			case 4:
			{
        // row > 0, col > 0
				sum = (img[(row+height-1)*real_width+(col+width-1)] - img[(row-1)*real_width+(col+width-1)] - img[(row+height-1)*real_width+(col-1)] + img[(row-1)*real_width+(col-1)]);
				break;
			}
			default:
			{
				TRACE_INFO(("Error: \" This case is impossible!!!\"\n"));
				break;
			}
		}

		if(sum >= DBL_MAX-1)
		{
			sum = DBL_MAX;
		}
	}
	return sum;
}
//end function: computeArea ****************************************************

/*** Compute parameters for each rectangle in a feature: 
****        upper-left corner, width, height, sign       ****/
void getRectangleParameters(CvHaarFeature *f, int iRectangle, int nRectangles, double scale, int rOffset, int cOffset, int *row, int *col, int *height, int *width)
{
	int r = 0, c = 0, h = 0, w = 0;

	w = f->rect[1].r.width;
	h = f->rect[1].r.height;

	if ((iRectangle > nRectangles) || (nRectangles < 2))
	{
		TRACE_INFO(("Problem with rectangle index %d/%d or number of rectangles.\n", iRectangle, nRectangles));
		return;
	}

  // get upper-left corner according to rectangle index in the feature (max 4-rectangle features)
	switch (iRectangle)
	{
		case 0: 
		{
			r = f->rect[0].r.y0;
			c = f->rect[0].r.x0;
			break;
		}
		case 1: 
		{
			switch (nRectangles)
			{
				case 2:
				{
					if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
					{
						if (f->rect[0].r.width == f->rect[1].r.width) 
						{
							r = f->rect[0].r.y0 + h;
							c = f->rect[0].r.x0;
						}
						else
						{
							r = f->rect[0].r.y0;
							c = f->rect[0].r.x0 + w;
						}
					}
					else
					{
						r = f->rect[1].r.y0;
						c = f->rect[1].r.x0;
					}
					break;
				}
				case 3:
				{
					r = f->rect[1].r.y0;
					c = f->rect[1].r.x0;
					break;
				}
				case 4:
				{
					if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
					{
						r = f->rect[0].r.y0;
						c = f->rect[0].r.x0 + w;
					}
					else
					{
						r = f->rect[1].r.y0;
						c = f->rect[1].r.x0;
					}
					break;
				}
			}
			break;
		}
		case 2: 
		{
			if (nRectangles == 3) 
			{
				if (f->rect[0].r.x0 == f->rect[1].r.x0)
				{
					r = f->rect[1].r.y0 + h;
					c = f->rect[0].r.x0;
				}
				else
				{
					r = f->rect[0].r.y0;
					c = f->rect[1].r.x0 + w;
				}
			}
			else
			{
				if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
				{
					r = f->rect[0].r.y0 + h;
					c = f->rect[0].r.x0;
				}
				else
				{
					r = f->rect[2].r.y0;
					c = f->rect[2].r.x0;
				}
			}
			break;
		}
		case 3: 
		{
			if ((f->rect[0].r.x0 == f->rect[1].r.x0) &&  (f->rect[0].r.y0 == f->rect[1].r.y0))
			{
				r = f->rect[2].r.y0;
				c = f->rect[2].r.x0;
			}
			else
			{
				r = f->rect[2].r.y0;
				c = f->rect[2].r.x0 + w;
			}
			break;
		}
	}

	*row = (int)(floor(r*scale)) + rOffset;
	*col = (int)(floor(c*scale)) + cOffset;
	*width = (int)(floor(w*scale));
	*height = (int)(floor(h*scale));
}
//end function: getRectangleParameters *****************************************

/*** Re-create feature structure from rectangle coordinates and feature type (test function!) ****/
void writeInFeature(int rowVect[4], int colVect[4], int hVect[4], int wVect[4], float weightVect[4], int nRects, CvHaarFeature *f_scaled)
{
	f_scaled->rect[1].r.width = wVect[1];
	f_scaled->rect[1].r.height = hVect[1];

	f_scaled->rect[0].r.x0 = colVect[0];
	f_scaled->rect[0].r.y0 = rowVect[0];

	switch (nRects)
	{
		case 2:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 2.0;
			f_scaled->rect[2].weight = 0.0;

			if ((weightVect[0] == 2.0) && (weightVect[2] == 0.0))
			{
				f_scaled->rect[1].r.x0 = colVect[0];
				f_scaled->rect[1].r.y0 = rowVect[0];
			}
			else
			{
				f_scaled->rect[1].r.x0 = colVect[1];
				f_scaled->rect[1].r.y0 = rowVect[1];
			}
			if (rowVect[0] == rowVect[1])
			{
				f_scaled->rect[0].r.width = wVect[1] * 2;
				f_scaled->rect[0].r.height = hVect[1];
			}
			else
			{
				f_scaled->rect[0].r.width = wVect[1];
				f_scaled->rect[0].r.height = hVect[1] * 2;
			} 
			break;
		}
		case 3:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 3.0;
			f_scaled->rect[2].weight = 0.0;

			if (rowVect[0] == rowVect[1])
			{
				f_scaled->rect[0].r.width = wVect[1] * 3;
				f_scaled->rect[0].r.height = hVect[1];
			}
			else
			{
				f_scaled->rect[0].r.width = wVect[1];
				f_scaled->rect[0].r.height = hVect[1] * 3;
			} 
			f_scaled->rect[1].r.x0 = colVect[1];
			f_scaled->rect[1].r.y0 = rowVect[1];
			break;
		}
		case 4:
		{
			f_scaled->rect[0].weight = -1.0;
			f_scaled->rect[1].weight = 2.0;
			f_scaled->rect[2].weight = 2.0;

			f_scaled->rect[0].r.width = wVect[1]*2;
			f_scaled->rect[0].r.height = hVect[1]*2;

			if (weightVect[0] == 2.0)
			{
				f_scaled->rect[1].r.x0 = colVect[0];
				f_scaled->rect[1].r.y0 = rowVect[0];
				f_scaled->rect[2].r.x0 = colVect[3];
				f_scaled->rect[2].r.y0 = rowVect[3];
			}
			else
			{
				f_scaled->rect[1].r.x0 = colVect[1];
				f_scaled->rect[1].r.y0 = rowVect[1];
				f_scaled->rect[2].r.x0 = colVect[2];
				f_scaled->rect[2].r.y0 = rowVect[2];
			}

			f_scaled->rect[2].r.width = wVect[1];
			f_scaled->rect[2].r.height = hVect[1];
			break;
		}
	}
}
//end function: writeInFeature *************************************************

/*** Compute feature value (this is the core function!) ****/
void computeFeature(double *img, double *imgSq, CvHaarFeature *f, double *featVal, int irow, int icol, int height, int width, double scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width)
{
	int nRects = 0;
	int col = 0;
	int row = 0;
	int wRect = 0;
	int hRect = 0;
	int i = 0;
	int rectArea = 0;
	int colVect[4] = {0};
	int rowVect[4] = {0};
	int wVect[4] = {0};
	int hVect[4] = {0};

	float w1 = 0.0, w1_orig = 0.0;
	float rectWeight[4] = {0};

	double val = 0.0;
	double s[N_RECTANGLES_MAX] = {0};

	*featVal = 0.0;

	w1_orig = f->rect[0].weight;
	w1 = f->rect[0].weight * scale_correction_factor;

  // Determine feature type (number of rectangles) according to weight
	if (f->rect[2].weight == 2.0)
	{
		nRects = 4;
		if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
		{
			rectWeight[0] = -w1; 
			rectWeight[1] = w1;
			rectWeight[2] = w1;
			rectWeight[3] = -w1; 
		}
		else
		{
			rectWeight[0] = w1;
			rectWeight[1] = -w1; 
			rectWeight[2] = -w1; 
			rectWeight[3] = w1; 
		}
	}
	else 
	{ 
		if (f->rect[1].weight == 2.0)
		{
			nRects = 2;
			if ((f->rect[0].r.x0 == f->rect[1].r.x0) && (f->rect[0].r.y0 == f->rect[1].r.y0))
			{
				rectWeight[0] = -w1;
				rectWeight[1] = w1;
			}
			else
			{
				rectWeight[0] = w1;
				rectWeight[1] = -w1;
			}
			rectWeight[2] = 0.0;
			rectWeight[3] = 0.0;
		}
		else
		{
			nRects = 3;
			rectWeight[0] = w1;
			rectWeight[1] = -2.0*w1;
			rectWeight[2] = w1;
			rectWeight[3] = 0.0;
		}
	}
	for (i = 0; i<nRects; i++)
	{
		s[i] = 0.0; 
		getRectangleParameters(f, i, nRects, scale, irow, icol, &row, &col, &hRect, &wRect);
		s[i] = computeArea((double *)img, row, col, hRect, wRect, real_height, real_width);
		rectArea = hRect*wRect;

		if (fabs(rectWeight[i]) > 0.0)
		{
			val += rectWeight[i]*s[i];
		}
    // test values for each rectangle
		rowVect[i] = row; colVect[i] = col; hVect[i] = hRect; wVect[i] = wRect;
	}
	*featVal = val;
	writeInFeature(rowVect,colVect,hVect,wVect,rectWeight,nRects,f_scaled);
}
//end function: computeFeature *************************************************

/*** Calculate the Variance ****/
double computeVariance(double *img, double *imgSq, int irow, int icol, int height, int width, int real_height, int real_width)
{
	int nPoints = 0;

	double s1 = 0.0;
	double s2 = 0.0;
	double f1 = 0.0;
	double f2 = 0.0;
	double varFact = 0.0;

	nPoints = height*width;

	s1 = (double)computeArea((double *)img, irow, icol, height, width, real_height, real_width);
	s2 = (double)computeArea((double *)imgSq, irow, icol, height, width, real_height, real_width);

	if(nPoints != 0)
	{
		f1 = (double)(s1/nPoints);
		f2 = (double)(s2/nPoints);
	}

	if(f1*f1 > f2)
	{
		varFact = 0.0;
	}
	else
	{
		varFact = f2 - f1*f1;
	}

	return varFact;
}
//end function: computeVariance ************************************************

/*** Allocate one dimension integer pointer ****/
uint32_t *alloc_1d_uint32_t(int n)
{
	uint32_t *new;

	new = (uint32_t *) malloc ((unsigned) (n * sizeof (uint32_t)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_1D_UINT_32T: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new);
}
//end function: alloc_1d_uint32_t **********************************************

/*** Allocate one dimension double pointer ****/
double *alloc_1d_double(int n)
{
	double *new;

	new = (double *) malloc ((unsigned) (n * sizeof (double)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_1D_DOUBLE: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new);
}
//end function: alloc_1d_double ************************************************

/*** Allocate 2d array of integers ***/
uint32_t **alloc_2d_uint32_t(int m, int n)
{
	int i;
	uint32_t **new;

	new = (uint32_t **) malloc ((unsigned) (m * sizeof (uint32_t *)));
	if (new == NULL) {
		TRACE_INFO(("ALLOC_2D_UINT_32T: Couldn't allocate array of integer ptrs\n"));
		return (NULL);
	}

	for (i = 0; i < m; i++) {
		new[i] = alloc_1d_uint32_t(n);
	}

	return (new);
}
//end function: alloc_2d_uint32_t **********************************************

/* Draws simple or filled square */
void raster_rectangle(uint32_t* img, int x0, int y0, int radius, int real_width)
{
	int i=0;
	for(i=-radius/2; i<radius/2; i++)
	{
		assert((i + x0 + (y0 + (int)(radius)) * real_width)>0);
		assert((i + x0 + (y0 + (int)(radius)) * real_width)<(480*640));
		assert((i + x0 + (y0 - (int)(radius)) * real_width)>0);
		assert((i + x0 + (y0 - (int)(radius)) * real_width)<(480*640));
		img[i + x0 + (y0 + (int)(radius)) * real_width]=255;
		img[i + x0 + (y0 - (int)(radius)) * real_width]=255;
	}
	for(i=-(int)(radius); i<(int)(radius); i++)
	{
		assert(((x0 + (int)(radius/2)) + (y0+i) * real_width)>0);
		assert(((x0 + (int)(radius/2)) + (y0+i) * real_width)<(480*640));
		assert(((x0 - (int)(radius/2)) + (y0+i) * real_width)>0);
		assert(((x0 - (int)(radius/2)) + (y0+i) * real_width)<(480*640));
		img[(x0 + (int)(radius/2)) + (y0+i) * real_width]=255;
		img[(x0 - (int)(radius/2)) + (y0+i) * real_width]=255;
	}	
}
//end function: raster_rectangle **************************************************


/* ********************************** MAIN ********************************** */
int main( int argc, char** argv )
{
	// Timer declaration 
	time_t start, end;

	// Pointer declaration
	CvHaarClassifierCascade* cascade = NULL;
	CvHaarClassifierCascade* cascade_scaled = NULL;
	CvHaarFeature *feature_scaled = NULL;

	char *imgName = NULL;
	char *haarFileName = NULL;
	char result_name[MAX_BUFFERSIZE]={0};

	uint32_t *img = NULL;
	uint32_t *imgInt = NULL;
	uint32_t *imgSq = NULL;
	uint32_t *imgSqInt = NULL;
	uint32_t **result2 = NULL;


	uint32_t **goodcenterX=NULL;
	uint32_t **goodcenterY=NULL;
	uint32_t **goodRadius=NULL;
	uint32_t *nb_obj_found2=NULL;

	uint32_t *position= NULL;
	uint32_t *goodPoints = NULL;

	// Counter Declaration 
	int rowStep = 1;
	int colStep = 1;
	int width = 0;
	int height = 0;
	int detSizeR = 0;  
	int detSizeC = 0; 
	int tileWidth = 0;
	int tileHeight = 0;
	int irow = 0;
	int icol = 0;
	int nStages = 0; 
	int foundObj = 0;
	int nTileRows = 0;
	int nTileCols = 0;
	int iStage = 0;
	int iNode = 0;
	int nNodes = 0;
	int pointCount = 0;
	int i = 0, j = 0;
	
	
	int real_height = 0, real_width = 0;
	int scale_index_found=0;
	int threshold_X=0;
	int threshold_Y=0;
	int nb_obj_found=0;


	int count = 0;
	int offset_X = 0, offset_Y = 0;

	// Threshold Declaration 
	float thresh = 0.0;
	float a = 0.0;
	float b = 0.0;
	float scale_correction_factor = 0.0;
	
	// /**! Brief: For the circle calculation */
	float centerX_tmp=0.0;
	float centerY_tmp=0.0;
	float radius_tmp=0.0;
	float centerX=0.0;
	float centerY=0.0;
	float radius=0.0;

	// Factor Declaration 
	double scaleFactorMax = 0.0;
	double scaleStep = 1.1; // 10% increment of detector size per scale. Change this value to test other increments 
	double scaleFactor = 0.0;
	double varFact = 0.0;
	double sumClassif = 0.0;
	double featVal = 0.0;
	double detectionTime = 0.0;

	// Integral Image Declaration 
	double *imgInt_f = NULL;
	double *imgSqInt_f = NULL;

	int rrr=0;	

	if (argc <= 2 || argc > 3)
	{
		TRACE_INFO(("Usage: %s image classifier\n", argv[0]));
		return(0);
	}

	// Get the Image name and the Cascade file name from the console 
	imgName=argv[1];
	haarFileName=argv[2];


	// Get the Input Image informations 
	getImgDims(imgName, &width, &height);

	// Get back the Real Size of the Input Image 
	real_height=height;
	real_width=width;

	// Allocate the Cascade Memory 
	cascade = allocCascade();
	cascade_scaled = allocCascade();
	feature_scaled = malloc(sizeof(CvHaarFeature));

	// Get the Classifier informations 
	readClassifCascade(haarFileName, cascade, &detSizeR, &detSizeC, &nStages);

	printf("detSizeR = %d\n", detSizeR);
	printf("detSizeC = %d\n", detSizeC);

	// Determine the Max Scale Factor
	if (detSizeR != 0 && detSizeC != 0)
	{
		scaleFactorMax = min((int)floor(height/detSizeR), (int)floor(width/detSizeC));
	}

	// Give the Allocation size 
	img=alloc_1d_uint32_t(width*height);
	imgInt=alloc_1d_uint32_t(width*height);
	imgSq=alloc_1d_uint32_t(width*height);
	imgSqInt=alloc_1d_uint32_t(width*height);
	result2=alloc_2d_uint32_t(N_MAX_STAGES, width*height);
	position=alloc_1d_uint32_t(width*height);

	printf("nStages: %d\n", nStages);
	printf("NB_MAX_DETECTION: %d\n", NB_MAX_DETECTION);
	
	
	//nstages
	goodcenterX=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	goodcenterY=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	goodRadius=alloc_2d_uint32_t(N_MAX_STAGES, NB_MAX_DETECTION);
	nb_obj_found2=alloc_1d_uint32_t(N_MAX_STAGES);


	goodPoints=alloc_1d_uint32_t(width*height);
	imgInt_f=alloc_1d_double(width*height);
	imgSqInt_f=alloc_1d_double(width*height);

	printf("Allocations finished!\n");

	// load the Image in Memory 
	load_image_check((uint32_t *)img, (char *)imgName, width, height);

	printf("Load Image Done!\n");




	// Compute the Interal Image 
	computeIntegralImg((uint32_t *)img, (uint32_t *)imgInt, height, width);

	// Calculate the Image square 
	imgDotSquare((uint32_t *)img, (uint32_t *)imgSq, height, width);
	/* Compute the Integral Image square */
	computeIntegralImg((uint32_t *)imgSq, (uint32_t *)imgSqInt, height, width);

 start = clock();
	// Copy the Image to float array 
	imgCopy((uint32_t *)imgInt, (double *)imgInt_f, height, width);
	imgCopy((uint32_t *)imgSqInt, (double *)imgSqInt_f, height, width);

	printf("Done with integral image\n");

	printf("scaleFactorMax = %f\n", scaleFactorMax);
	printf("scaleStep = %f\n", scaleStep);
	printf("nStages = %d\n", nStages);
	

	// Launch the Main Loop 
	for (scaleFactor = 1; scaleFactor <= scaleFactorMax; scaleFactor *= scaleStep)
	{
		for (irow = 0; irow < real_height; irow ++)
			for (icol = 0; icol < real_width; icol++)
		{
			goodPoints[irow*real_width+icol] = 255;
		}
//		TRACE_INFO(("Processing scale %f/%f\n", scaleFactor, scaleFactorMax));
		tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
		tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);
		rowStep = max(2, (int)floor(scaleFactor + 0.5));
		colStep = max(2, (int)floor(scaleFactor + 0.5));

    //(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
		scale_correction_factor = (float)1.0/(float)((int)floor((detSizeC-2)*scaleFactor)*(int)floor((detSizeR-2)*scaleFactor)); 

		// compute number of tiles: difference between the size of the Image and the size of the Detector 
		nTileRows = height-tileHeight;
		nTileCols = width-tileWidth;
		
		printf("Inside scale for and before stage for!\n");

		// Operation used for every Stage of the Classifier 
		for (iStage = 0; iStage < nStages; iStage++)
		{
			nNodes = cascade->stageClassifier[iStage].count;
			foundObj = 0;

			for (irow = 0; irow < nTileRows; irow+=rowStep)
			{
				for (icol = 0; icol < nTileCols; icol+=colStep)
				{
					if (goodPoints[irow*real_width+icol])
					{
						sumClassif = 0.0;
						/* Operation used for every Stage of the Classifier */
						varFact=computeVariance((double *)imgInt_f, (double *)imgSqInt_f, irow, icol, tileHeight, tileWidth, real_height, real_width);

						if (varFact < 10e-15)
						{
							// this should not occur (possible overflow BUG)
							varFact = 1.0; 
							goodPoints[irow*real_width+icol] = 0; 
							continue;
						}
						else
						{
							// Get the standard deviation 
							varFact = sqrt(varFact);
						}
						pointCount = 0;
						for (iNode = 0; iNode < nNodes; iNode++)
						{
							computeFeature((double *)imgInt_f, (double *)imgSqInt_f, cascade->stageClassifier[iStage].classifier[iNode].haarFeature,
											&featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, feature_scaled, real_height, real_width);
							// Get the thresholds for every Node of the stage 
							thresh = cascade->stageClassifier[iStage].classifier[iNode].threshold;
							a = cascade->stageClassifier[iStage].classifier[iNode].left;
							b = cascade->stageClassifier[iStage].classifier[iNode].right;
							sumClassif += (featVal < (double)(thresh*varFact) ? a : b);
						}
						// Update goodPoints according to detection threshold 
						if (sumClassif < cascade->stageClassifier[iStage].threshold)
						{
							goodPoints[irow*real_width+icol] = 0;
						}	  
						else
						{	
							if (iStage == nStages - 1)
							{ 
								foundObj++;
							}
						}	  
					}
				}
			}
		}

	printf("Finished founds\n");	

	// ************************************************************* 
	// One scale done, post-process the results 
	// ************************************************************* 
	
		// Determine used object 
		if (foundObj)
		{
			nb_obj_found=0;

			// Determine the position of the object detection 
			for (irow=0; irow<nTileRows; irow+=rowStep)
			{
				for (icol=0; icol<nTileCols; icol+=colStep)
				{
					// Only the detection is used 
					if (goodPoints[irow*real_width+icol])
					{
						// Calculation of the Center of the detection 
						centerX=(((tileWidth-1)*0.5+icol));
						centerY=(((tileHeight-1)*0.5+irow));
						
						//Calculation of the radius of the circle surrounding object 
						radius = sqrt(pow(tileHeight-1, 2)+pow(tileWidth-1, 2))/2;

						//Threshold calculation: proportionnal to the size of the Detector 
						threshold_X=(int)((tileHeight-1)/(2*scaleFactor));
						threshold_Y=(int)((tileWidth-1)/(2*scaleFactor));

						//Reduce number of detections in a given range 
						if(centerX > (centerX_tmp+threshold_X) || centerX < (centerX_tmp-threshold_X) || (centerY > centerY_tmp+threshold_Y) || centerY < (centerY_tmp-threshold_Y))
						{
							centerX_tmp=centerX;
							centerY_tmp=centerY;
							radius_tmp=radius;
							// Get only the restricted Good Points and get back the size for each one 
							goodcenterX[scale_index_found][nb_obj_found]=centerX_tmp;
							goodcenterY[scale_index_found][nb_obj_found]=centerY_tmp;
							goodRadius[scale_index_found][nb_obj_found]=radius_tmp;

							nb_obj_found=nb_obj_found + 1;
							// store number of detections at each scale
							nb_obj_found2[scale_index_found]=nb_obj_found; 
						}
					}
				}
			}
			scale_index_found++;
		}

//printf("ScaleFactor: %f\n", scaleFactor);
/*                            
for(rrr=0; rrr<nStages; rrr++)
{                          
      printf("Number: %d --- nb_obj_found=%d\n", rrr,  nb_obj_found);
}
*/


	printf("Finished founds processing\n");

	}
	// Done processing all scales 
	

	printf("Finished scales\n");
	// Multi-scale fusion and detection display (note: only a simple fusion scheme implemented
	for(i=0; i<scale_index_found; i++)
	{
		nb_obj_found2[scale_index_found]=max(nb_obj_found2[scale_index_found], nb_obj_found2[i]);
	}

	// Keep the position of each circle from the bigger to the smaller 
	for(i=scale_index_found; i>=0; i--)
	{
		for(j=0; j<nb_obj_found2[scale_index_found]; j++)
		{
			// Normally if (goodcenterX=0 so goodcenterY=0) or (goodcenterY=0 so goodcenterX=0) 
			if(goodcenterX[i][j] !=0 || goodcenterY[i][j] !=0)
			{
				position[count]=goodcenterX[i][j];
				position[count+1]=goodcenterY[i][j];
				position[count+2]=goodRadius[i][j];
				count=count+3;
			}
		}
	}


	// Create the offset for X and Y 
	offset_X=(int)(real_width/(float)(scale_index_found*1.2));
	offset_Y=(int)(real_height/(float)(scale_index_found*1.2));

	// Delete detections which are too close 
	for(i=0; i<NB_MAX_POINTS; i+=3)
	{
		for(j=3; j<NB_MAX_POINTS-i; j+=3)
		{
			if(position[i] != 0 && position[i+j] != 0 && position[i+1] != 0 && position[i+j+1] != 0)
			{
				if(offset_X >= abs(position[i]-position[i+j]) && offset_Y >= abs(position[i+1]-position[i+j+1]))
				{
					printf("Mphke sta positions!!!\n");
			
					position[i+j] = 0;
					position[i+j+1] = 0;
					position[i+j+2] = 0;
				}
			}
		}
	}	

	// Draw detection
	for(i=0; i<NB_MAX_POINTS; i+=3)
	{
		if(position[i] != 0 && position[i+1] != 0 && position[i+2] != 0)
		{
			printf("Mphke sto if...twra raster_rectangle\n");	

			raster_rectangle(result2[scale_index_found], (int)position[i], (int)position[i+1], (int)(position[i+2]/2), real_width);
		}
		
	}

	// Re-build the result image with highlighted detections
	for(i=0; i<real_height; i++)
	{
		for(j=0; j<real_width; j++)
		{
			if(result2[scale_index_found][i*real_width+j]!= 255)
			{
				result2[scale_index_found][i*real_width+j] = img[i*real_width+j];
			}
		}
	}
	
	int finalNb = 0;
	
	for(i=0; i<NB_MAX_POINTS; i+=3)
	{
		if (position[i]!=0) 
		{
			finalNb++;
		}
	}




printf("END\n");


end = clock();
detectionTime = (double)(end-start)/CLOCKS_PER_SEC * 1000;
TRACE_INFO(("\nExecution time = %f ms.\n", detectionTime));









	
	sprintf(result_name, "result_%d.pgm", finalNb);

	// Write the final result of the detection application 
	imgWrite((uint32_t *)result2[scale_index_found], result_name, height, width);
	
	
	// FREE ALL the allocations 
	releaseCascade(cascade);
	releaseCascade(cascade_scaled);
	free(feature_scaled);
	free(img);
	free(imgInt);
	free(imgSq);
	free(imgSqInt);
	free(result2);

	free(goodcenterX);
	free(goodcenterY);
	free(goodRadius);
	free(nb_obj_found2);
	free(goodPoints);
	free(imgInt_f);
	free(imgSqInt_f);

	TRACE_INFO(("\n FOUND %d OBJECTS \n",finalNb));
	return 0;
}

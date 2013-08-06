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
## Title:             violajones.c
##
## File:              CUDA file
## Author:            Teodora Petrisor <claudia-teodora.petrisor@thalesgroup.com>
##                    Matina Maria Trompouki  <mtrompou@ac.upc.edu>
## Description:       CUDA source file
##
## Modification:
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
## Porting into CUDA:
## Author:	      Matina Maria Trompouki  <mtrompou@ac.upc.edu>
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
* CUDA code: Matina Maria Trompouki
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
#include "cutil.h"
#include <assert.h>

/* Static Library */
#include "violajones.h"

#include "violajones_kernels.cu"

/******MACROS ********/
//#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#define INFO      0

#define MAX_BUFFERSIZE    256                    /* Maximum name file */
#define MAX_IMAGESIZE     1024                   /* Maximum Image Size */
#define MAX_BRIGHTNESS    255                    /* Maximum gray level */


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
	TRACE_INFO(("\n----------------------------------------------------------------------------------\n"));
	TRACE_INFO(("PGM image file input routine \n"));
	TRACE_INFO(("----------------------------------------------------------------------------------\n"));
	
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
		TRACE_INFO(("\n     Image width = %d, Image height = %d\n", x_size1, y_size1));
		TRACE_INFO(("     Maximum gray level = %d\n",max_gray));
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
		TRACE_INFO(("\n     Image width = %d, Image height = %d\n", x_size1, y_size1));
		TRACE_INFO(("     Maximum gray level = %d\n", max_gray));
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


/*Allocation function for GPU*/
CvHaarClassifierCascade* cudaAllocCascade_continuous()
{
	CvHaarClassifierCascade *dev_cc;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_cc, (unsigned) (sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature))));
        ERROR_CHECK
	
        if (dev_cc == NULL) {
               TRACE_INFO(("CUDA_ALLOCCASCADE: Couldn't allocate classifier cascade in the GPU\n"));
               exit(-1);
        }
	
	return dev_cc; 
}

/**Copies the entire cascade from the host to device**/
void copyCascadeFromHostToDevice(CvHaarClassifierCascade* cc_device, CvHaarClassifierCascade* cc_host, cudaStream_t *stream)
{	
	CUDA_SAFE_CALL(cudaMemcpyAsync(cc_device, cc_host, (unsigned) sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature),
				  cudaMemcpyHostToDevice, *stream));
        ERROR_CHECK	
}

/*** Allocation function for Classifier Cascade ***/
CvHaarClassifierCascade* allocCascade_continuous()
{
	int i = 0;
	int j = 0;
	int k = 0;

	CvHaarClassifierCascade *cc;

	cc = (CvHaarClassifierCascade *)malloc(sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));

/*
	cudaMallocHost((CvHaarClassifierCascade **)&cc, sizeof(CvHaarClassifierCascade)
					+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
					+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));
*/

	memset(cc,0,sizeof(CvHaarClassifierCascade)
			+ N_MAX_STAGES * sizeof(CvHaarStageClassifier)
			+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarClassifier)
			+ N_MAX_STAGES * N_MAX_CLASSIFIERS * sizeof(CvHaarFeature));

	cc->stageClassifier = (CvHaarStageClassifier*)(((char*)cc) + sizeof(CvHaarClassifierCascade));     


	for (i = 0; i < N_MAX_STAGES; i++)
	{
		cc->stageClassifier[i].classifier = (CvHaarClassifier*)(((char*)cc->stageClassifier) + (N_MAX_STAGES * sizeof(CvHaarStageClassifier)) + (i*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)));	

		
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			cc->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature*)(((char*)&(cc->stageClassifier[N_MAX_STAGES])) + (N_MAX_STAGES*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)) + (((i*N_MAX_CLASSIFIERS)+j)*sizeof(CvHaarFeature)));
			
			for (k = 0; k<2; k++)
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

/*** Allocation function for Classifier Cascade ***/
CvHaarClassifierCascade* allocCascade()
{
	int i = 0;
	int j = 0;
	int k = 0;

	CvHaarClassifierCascade *cc;

	cc = (CvHaarClassifierCascade *)malloc(sizeof(CvHaarClassifierCascade));
	cc->stageClassifier = (CvHaarStageClassifier *)calloc(N_MAX_STAGES,sizeof(CvHaarStageClassifier));
	for (i = 0; i < N_MAX_STAGES; i++)
	{
		cc->stageClassifier[i].classifier = (CvHaarClassifier *)calloc(N_MAX_CLASSIFIERS,sizeof(CvHaarClassifier));
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			cc->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature *)malloc(sizeof(CvHaarFeature));
			for (k = 0; k<2; k++)
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


void releaseCascade_continuous(CvHaarClassifierCascade *cc)
{
	free(cc);
	//cudaFreeHost(cc);
}

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
					cascade->stageClassifier[iStage].classifier[iNode].left = a;
					break;
				}                                
				case 'b':
				{
					sscanf(line,"%*s %f", &b);
					cascade->stageClassifier[iStage].classifier[iNode].right = b;
					break;
				}
				default:
				{
					isRect = 0;
					sscanf(line,"%f",&featThresh);
					cascade->stageClassifier[iStage].classifier[iNode].threshold = featThresh;
				}
			}
		}        
		*nStages = iStage+1;
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
void imgNormalize(uint32_t *imgIn, float *imgOut, float normFact, int height, int width)
{
	int irow = 0, icol = 0;
	int dim = 0;
	float meanImg = 0.0;

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

/*** Cast int image as float ****/
void imgCopy(uint32_t *imgIn, float *imgOut, int height, int width)
{
	int irow = 0, icol = 0;

	for (irow = 0; irow < height; irow++)
	{
		for (icol = 0; icol < width; icol++)
		{
			imgOut[irow*width+icol] = (float)imgIn[irow*width+icol];
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
__host__ __device__ float getImgIntPixel(float *img, int row, int col, int real_height, int real_width)
{
	float pval = 0.0;

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
__host__ __device__ float computeArea(float *img, int row, int col, int height, int width, int real_height, int real_width)
{
	float sum = 0.0;
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
			sum = getImgIntPixel((float *)img, row, col, real_height, real_width);
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
				//TRACE_INFO(("Error: \" This case is impossible!!!\"\n"));
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
__host__ __device__ void getRectangleParameters(CvHaarFeature *f, int iRectangle, int nRectangles, float scale, int rOffset, int cOffset, int *row, int *col, int *height, int *width)
{
	int r = 0, c = 0, h = 0, w = 0;

	w = f->rect[1].r.width;
	h = f->rect[1].r.height;

	if ((iRectangle > nRectangles) || (nRectangles < 2))
	{
		//TRACE_INFO(("Problem with rectangle index %d/%d or number of rectangles.\n", iRectangle, nRectangles));
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
__host__ __device__ void writeInFeature(int rowVect[4], int colVect[4], int hVect[4], int wVect[4], float weightVect[4], int nRects, CvHaarFeature *f_scaled)
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
__host__ __device__ void computeFeature(float *img, float *imgSq, CvHaarFeature *f, float *featVal, int irow, int icol, int height, int width, float scale, float scale_correction_factor, CvHaarFeature *f_scaled, int real_height, int real_width)
{
	int nRects = 0;
	int col = 0;
	int row = 0;
	int wRect = 0;
	int hRect = 0;
	int i = 0;
	//int rectArea = 0;
	int colVect[4] = {0};
	int rowVect[4] = {0};
	int wVect[4] = {0};
	int hVect[4] = {0};

	float w1 = 0.0; 
	//float w1_orig = 0.0;
	float rectWeight[4] = {0};

	float val = 0.0;
	float s[N_RECTANGLES_MAX] = {0};

	*featVal = 0.0;

	//w1_orig = f->rect[0].weight;
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
		s[i] = computeArea((float *)img, row, col, hRect, wRect, real_height, real_width);
		//rectArea = hRect*wRect;

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
__host__ __device__ float computeVariance(float *img, float *imgSq, int irow, int icol, int height, int width, int real_height, int real_width)
{
	int nPoints = 0;

	float s1 = 0.0;
	float s2 = 0.0;
	float f1 = 0.0;
	float f2 = 0.0;
	float varFact = 0.0;

	nPoints = height*width;

	s1 = (float)computeArea((float *)img, irow, icol, height, width, real_height, real_width);
	s2 = (float)computeArea((float *)imgSq, irow, icol, height, width, real_height, real_width);

	if(nPoints != 0)
	{
		f1 = (float)(s1/nPoints);
		f2 = (float)(s2/nPoints);
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

	uint32_t *new_variable = NULL;

	new_variable = (uint32_t *) malloc ((unsigned) (n * sizeof (uint32_t)));
	if (new_variable == NULL) {
		TRACE_INFO(("ALLOC_1D_UINT_32T: Couldn't allocate array of integer\n"));
		return (NULL);
	}
	return (new_variable);

}
//end function: alloc_1d_uint32_t **********************************************


uint32_t *cuda_alloc_1d_uint32_t(int n)
{
	uint32_t *dev_new_variable = NULL;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_new_variable, (unsigned) (n * sizeof (uint32_t))));
        ERROR_CHECK

	if (dev_new_variable == NULL) {
                TRACE_INFO(("ALLOC_1D_UINT_32T: Couldn't allocate array of integer in the GPU\n"));
                return (NULL);
        }

        return (dev_new_variable);
}

unsigned char *cuda_alloc_1d_unsigned_char(int n)
{
	unsigned char*dev_new_variable = NULL;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_new_variable, (unsigned) (n * sizeof (unsigned char))));
        ERROR_CHECK

	if (dev_new_variable == NULL) {
                TRACE_INFO(("ALLOC_1D_UNSIGNED_CHAR: Couldn't allocate array of chars in the GPU\n"));
                return (NULL);
        }

        return (dev_new_variable);
}

/*** Allocate one dimension float pointer ****/
float *alloc_1d_float(int n)
{

	float *new_variable;

	new_variable = (float *) malloc ((unsigned) (n * sizeof (float)));
	if (new_variable == NULL) {
		TRACE_INFO(("ALLOC_1D_DOUBLE: Couldn't allocate array of float\n"));
		return (NULL);
	}
	return (new_variable);

}
//end function: alloc_1d_float ************************************************

float *cuda_alloc_1d_float(int n)
{

	float *dev_new_variable;

        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_new_variable, (unsigned) (n * sizeof (float))));
        ERROR_CHECK

	if (dev_new_variable == NULL) {
                TRACE_INFO(("ALLOC_1D_DOUBLE: Couldn't allocate array of float in the GPU\n"));
                return (NULL);
        }
        return (dev_new_variable);
}

/*** Allocate 2d array of integers ***/
uint32_t **alloc_2d_uint32_t(int m, int n)
{
	int i;
	uint32_t **new_variable;

	new_variable = (uint32_t **) malloc ((unsigned) (m * sizeof (uint32_t *)));
	if (new_variable == NULL) {
		TRACE_INFO(("ALLOC_2D_UINT_32T: Couldn't allocate array of integer ptrs\n"));
		return (NULL);
	}

	for (i = 0; i < m; i++) {
		new_variable[i] = alloc_1d_uint32_t(n);
	}

	return (new_variable);
}
//end function: alloc_2d_uint32_t **********************************************

/* Draws simple or filled square */
__host__ __device__ void raster_rectangle(uint32_t* img, int x0, int y0, int radius, int real_width)
{
	int i=0;
	for(i=-radius/2; i<radius/2; i++)
	{
		img[i + x0 + (y0 + (int)(radius)) * real_width]=255;
		img[i + x0 + (y0 - (int)(radius)) * real_width]=255;
	}
	for(i=-(int)(radius); i<(int)(radius); i++)
	{
		img[(x0 + (int)(radius/2)) + (y0+i) * real_width]=255;
		img[(x0 - (int)(radius/2)) + (y0+i) * real_width]=255;
	}	
}
//end function: raster_rectangle **************************************************

float memcmp_for_float(const float *s1, const float *s2, size_t n_floats)
{
	int i=0;

	while(i++<n_floats)
	{
		if((float)*s1 == (float)*s2)
		{ 
			s1++;
			s2++;
			continue;
		}
		else
		{
			printf("\ns1: %f", *s1);			
			printf("\ns2: %f", *s2);			
			printf("\ns1-s2: %f", *s1-*s2);			
			printf("\ni: %d\n", i);			

			if(*s1<*s2)	
				return -1;
			else
				return 1;
		}

	}
		
	return 0;
}

/* ********************************** MAIN ********************************** */
int main( int argc, char** argv )
{
	// Timer declaration 
	time_t start, end;

	// Pointer declaration
	CvHaarClassifierCascade* cascade = NULL;
	CvHaarClassifierCascade* cascade_scaled = NULL;
	CvHaarFeature *feature_scaled = NULL;


	CvHaarClassifierCascade* dev_cascade = NULL;


	char *imgName = NULL;
	char *haarFileName = NULL;
	char result_name[MAX_BUFFERSIZE]={0};

	uint32_t *img = NULL;
	uint32_t *imgInt = NULL;
	uint32_t *imgSq = NULL;
	uint32_t *imgSqInt = NULL;
	uint32_t *result2 = NULL;

	float *cuda_imgInt_f = NULL;
	float *cuda_imgSqInt_f = NULL;
	
	uint32_t *cuda_imgInt = NULL;
        uint32_t *cuda_imgInt2 = NULL;
        uint32_t *cuda_imgInt3 = NULL;

	uint32_t *dev_img = NULL;
	uint32_t *dev_imgInt = NULL;
	uint32_t *dev_imgSq = NULL;
	uint32_t *dev_imgSqInt = NULL;	
	float *dev_imgInt_f = NULL;
	float *dev_imgSqInt_f = NULL;
	//unsigned char*dev_goodPoints = NULL;
	uint32_t *dev_goodcenterX = NULL;
	uint32_t *dev_goodcenterY = NULL;
	uint32_t *dev_goodRadius = NULL;
	uint32_t *dev_nb_obj_found2 = NULL;
	uint32_t *dev_position = NULL;
	uint32_t *dev_result2 = NULL;

//	Lock lock;
/*
	int *dev_foundObj = NULL;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_foundObj, (sizeof (int))));
        ERROR_CHECK

        if (dev_foundObj == NULL) {
               TRACE_INFO(("CUDA_ALLOC_FOUND_OBJ: Couldn't allocate foundObj GPU\n"));
               exit(-1);
        }
*/	
	int *dev_scale_index_found = NULL;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_scale_index_found, (sizeof (int))));
        ERROR_CHECK

        if (dev_scale_index_found == NULL) {
               TRACE_INFO(("CUDA_ALLOC_SCALE_INDEX_FOUND: Couldn't allocate scale_index_found GPU\n"));
               exit(-1);
        }
	


	int nb_obj_found=0;	

	uint32_t *goodcenterX=NULL;
	uint32_t *goodcenterY=NULL;
	uint32_t *goodRadius=NULL;
	uint32_t *nb_obj_found2=NULL;

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
	int nStages = 0; 
	//int *foundObj = 0;
	int nTileRows = 0;
	int nTileCols = 0;
	
	//int i = 0, j = 0;
	
	int real_height = 0, real_width = 0;
	int scale_index_found=0;

	int *dev_count = 0;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_count, (sizeof (int))));
        ERROR_CHECK

        if (dev_count == NULL) {
               TRACE_INFO(("CUDA_ALLOC_DEV_COUNT: Couldn't allocate dev_count in GPU\n"));
               exit(-1);
        }



	//int offset_X = 0, offset_Y = 0;
	//float scale_correction_factor = 0.0;
	
	// Factor Declaration 
	float scaleFactorMax = 0.0;
	float scaleStep = 1.1; // 10% increment of detector size per scale. Change this value to test other increments 
	float scaleFactor = 0.0;
	
	int total_scales = 0;

	float detectionTime = 0.0;

	// Integral Image Declaration 
	float *imgInt_f = NULL;
	float *imgSqInt_f = NULL;

	//Declare timer
	//unsigned int timer_compute=0;

	int block_size = 16;

	if (argc <= 2)
	{
		TRACE_INFO(("Usage: %s classifier image1 image2 ...\n", argv[0]));
		return(0);
	}

	// Get the Image name and the Cascade file name from the console 
	haarFileName=argv[1];

	TRACE_INFO(("\n----------------------------------------------------------------------------------\nSmart Camera application running.... \n----------------------------------------------------------------------------------\n"));

	// Start the clock counter 
	start = clock();

	//All the images _MUST_ have the same dimensions
	imgName=argv[2];
	// Get the Input Image informations 
	getImgDims(imgName, &width, &height);

	// Get back the Real Size of the Input Image 
	real_height=height;
	real_width=width;

	// Allocate the Cascade Memory 
	cascade = allocCascade_continuous();
	cascade_scaled = allocCascade_continuous(); 
	feature_scaled = (CvHaarFeature *)malloc(sizeof(CvHaarFeature));

	//Allocate the Cascade Memory for GPU
	dev_cascade = cudaAllocCascade_continuous();

	// Get the Classifier informations 
	readClassifCascade(haarFileName, cascade, &detSizeR, &detSizeC, &nStages);

	
	//Create streams used for ovelapping
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	copyCascadeFromHostToDevice(dev_cascade, cascade, &stream2);

	fix_links_cascade_continuous_memory<<<1,1,0,stream2>>>(dev_cascade);      
	ERROR_CHECK


	TRACE_INFO(("\n----------------------------------------------------------------------------------\n"));
	TRACE_INFO(("Classifier file input routine \n"));
	TRACE_INFO(("----------------------------------------------------------------------------------\n\n"));
	TRACE_INFO(("     Number of Stages = %d\n", nStages));
	TRACE_INFO(("     Original Feature Height = %d\n", detSizeR));
	TRACE_INFO(("     Original Feature Width = %d\n", detSizeC));
	// Determine the Max Scale Factor
	if (detSizeR != 0 && detSizeC != 0)
	{
		scaleFactorMax = min((int)floor(height/detSizeR), (int)floor(width/detSizeC));
	}

	for (scaleFactor = 1; scaleFactor <= scaleFactorMax; scaleFactor *= scaleStep)
        {
               total_scales++;
        }
	
	int *dev_foundObj = NULL;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_foundObj, (sizeof(int) * total_scales)));
        ERROR_CHECK

        if (dev_foundObj == NULL) {
               TRACE_INFO(("CUDA_ALLOC_FOUND_OBJ: Couldn't allocate foundObj GPU\n"));
               exit(-1);
        }

	int *dev_nb_obj_found = NULL;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_nb_obj_found, (sizeof (int)*total_scales)));
        ERROR_CHECK

        if (dev_nb_obj_found == NULL) {
               TRACE_INFO(("CUDA_ALLOC_NB_OBJ_FOUND: Couldn't allocate dev_nb_obj_found GPU\n"));
               exit(-1);
        }
	

	// Give the Allocation size 
	img=alloc_1d_uint32_t(width*height);
	imgInt=alloc_1d_uint32_t(width*height);
	imgSq=alloc_1d_uint32_t(width*height);
	imgSqInt=alloc_1d_uint32_t(width*height);
	result2 = alloc_1d_uint32_t(nStages*width*height);

	//images returned from GPU
	cuda_imgInt_f = alloc_1d_float(width*height);
	cuda_imgSqInt_f = alloc_1d_float(width*height); 

	//CUDA allocations
	dev_img = cuda_alloc_1d_uint32_t(width*height);
	dev_imgInt = cuda_alloc_1d_uint32_t(width*height);	
	dev_imgSq = cuda_alloc_1d_uint32_t(width*height);
	dev_imgSqInt = cuda_alloc_1d_uint32_t(width*height);
	//dev_goodPoints = cuda_alloc_1d_unsigned_char(width*height*total_scales);
	//dev_goodPoints = cuda_alloc_1d_uint32_t(width*height);
	dev_imgInt_f = cuda_alloc_1d_float(width*height);
	dev_imgSqInt_f = cuda_alloc_1d_float(width*height);
	dev_goodcenterX=cuda_alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);
	dev_goodcenterY=cuda_alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);
	dev_goodRadius=cuda_alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);
	dev_nb_obj_found2 = cuda_alloc_1d_uint32_t(nStages);
	dev_position = cuda_alloc_1d_uint32_t(width*height);
	dev_result2 = cuda_alloc_1d_uint32_t(nStages*width*height);


	//images returned from GPU
        cuda_imgInt=alloc_1d_uint32_t(width*height);
        cuda_imgInt2=alloc_1d_uint32_t(width*height);
        cuda_imgInt3=alloc_1d_uint32_t(width*height);	

	nb_obj_found2=alloc_1d_uint32_t(nStages);

	goodcenterX=alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);
	goodcenterY=alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);
	goodRadius=alloc_1d_uint32_t(nStages*NB_MAX_DETECTION);

	goodPoints=alloc_1d_uint32_t(width*height);
	imgInt_f=alloc_1d_float(width*height);
	imgSqInt_f=alloc_1d_float(width*height);

	//Loop for all the images
	//--------------------------------------------------------------------------------
	
	for(int image_counter=0; image_counter < argc-2; image_counter++)
	{	
	
	imgName=argv[image_counter+2];
	
	//printf("     Number of arg: %d %s\n",image_counter+2, imgName);
	
	//Memsets

	CUDA_SAFE_CALL(cudaMemset(dev_scale_index_found, 0, sizeof(int)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_count, 0, sizeof(int)));
        ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_goodcenterX, 0, (sizeof(uint32_t)*nStages*NB_MAX_DETECTION)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_goodcenterY, 0, (sizeof(uint32_t)*nStages*NB_MAX_DETECTION)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_goodRadius, 0, (sizeof(uint32_t)*nStages*NB_MAX_DETECTION)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_nb_obj_found2, 0, (sizeof(uint32_t)*nStages)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_position, 0, (sizeof(uint32_t)*width*height)));
	ERROR_CHECK

	CUDA_SAFE_CALL(cudaMemset(dev_result2, 0, (sizeof(uint32_t)*nStages*width*height)));
	ERROR_CHECK

	// load the Image in Memory 
	load_image_check((uint32_t *)img, (char *)imgName, width, height);

	

//	CUDA_SAFE_CALL(cudaMemcpy(dev_img, img, sizeof(uint32_t)*(width*height), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyAsync(dev_img, img, sizeof(uint32_t)*(width*height), cudaMemcpyHostToDevice, stream1));
	ERROR_CHECK



        dim3 block(block_size); 

	//+++++++++++++++++++++ FOR THE ROWS THREAD IMPLEMENTATION +++++++
	//This implementation works only for height multiple to block_size!
	//assert(height % block_size==0);
        dim3 grid_row((height+block_size-1)/block_size);   

	//++++++++++++++++++++ FOR THE COLUMNS THREAD IMPLEMENTATION ++++
	//This implementation works only for width multiple to block_size!
	//assert(width % block_size==0);
	dim3 grid_column((width+block_size-1)/block_size);

	computeIntegralImgRowCuda<<<grid_row,block,0,stream1>>>((uint32_t *)dev_img, (uint32_t *)dev_imgInt, width, height);
	ERROR_CHECK

	computeIntegralImgColCuda<<<grid_column,block,0,stream1>>>((uint32_t *)dev_imgInt, width, height);      
	ERROR_CHECK
/*
	//Square image computation with ROWS----------------------------

        computeSquareImageCuda_rows<<<block, grid_row>>>((uint32_t *)dev_img, (uint32_t *)dev_imgSq, width,height);
        ERROR_CHECK

*/	
	//Square image computation with COLUMNS-------------------------

	computeSquareImageCuda_cols<<<grid_column,block, 0,stream1>>>((uint32_t *)dev_img, (uint32_t *)dev_imgSq, height, width);
        ERROR_CHECK


        computeIntegralImgRowCuda<<<grid_row,block,0,stream1>>>((uint32_t *)dev_imgSq, (uint32_t *)dev_imgSqInt, width, height);
        ERROR_CHECK

        computeIntegralImgColCuda<<<grid_column,block,0,stream1>>>((uint32_t *)dev_imgSqInt, width, height);
        ERROR_CHECK

	//Synchronize the streams
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

//Transfer integral image, dotsquare image and dotsquare integral image back to host
/*
	CUDA_SAFE_CALL(cudaMemcpy(cuda_imgInt, dev_imgInt, sizeof(uint32_t)*(width*height), cudaMemcpyDeviceToHost));
	ERROR_CHECK
        
	CUDA_SAFE_CALL(cudaMemcpy(cuda_imgInt2, dev_imgSq, sizeof(uint32_t)*(width*height), cudaMemcpyDeviceToHost));
	ERROR_CHECK
        
	CUDA_SAFE_CALL(cudaMemcpy(cuda_imgInt3, dev_imgSqInt, sizeof(uint32_t)*(width*height), cudaMemcpyDeviceToHost));
	ERROR_CHECK
*/


/*-------------------------------------------------------------------------------------------------------*/
/*					   CLASSIFICATION PHASE						 */
/*-------------------------------------------------------------------------------------------------------*/


	// Copy the Image to float array 
	imgCopy((uint32_t *)imgInt, (float *)imgInt_f, height, width);
	imgCopy((uint32_t *)imgSqInt, (float *)imgSqInt_f, height, width);

	imgCopyCuda<<<(width*height)/128, 128>>>((uint32_t *)dev_imgInt, (float *)dev_imgInt_f, height, width);
        ERROR_CHECK

	imgCopyCuda<<<(width*height)/128, 128>>>((uint32_t *)dev_imgSqInt, (float *)dev_imgSqInt_f, height, width);	
        ERROR_CHECK

/*
	CUDA_SAFE_CALL(cudaMemcpy(cuda_imgInt_f, dev_imgInt_f, sizeof(float)*(width*height), cudaMemcpyDeviceToHost));
	ERROR_CHECK

        CUDA_SAFE_CALL(cudaMemcpy(cuda_imgSqInt_f, dev_imgSqInt_f, sizeof(float)*(width*height), cudaMemcpyDeviceToHost));
        ERROR_CHECK
*/
	
	TRACE_INFO(("\n----------------------------------------------------------------------------------\n"));
	TRACE_INFO(("Processing scales routine \n"));
	TRACE_INFO(("----------------------------------------------------------------------------------\n\n"));

	int num_pixels = real_width*real_height;

	dim3 block_good((real_width*real_height)/128, total_scales);
	dim3 thread_good(128);

	//initializeGoodPointsCuda<<<block_good, thread_good>>>((uint32_t *)dev_goodPoints, num_pixels);
	//ERROR_CHECK
	//CUDA_SAFE_CALL(cudaMemset(dev_goodPoints, 255, total_scales*width*height));
	//ERROR_CHECK

	// Launch the Main Loop 
//	for (scaleFactor = 1; scaleFactor <= scaleFactorMax; scaleFactor *= scaleStep)
//	{
		//int num_pixels = real_width*real_height;
		
		//Initializing the goodPoints

		//initializeGoodPointsCuda<<<(real_width*real_height)/128, 128>>>((uint32_t *)dev_goodPoints, num_pixels);
        	//ERROR_CHECK

		scaleFactor = 1;

		tileWidth = (int)floor(detSizeC * scaleFactor + 0.5);
		tileHeight = (int)floor(detSizeR * scaleFactor + 0.5);
		rowStep = max(2, (int)floor(scaleFactor + 0.5));
		colStep = max(2, (int)floor(scaleFactor + 0.5));

    		//(TP) according to openCV: added correction by reducing scaled detector window by 2 pixels in each direction
		
//		scale_correction_factor = (float)1.0/(float)((int)floor((detSizeC-2)*scaleFactor)*(int)floor((detSizeR-2)*scaleFactor)); 

		// compute number of tiles: difference between the size of the Image and the size of the Detector 
		nTileRows = height-tileHeight;
		nTileCols = width-tileWidth;

/*		foundObj = 0;
		
		CUDA_SAFE_CALL(cudaMemcpy(dev_foundObj, &foundObj, sizeof(int), cudaMemcpyHostToDevice));
		ERROR_CHECK*/

         	CUDA_SAFE_CALL(cudaMemset(dev_foundObj, 0, (sizeof(int) * total_scales)));
         	ERROR_CHECK


		int irowiterations = 0;
		int icoliterations = 0;
		int number_of_threads = 0;

		//Prepei na exw to nTileRows, nTileCols, colStep, rowStep!!!!

		irowiterations = (int)ceilf((float)nTileRows/rowStep);
		icoliterations = (int)ceilf((float)nTileCols/colStep);

		number_of_threads = irowiterations*icoliterations;

	int block_size_sub = 64;

	//printf("total scales:%d\n", total_scales);
	dim3 block_subwindow((number_of_threads+(block_size_sub-1))/block_size_sub, total_scales);
	dim3 thread_subwindow(block_size_sub);


	CUDA_SAFE_CALL(cudaMemset(dev_nb_obj_found, 0, sizeof(int)*total_scales));
	ERROR_CHECK

	subwindow_find_candidates<<<block_subwindow,thread_subwindow>>>(nStages, dev_cascade, /*dev_goodPoints,*/ real_width, dev_imgInt_f, dev_imgSqInt_f, real_height, dev_foundObj, dev_nb_obj_found, detSizeC, detSizeR, dev_goodcenterX, dev_goodcenterY, dev_goodRadius, dev_scale_index_found, dev_nb_obj_found2);

	ERROR_CHECK


		CUDA_SAFE_CALL(cudaMemcpy(&scale_index_found, dev_scale_index_found, sizeof(int), cudaMemcpyDeviceToHost));
		ERROR_CHECK



//	}	
	// Done processing all scales 


	// Timer end 
	end = clock();

	// Timer calculation (for detection time) and convert in ms 
	detectionTime = (float)(end-start)/CLOCKS_PER_SEC * 1000;

	TRACE_INFO(("     Finished processing tiles up to (%d/%d,%d/%d) position. Detection time = %f ms.\n",
				nTileRows-1, height, nTileCols-1, width, detectionTime));


	if (scale_index_found)
		TRACE_INFO(("\n----------------------------------------------------------------------------------\nHandling multiple detections\n----------------------------------------------------------------------------------\n"));

	//printf("Scale Index Found:%d\n", scale_index_found);

	if (scale_index_found)
	kernel_one<<<1,scale_index_found>>>(dev_scale_index_found, dev_nb_obj_found2);
	ERROR_CHECK
	
	
	kernel_two_alt<<<1,1>>>(dev_scale_index_found, dev_nb_obj_found2, dev_goodcenterX, dev_goodcenterY, dev_goodRadius, dev_position, dev_count);
	ERROR_CHECK
	
/*	
	int irowiterations = 0;
	int icoliterations = 0;
	int number_of_threads = 0;
*/	
	irowiterations = (int)ceilf((float)NB_MAX_POINTS/3);
	icoliterations = (int)ceilf((float)NB_MAX_POINTS/3);
	
	number_of_threads = irowiterations*icoliterations;
	
	kernel_three<<<(number_of_threads+127)/128,128>>>(dev_position, dev_scale_index_found, real_width, real_height);
	ERROR_CHECK
	
	kernel_draw_detection<<<(irowiterations+127)/128,128>>>(dev_position, dev_scale_index_found, real_width, dev_result2, width*height);
	ERROR_CHECK
	
	
	number_of_threads = real_width * real_height;
	
	kernel_highlight_detection<<<(number_of_threads+127)/128,128>>>(dev_img, dev_scale_index_found, real_width, real_height, dev_result2, width*height);
	ERROR_CHECK
	
	CUDA_SAFE_CALL(cudaMemcpy(result2, dev_result2, sizeof(uint32_t)*nStages*width*height, cudaMemcpyDeviceToHost));
	ERROR_CHECK

	
	int finalNb = 0;
		
	int *dev_finalNb = NULL;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_finalNb, (sizeof (int))));
	ERROR_CHECK
	
	if (dev_finalNb == NULL) {
	       TRACE_INFO(("CUDA_ALLOC_FINALNB: Couldn't allocate finalNb in GPU\n"));
	       exit(-1);
	}
	
	CUDA_SAFE_CALL(cudaMemset(dev_finalNb, 0, sizeof(int)));
	ERROR_CHECK
	
	kernel_finalNb<<<(irowiterations+127)/128,128>>>(dev_finalNb, dev_position);
	ERROR_CHECK
	
	CUDA_SAFE_CALL(cudaMemcpy(&finalNb, dev_finalNb, sizeof(int), cudaMemcpyDeviceToHost));
	ERROR_CHECK


	sprintf(result_name, "gpu_result_%s", imgName);

	// Write the final result of the detection application 
	imgWrite((uint32_t *)&(result2[scale_index_found*width*height]), result_name, height, width);
	
	TRACE_INFO(("\n     FOUND %d OBJECTS \n",finalNb));
	TRACE_INFO(("\n----------------------------------------------------------------------------------\nSmart Camera application ended OK! Check %s file!\n----------------------------------------------------------------------------------\n", result_name));
	
	} //for of all images

	// FREE ALL the allocations
 
	releaseCascade_continuous(cascade);	
	releaseCascade_continuous(cascade_scaled);
	
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
	
	free(cuda_imgInt);
	free(cuda_imgInt2);
	free(cuda_imgInt3);
	free(cuda_imgInt_f);
	free(cuda_imgSqInt_f);


	TRACE_INFO(("\n----------------------------------------------------------------------------------\n%d images processed!\n----------------------------------------------------------------------------------\n",argc-2));
	
	return 0;
}

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
###############################################################################


###############################################################################
##
## Author:	      Matina Maria Trompouki  <mtrompou@ac.upc.edu>
##
###############################################################################
*/

#ifndef _VIOLAJONES_KERNEL_H_
#define _VIOLAJONES_KERNEL_H_

#include "violajones.h"

__global__ void
computeIntegralImgRowCuda(uint32_t *imgIn, uint32_t *imgOut, int width)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;

	int row_sum=0;	

	for(int i=0; i<width; i++)
	{
		row_sum += imgIn[row*width+i];
		imgOut[row*width+i] = row_sum; 
	}

	
}


__global__ void
computeIntegralImgColCuda(uint32_t *imgOut, int width, int height)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;

        int col_sum=0;

        for(int i=0; i<height; i++)
        {
                col_sum += imgOut[col+i*width];
                imgOut[col+i*width] = col_sum;
        }


}


__global__ void
computeSquareImageCuda_rows(uint32_t *imgIn, uint32_t *imgOut, int width)
{
        int row = blockDim.x * blockIdx.x + threadIdx.x;

        for(int i=0; i<width; i++)
        {
                imgOut[row*width+i] = imgIn[row*width+i] * imgIn[row*width+i];
        }


}

__global__ void
computeSquareImageCuda_cols(uint32_t *imgIn, uint32_t *imgOut, int height, int width)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;

        for(int i=0; i<height; i++)
        {
                imgOut[col+width*i] = imgIn[col+width*i] * imgIn[col+width*i];
        }


}


__global__ void
fix_links_cascade_continuous_memory(CvHaarClassifierCascade* dev_cascade)
{
	int i = 0;
	int j = 0;

	dev_cascade->stageClassifier = (CvHaarStageClassifier*)(((char*)dev_cascade) + sizeof(CvHaarClassifierCascade));     

	for (i = 0; i < N_MAX_STAGES; i++)
	{
		dev_cascade->stageClassifier[i].classifier = (CvHaarClassifier*)(((char*)dev_cascade->stageClassifier) + (N_MAX_STAGES * sizeof(CvHaarStageClassifier)) + (i*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)));	
		for(j = 0; j < N_MAX_CLASSIFIERS; j++)
		{
			dev_cascade->stageClassifier[i].classifier[j].haarFeature = (CvHaarFeature*)(((char*)&(dev_cascade->stageClassifier[N_MAX_STAGES])) + (N_MAX_STAGES*N_MAX_CLASSIFIERS*sizeof(CvHaarClassifier)) + (((i*N_MAX_CLASSIFIERS)+j)*sizeof(CvHaarFeature)));
			
		}
	}
}


__global__ void
imgCopyCuda(uint32_t *dev_imgIn, float *dev_imgOut, int height, int width)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(index < height*width)
	{	
		dev_imgOut[index] = (float)dev_imgIn[index];
	}
}



__global__ void
initializeGoodPointsCuda(uint32_t *dev_goodPoints, int num_pixels)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < num_pixels)
	{
		dev_goodPoints[index] = 255;
	}
}



__global__ void
subwindow_find_candidates(int nStages, CvHaarClassifierCascade *dev_cascade, uint32_t *dev_goodPoints, int nTileRows, int nTileCols, int rowStep, int colStep, int real_width, float *dev_imgInt_f, float *dev_imgSqInt_f, int tileHeight, int tileWidth, int real_height, float scaleFactor, float scale_correction_factor, int *dev_foundObj, int *dev_nb_obj_found)
{
	
	int iStage = 0;
	int nNodes = 0;
	int iNode = 0;
	float sumClassif = 0.0;
	float varFact = 0.0;	
	float featVal = 0.0;

	CvHaarFeature feature_scaled;

	float thresh = 0.0;
	float a = 0.0;
	float b = 0.0;

	int stride = (nTileCols + colStep -1 )/colStep;

	int counter = blockDim.x * blockIdx.x + threadIdx.x;

	int irow = (counter / stride)*rowStep;

	int icol = (counter % stride)*colStep;

	if ((irow < nTileRows) && (icol < nTileCols))
	{

		for (iStage = 0; iStage < nStages; iStage++)
		{
			nNodes = dev_cascade->stageClassifier[iStage].count;
	
	
			if (dev_goodPoints[irow*real_width+icol])
			{
				sumClassif = 0.0;
				//Operation used for every Stage of the Classifier 
				varFact=computeVariance((float *)dev_imgInt_f, (float *)dev_imgSqInt_f, irow, icol, tileHeight, tileWidth, real_height, real_width);
	
				if (varFact < 10e-15)
				{
					// this should not occur (possible overflow BUG)
					varFact = 1.0; 
					dev_goodPoints[irow*real_width+icol] = 0; 
					break;
				}
				else
				{
					// Get the standard deviation 
					varFact = sqrt(varFact);
				}
				
				for (iNode = 0; iNode < nNodes; iNode++)
				{
					computeFeature((float *)dev_imgInt_f, (float *)dev_imgSqInt_f, dev_cascade->stageClassifier[iStage].classifier[iNode].haarFeature, &featVal, irow, icol, tileHeight, tileWidth, scaleFactor, scale_correction_factor, &feature_scaled, real_height, real_width);
					
					// Get the thresholds for every Node of the stage 
					thresh = dev_cascade->stageClassifier[iStage].classifier[iNode].threshold;
					a = dev_cascade->stageClassifier[iStage].classifier[iNode].left;
					b = dev_cascade->stageClassifier[iStage].classifier[iNode].right;
					sumClassif += (featVal < (float)(thresh*varFact) ? a : b);
				}
				
				// Update goodPoints according to detection threshold 
				if (sumClassif < dev_cascade->stageClassifier[iStage].threshold)
				{
					dev_goodPoints[irow*real_width+icol] = 0;
				}	  
				else
				{	
					if (iStage == nStages - 1)
					{
						atomicAdd(dev_foundObj, 1);
					}
				}	  
			}
	
		}

	}


	__threadfence();
	
	if(irow==0 && icol==0)
	{
		*dev_nb_obj_found=0;
	}


}



__global__ void
subwindow_examine_candidates(Lock lock, uint32_t *dev_goodPoints, int nTileRows, int nTileCols, int rowStep, int colStep, int real_width, int tileHeight, int tileWidth, float scaleFactor, int *dev_foundObj, uint32_t *dev_goodcenterX, uint32_t *dev_goodcenterY, uint32_t *dev_goodRadius, int *dev_scale_index_found, int *dev_nb_obj_found, uint32_t *dev_nb_obj_found2)
{
	float centerX=0.0;
	float centerY=0.0;
	float radius=0.0;
       	
	float centerX_tmp=0.0;
	float centerY_tmp=0.0;
	float radius_tmp=0.0;
	
	int threshold_X=0;
	int threshold_Y=0;

	// Determine used object 
       	if (*dev_foundObj)
       	{	
		int stride = (nTileCols + colStep -1 )/colStep;
		int counter = blockDim.x * blockIdx.x + threadIdx.x;
		int irow = (counter / stride)*rowStep;
		int icol = (counter % stride)*colStep;

		if ((irow < nTileRows) && (icol < nTileCols))
		{
			// Only the detection is used 
			if (dev_goodPoints[irow*real_width+icol])
			{
				// Calculation of the Center of the detection 
				centerX=(((tileWidth-1)*0.5+icol));
				centerY=(((tileHeight-1)*0.5+irow));
				
				//Calculation of the radius of the circle surrounding object 
				radius = sqrt(pow((float)tileHeight-1, 2)+pow((float)tileWidth-1, 2))/2;

				//Threshold calculation: proportionnal to the size of the Detector 
				threshold_X=(int)((tileHeight-1)/(2*scaleFactor));
				threshold_Y=(int)((tileWidth-1)/(2*scaleFactor));

				//Reduce number of detections in a given range 

				int dev_nb_obj_found_tmp = (*dev_scale_index_found)*NB_MAX_DETECTION + ((*dev_nb_obj_found)?(*dev_nb_obj_found)-1:0);
				
				
				if(centerX > (dev_goodcenterX[dev_nb_obj_found_tmp]+threshold_X)
				||
				centerX < (dev_goodcenterX[dev_nb_obj_found_tmp]-threshold_X)
				||
				centerY > (dev_goodcenterY[dev_nb_obj_found_tmp]+threshold_Y)
				||
				centerY < (dev_goodcenterY[dev_nb_obj_found_tmp]-threshold_Y))
				{

					centerX_tmp=centerX;
					centerY_tmp=centerY;
					radius_tmp=radius;
					// Get only the restricted Good Points and get back the size for each one 

					int nb_obj_found_tmp = atomicAdd(dev_nb_obj_found, 1);
					int dev_scale_index_found_tmp = ((*dev_scale_index_found)?(*dev_scale_index_found)-1:0)*NB_MAX_DETECTION + (nb_obj_found_tmp);


				        dev_goodcenterX[dev_scale_index_found_tmp]=centerX_tmp;
				        dev_goodcenterY[dev_scale_index_found_tmp]=centerY_tmp;
					dev_goodRadius[dev_scale_index_found_tmp]=radius_tmp;


					atomicMax(&(dev_nb_obj_found2[(*dev_scale_index_found)?(*dev_scale_index_found)-1:0]), nb_obj_found_tmp+1); 
			
					}
			}
		}


		__threadfence();

		if(irow==0 && icol==0)
			atomicAdd(dev_scale_index_found, 1);

  
     	}
}




__global__ void
kernel_one(int *dev_scale_index_found, uint32_t *dev_nb_obj_found2)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(index < (*dev_scale_index_found))
	{	
		atomicMax(&dev_nb_obj_found2[*dev_scale_index_found], dev_nb_obj_found2[index]);
	}
}


__global__ void
kernel_two_alt(int *dev_scale_index_found, uint32_t *dev_nb_obj_found2, uint32_t *dev_goodcenterX, uint32_t *dev_goodcenterY, uint32_t *dev_goodRadius, uint32_t *dev_position, int *dev_count)
{
	for(int i=*dev_scale_index_found; i>=0; i--)
	{
		for(int j=0; j<dev_nb_obj_found2[*dev_scale_index_found]; j++)
		{
			if(dev_goodcenterX[i*NB_MAX_DETECTION+j] !=0 || dev_goodcenterY[i*NB_MAX_DETECTION+j] !=0)
			{
				dev_position[*dev_count]=dev_goodcenterX[i*NB_MAX_DETECTION+j];
				dev_position[*dev_count+1]=dev_goodcenterY[i*NB_MAX_DETECTION+j];
				dev_position[*dev_count+2]=dev_goodRadius[i*NB_MAX_DETECTION+j];
				*dev_count=*dev_count+3;
			}
		}
	}
}



__global__ void
kernel_three(uint32_t *dev_position, int *dev_scale_index_found, int real_width, int real_height)
{
	int offset_X = 0;
	int offset_Y = 0;

	// Create the offset for X and Y 
	offset_X=(int)(real_width/(float)((*dev_scale_index_found)*1.2));
	offset_Y=(int)(real_height/(float)((*dev_scale_index_found)*1.2));

	int stride = (NB_MAX_POINTS + 2 )/3;
	int counter = blockDim.x * blockIdx.x + threadIdx.x;
	int i = (counter / stride)*3;
	int j = ((counter % stride)*3) + 3;

	if ((i < NB_MAX_POINTS) && (j < NB_MAX_POINTS-i))
	{

		if(dev_position[i] != 0 && dev_position[i+j] != 0 && dev_position[i+1] != 0 && dev_position[i+j+1] != 0)
		{
			if((float)offset_X >= abs((float)dev_position[i]-dev_position[i+j]) && (float)offset_Y >= abs((float)dev_position[i+1]-dev_position[i+j+1]))
			{
				dev_position[i+j] = 0;
				dev_position[i+j+1] = 0;
				dev_position[i+j+2] = 0;
			}
		}
	}
}



__global__ void
kernel_draw_detection(uint32_t *dev_position, int *dev_scale_index_found, int real_width, uint32_t *dev_result2, int width_height)
{
	int counter = blockDim.x * blockIdx.x + threadIdx.x;
	int i = counter * 3;

	// Draw detection
	if(i < NB_MAX_POINTS)
	{
		if(dev_position[i] != 0 && dev_position[i+1] != 0 && dev_position[i+2] != 0)
		{
			raster_rectangle(&(dev_result2[(*dev_scale_index_found)*width_height]), (int)dev_position[i], (int)dev_position[i+1], (int)(dev_position[i+2]/2), real_width);
		}
		
	}
}



__global__ void
kernel_highlight_detection(uint32_t *imgIn, int *dev_scale_index_found, int real_width, int real_height, uint32_t *dev_result2, int width_height)
{
	// Re-build the result image with highlighted detections
	
	int counter = blockDim.x * blockIdx.x + threadIdx.x;
	int i = counter / real_width;
	int j = counter % real_width;

	if ((i < real_height) && (j < real_width))
	{
		if(dev_result2[((*dev_scale_index_found)*width_height) + (i*real_width+j)]!= 255)
		{
			dev_result2[((*dev_scale_index_found)*width_height)+(i*real_width+j)] = imgIn[i*real_width+j];
		}
	}
}


__global__ void
kernel_finalNb(int *dev_finalNb, uint32_t *dev_position)
{
	int counter = blockDim.x * blockIdx.x + threadIdx.x;
	int i = counter * 3;

	if(i < NB_MAX_POINTS)
	{
		if (dev_position[i]!=0) 
		{
			atomicAdd(dev_finalNb, 1);
		}
	}
}


#endif



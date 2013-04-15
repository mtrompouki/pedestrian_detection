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
## Title:             Makefile pedestrian detection application
##
## File:              Makefile
## Authors:           Paul Brelet  <paul.brelet@thalesgroup.com>
##                    Matina Maria Trompouki  <mtrompou@ac.upc.edu>
##
###############################################################################

all: violajones

violajones.o: violajones.cu violajones.h
	nvcc -arch=sm_11 -g -c violajones.cu 

violajones:	 violajones.o
	nvcc -g -lm -L. -lcutil -lcudart -o violajones violajones.o 

clean:
	rm -rf violajones *.o

run: all
	./violajones classifier.txt person_015.pgm 

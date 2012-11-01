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
## Title:             Makefile pedestrian detection application
##
## File:              Makefile
## Author:            Paul Brelet  <paul.brelet@thalesgroup.com>
##
###############################################################################

all: violajones

violajones: violajones.c violajones.h
#	gcc -ftree-vectorize -ftree-loop-linear -ftree-loop-im -funroll-loops -ffast-math -ftrapv -o3 -g -Wall -Werror -lm -o violajones violajones.c
	gcc -g -Wall -lm -o violajones violajones.c

clean:
	rm -rf *.o violajones

run: all
	./violajones person_015.pgm classifier.txt

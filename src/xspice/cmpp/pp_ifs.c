/*============================================================================
FILE  pp_ifs.c

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the main function for processing an Interface Spec
    File (ifspec.ifs).

INTERFACES

    preprocess_ifs_file()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <stdlib.h>
#include  "cmpp.h"



/* *********************************************************************** */


/*
preprocess_ifs_file

Function preprocess_ifs_file is the top-level driver function for
preprocessing an Interface Specification file (ifspec.ifs).  This
function calls read_ifs_file() requesting it to read and parse
the Interface Specification file and place the information
contained in it into an internal data structure.  Then
write_ifs_c_file() is called to write the information out in a C
file that will be compiled and linked with the simulator.
*/


void preprocess_ifs_file(void)
{

    Ifs_Table_t     ifs_table;   /* Repository for info read from ifspec.ifs file */

    Status_t        status;      /* Return status */


    /* Read the entire ifspec.ifs file and load the data into ifs_table */

    status = read_ifs_file(IFSPEC_FILENAME,GET_IFS_TABLE,&ifs_table);

    if(status != OK) {
        exit(1);
    }


    /* Write the ifspec.c file required by the spice simulator */

    status = write_ifs_c_file("ifspec.c",&ifs_table);

    if(status != OK) {
        exit(1);
    }

}




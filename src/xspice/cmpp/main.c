/*============================================================================
FILE  main.c

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

    This file contains the top-level function for the Code Model
    PreProcessor (cmpp).  It handles reading the command-line
    arguments, and then vectors to an appropriate function.

INTERFACES

    main()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include  "cmpp.h"


#define USAGE_MSG "Usage:  cmpp [-ifs] [-mod [<filename>]] [-lst]"
#define TOO_FEW_ARGS "ERROR - Too few arguments"
#define TOO_MANY_ARGS "ERROR - Too many arguments"
#define UNRECOGNIZED_ARGS "ERROR - Unrecognized argument"



/* *********************************************************************** */


/*
main

Function main checks the validity of the command-line arguments
supplied when the program is invoked and calls one of the three
major functions as appropriate:

     preprocess_ifs_file   Process Interface Specification File.  
     preprocess_mod_file   Process Model Definition File.         
     preprocess_lst_file   Process Pathname List Files.           

depending on the argument.
*/

int main(
         int argc,      /* Number of command line arguments */
         char *argv[])  /* Command line argument text */
{

   init_error (argv[0]);

   /* Process command line arguments and vector to appropriate function */

   if(argc < 2) {
      print_error(TOO_FEW_ARGS);
      print_error(USAGE_MSG);
      exit(1);
   }

   if(strcmp(argv[1],"-ifs") == 0) {
      if(argc == 2) {
         preprocess_ifs_file();
      }
      else {
         print_error(TOO_MANY_ARGS);
         print_error(USAGE_MSG);
         exit(1);
      }
   }
   else if(strcmp(argv[1],"-lst") == 0) {
      if(argc == 2) {
         preprocess_lst_files();
      }
      else {
         print_error(TOO_MANY_ARGS);
         print_error(USAGE_MSG);
         exit(1);
      }
   }
   else if(strcmp(argv[1],"-mod") == 0) {
      if(argc == 2) {
         preprocess_mod_file("cfunc.mod");
      }
      else if(argc == 3) {
         preprocess_mod_file(argv[2]);
      }
      else {
         print_error(TOO_MANY_ARGS);
         print_error(USAGE_MSG);
         exit(1);
      }
   }
   else {
      print_error(UNRECOGNIZED_ARGS);
      print_error(USAGE_MSG);
      exit(1);
   }

   exit(0);
}


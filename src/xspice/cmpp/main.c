/*============================================================================
FILE  main.c

MEMBER OF process cmpp

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
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
#ifdef DEBUG_CMPP
#ifdef _WIN32
#include <windows.h>
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include  "cmpp.h"


#define USAGE_MSG \
        "Usage:  cmpp [-ifs] | [-mod [<filename>]] | [-lst] | [-p fn.lst]"
#define TOO_FEW_ARGS "ERROR - Too few arguments"
#define TOO_MANY_ARGS "ERROR - Too many arguments"
#define UNRECOGNIZED_ARGS "ERROR - Unrecognized argument"


#ifdef DEBUG_CMPP
#define PRINT_CMPP_INFO(argc, argv) print_cmpp_info(argc, argv);
static void print_cmpp_info(int argc, char **argv);
#else
#define PRINT_CMPP_INFO(argc, argv)
#endif



/* *********************************************************************** */
/*
main

Function main checks the validity of the command-line arguments
supplied when the program is invoked and calls one of the three
major functions as appropriate:

     preprocess_ifs_file   Process Interface Specification File.
     preprocess_mod_file   Process Model Definition File.
     preprocess_lst_file   Process Pathname List Files.
     output_paths_from_lst_file     Write paths from a list file to stdout
                                    to facilite building the code models

depending on the argument.
*/

int main(
         int argc,      /* Number of command line arguments */
         char *argv[])  /* Command line argument text */
{
   init_error (argv[0]);

    PRINT_CMPP_INFO(argc, argv)

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
    else if (strcmp(argv[1],"-p") == 0) { /* Output paths from a list file */
        if (argc == 3) {
            const int n_item = output_paths_from_lst_file(argv[2]);
            if (n_item < 0) {
                print_error("Unable to print paths to stdout");
                exit(-1);
            }
            exit(n_item);
        }
        else { /* Wrong number of arguments */
            if (argc < 3) {
                print_error(TOO_FEW_ARGS);
            }
            else {
                print_error(TOO_MANY_ARGS);
            }
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
} /* end of function main */



#ifdef DEBUG_CMPP
/* Print some debugging information for cmpp
 *
 * With the use of environment variables to locate files, it is helpful
 * to know some information about the program when debugging. If the
 * build of a code model fails due to cmpp, the macro can be enabled to
 * find the information needed to separately debug cmpp */
static void print_cmpp_info(int argc, char **argv)
{
    /* Print the program and its arguments */
    {
        int i;
        for (i = 0; i < argc; ++i) {
            (void) fprintf(stdout, "%s ", argv[i]);
        }
        (void) fprintf(stdout, "\n");
    }

#ifdef _WIN32
    /* Print the current directory */
    {
        const DWORD n_char = GetCurrentDirectoryA(0, (char *) NULL);
        if (n_char > 0) {
            char *p_buf = (char *) malloc(n_char);
            if (p_buf != (char *) NULL) {
                if (GetCurrentDirectoryA(n_char, p_buf) != 0) {
                    (void) fprintf(stdout, "Current Directory: \"%s\".\n",
                            p_buf);
                }
                free(p_buf);
            }
        }
    }
#endif

    /* Print the CMPP_IDIR environment variable if defined */
    {
        const char * const ev = getenv("CMPP_IDIR");
        if (ev) {
            (void) fprintf(stdout, "CMPP_IDIR = \"%s\"\n", ev);
        }
        else {
            (void) fprintf(stdout, "CMPP_IDIR is not defined\n");
        }
    }

    /* Print the CMPP_ODIR environment variable if defined */
    {
        const char * const ev = getenv("CMPP_ODIR");
        if (ev) {
            (void) fprintf(stdout, "CMPP_ODIR = \"%s\"\n", ev);
        }
        else {
            (void) fprintf(stdout, "CMPP_ODIR is not defined\n");
        }
    }
} /* end of function print_cmpp_info */
#endif /* #ifdef DEBUG_CMPP */



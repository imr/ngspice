/*============================================================================
FILE  util.c

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn and Steve Tynor

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains miscellaneous utility functions used in cmpp.

INTERFACES

    init_error()
    print_error()
    str_to_lower()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include  "cmpp.h"
#include  <stdio.h>
#include  <ctype.h>
#include  <string.h>



/* *********************************************************************** */

char *prog_name;


/* Initialize print_error() with the name of the program */

void init_error (char *program_name)
{
   prog_name = program_name;
}



/* Print an error message to stderr */

void print_error(
    char *msg)       /* The message to write */
{
    fprintf(stderr, "%s: %s\n", prog_name, msg);
}



/* Convert a string to all lower case */

str_to_lower(s)

char *s;   /* The string to convert */
{
    int     i;
    char    c;

    for(i = 0; (c = s[i]) != '\0'; i++)
        if(isalpha(c))
            if(isupper(c))
                s[i] = tolower(c);
}



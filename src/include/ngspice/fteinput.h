/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
    Defs to use the Input routine.

    char_option is used by the lexer and the command interpreter
      response.reply contains a pointer to a character
    button_option is used by the menu system and the help package
      response.reply contains the button number
    click_option is used in the X10 version for the hardcopy command
      response.reply is the associated graph structure
    checkup_option is for handling and pending asynchonous events
*/

#ifndef ngspice_FTEINPUT_H
#define ngspice_FTEINPUT_H


#include <stdio.h>
#include "ngspice/graph.h"

typedef enum {
    error_option,       /* a reply option only */
    button_option,
    char_option,
    click_option,
    checkup_option
} OPTION;

typedef struct request {
    OPTION option;
    FILE *fp;
} REQUEST;

typedef struct response {
    OPTION option;
    union {
      int ch;
      GRAPH *graph;
      int button;
    } reply;
} RESPONSE;

#endif

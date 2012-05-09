/*============================================================================
FILE  mod_yacc.h

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Steve Tynor

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    Typedefs needed by the YYSTYPE union (%union operator) in the yacc
    file. These are only used in the yacc file, but must be defined here since
    the generated token.h file includes a definition of the union YYSTYPE.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "cmpp.h"

typedef struct {
   char *id;
   Boolean_t has_subscript;
   char *subscript;
} Sub_Id_t;

extern void mod_yyerror(char*);

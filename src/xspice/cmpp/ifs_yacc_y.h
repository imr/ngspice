/*============================================================================
FILE  ifs_yacc.h

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
   Boolean_t	has_value;
   Data_Type_t	kind;
   union {
      Boolean_t   bvalue;
      int         ivalue;
      double      rvalue;
      Complex_t   cvalue;
      char        *svalue;
   } u;
} My_Value_t;

typedef struct {
   Boolean_t	has_bound;
   My_Value_t	bound;
} Bound_t;

typedef struct {
   Boolean_t is_named;
   union {
      char *name;
      struct {
         Bound_t upper;
         Bound_t lower;
      } bounds;
   } u;
} Range_t;

typedef struct {
   Port_Type_t	kind;
   char 	*id;	/* undefined unless kind == USER_DEFINED */
} My_Port_Type_t;

typedef struct ctype_list_s {
   My_Port_Type_t      ctype;
   struct ctype_list_s *next;
} Ctype_List_t;


extern void ifs_yyerror(char*);

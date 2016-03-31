%option case-insensitive
%option prefix="ifs_yy"
%option yylineno
%option noyywrap
%option noinput
%option nounput
%{

/*============================================================================
FILE  ifs_lex.l

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Steve Tynor

MODIFICATIONS

    12/31/91  Bill Kuhn  Change "array" to "vector" and "array_bounds"
                         to "vector_bounds".

SUMMARY

    This file defines tokens applicable to parsing the ifspec.ifs
    file, and actions to be taken on encountering those tokens.

INTERFACES

    None.

REFERENCED FILES

    ifs_yacc.y

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <stdlib.h>

#include "ifs_yacc_y.h"
#include "ifs_yacc.h"

int yyival;
double yydval;

/*
 * IFS specs are case insensitive:
 */

/*---------------------------------------------------------------------------*/

%}

%start BOOL CTYPE DIR DTYPE

%x comment stringl

%p 5000

W	[ \t\n]
A	[_a-z]
D	[0-9]
I	[a-z_]
Z	[0-9a-z_]
E	[eE][+-]?{D}+

%%

"/*"         			{ BEGIN(comment); }

<comment>[^*\n]*        /* eat anything that's not a '*' */

<comment>"*"+[^*/\n]*   /* eat up '*'s not followed by '/'s */

<comment>\n             /* new line */

<comment><<EOF>>		{ifs_yyerror ("Unterminated comment"); 
						BEGIN(INITIAL);
						yyterminate(); }

<comment>"*"+"/"        { BEGIN(INITIAL); }

"\""					{ BEGIN(stringl); }

<stringl>[^\"]*			{ return TOK_STRING_LITERAL; }
						
<stringl>"\""			{ BEGIN(INITIAL); }

<stringl><<EOF>>		{ifs_yyerror ("Unterminated string literal");
						BEGIN(INITIAL); 
						yyterminate(); }
	
allowed_types{W}*:		{BEGIN CTYPE; return TOK_ALLOWED_TYPES;}
vector{W}*:			{BEGIN BOOL;  return TOK_ARRAY;}
vector_bounds{W}*:		{return TOK_ARRAY_BOUNDS;}
c_function_name{W}*:		{return TOK_C_FUNCTION_NAME;}
port_name{W}*:			{return TOK_PORT_NAME;}
port_table{W}*:			{return TOK_PORT_TABLE;}
data_type{W}*:			{BEGIN DTYPE; return TOK_DATA_TYPE;}
default_type{W}*:		{BEGIN CTYPE; return TOK_DEFAULT_TYPE;}
default_value{W}*:		{return TOK_DEFAULT_VALUE;}
description{W}*:		{return TOK_DESCRIPTION;}
direction{W}*:			{BEGIN DIR;  return TOK_DIRECTION;}
static_var_name{W}*:		{return TOK_STATIC_VAR_NAME;}
static_var_table{W}*:		{return TOK_STATIC_VAR_TABLE;}
limits{W}*:			{return TOK_LIMITS;}
name_table{W}*:			{return TOK_NAME_TABLE;}
null_allowed{W}*:		{BEGIN BOOL; return TOK_NULL_ALLOWED;}
parameter_name{W}*:		{return TOK_PARAMETER_NAME;}
parameter_table{W}*:		{return TOK_PARAMETER_TABLE;}
spice_model_name{W}*:		{return TOK_SPICE_MODEL_NAME;}

<BOOL>yes			{return TOK_BOOL_YES;}
<BOOL>no			{return TOK_BOOL_NO;}
true				{return TOK_BOOL_YES;}
false				{return TOK_BOOL_NO;}

<CTYPE>v			{return TOK_CTYPE_V;}
<CTYPE>vd			{return TOK_CTYPE_VD;}
<CTYPE>vnam			{return TOK_CTYPE_VNAM;}
<CTYPE>i			{return TOK_CTYPE_I;}
<CTYPE>id			{return TOK_CTYPE_ID;}
<CTYPE>g			{return TOK_CTYPE_G;}
<CTYPE>gd			{return TOK_CTYPE_GD;}
<CTYPE>h			{return TOK_CTYPE_H;}
<CTYPE>hd			{return TOK_CTYPE_HD;}
<CTYPE>d			{return TOK_CTYPE_D;}

<DIR>in				{return TOK_DIR_IN;}
<DIR>out			{return TOK_DIR_OUT;}
<DIR>inout			{return TOK_DIR_INOUT;}

<DTYPE>real			{return TOK_DTYPE_REAL;}
<DTYPE>int			{return TOK_DTYPE_INT;}
<DTYPE>boolean			{return TOK_DTYPE_BOOLEAN;}
<DTYPE>complex			{return TOK_DTYPE_COMPLEX;}
<DTYPE>string			{return TOK_DTYPE_STRING;}
<DTYPE>pointer			{return TOK_DTYPE_POINTER;}

"<"				{return TOK_LANGLE;}
">"				{return TOK_RANGLE;}
"["				{return TOK_LBRACKET;}
"]"				{return TOK_RBRACKET;}
","				{return TOK_COMMA;}
"-"				{return TOK_DASH;}

			 
{I}+{Z}*			{return TOK_IDENTIFIER;}

[+-]?{D}+			{yyival = atoi (yytext);
				 return TOK_INT_LITERAL;}

[+-]?{D}+"."{D}*({E})? |
[+-]?{D}*"."{D}+({E})? |
[+-]?{D}+({E})?	 		{yydval = atof (yytext);
				 return TOK_REAL_LITERAL;}

.				; /* ignore anything else */
\n				; /* ignore anything else */

%%

/*--------------------------------------------------------------------------*/

#if 0
void reset_lex_context (void)
{
   BEGIN 0;
}
#endif

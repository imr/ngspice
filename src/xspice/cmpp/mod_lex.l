%option yylineno
%option noyywrap
%array
%{ /* $Id$ */

/*============================================================================
FILE  mod_lex.l

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

    This file defines tokens applicable to parsing the cfunc.mod
    file, and actions to be taken on encountering those tokens.

INTERFACES

    None.

REFERENCED FILES

    mod_yacc.y

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "mod_yacc.h"
#include "mod_tok.h"

%}

I	[A-Za-z_]
Z	[0-9A-Za-z_]
   
%%

"/*"			{char ch, last_ch;
			 ECHO; /* a comment - repeat it */
			 ch = '\0';
			 do {
			    last_ch = ch;
			    ch = input();
			    fputc(ch,mod_yyout);
			 } while (ch && !((last_ch == '*') && (ch == '/')));
		         if (!ch) {mod_yyerror ("Unterminated comment");}}

ARGS			{return TOK_ARGS;}
INIT			{return TOK_INIT;}
ANALYSIS		{return TOK_ANALYSIS;}
NEW_TIMEPOINT		{return TOK_NEW_TIMEPOINT;}
CALL_TYPE		{return TOK_CALL_TYPE;}
TIME			{return TOK_TIME;}
RAD_FREQ		{return TOK_RAD_FREQ;}
TEMPERATURE		{return TOK_TEMPERATURE;}
T			{return TOK_T;}
LOAD			{return TOK_LOAD;}
TOTAL_LOAD		{return TOK_TOTAL_LOAD;}
MESSAGE			{return TOK_MESSAGE;}
PARAM			{return TOK_PARAM;}
PARAM_SIZE		{return TOK_PARAM_SIZE;}
PARAM_NULL		{return TOK_PARAM_NULL;}
PORT_SIZE		{return TOK_PORT_SIZE;}
PORT_NULL		{return TOK_PORT_NULL;}
PARTIAL			{return TOK_PARTIAL;}
AC_GAIN			{return TOK_AC_GAIN;}
OUTPUT_DELAY		{return TOK_OUTPUT_DELAY;}
STATIC_VAR		{return TOK_STATIC_VAR;}
STATIC_VAR_SIZE		{return TOK_STATIC_VAR_SIZE;}
INPUT			{return TOK_INPUT;}
INPUT_STATE		{return TOK_INPUT_STATE;}
INPUT_TYPE		{return TOK_INPUT_TYPE;}
INPUT_STRENGTH		{return TOK_INPUT_STRENGTH;}
OUTPUT			{return TOK_OUTPUT;}
OUTPUT_STATE		{return TOK_OUTPUT_STATE;}
OUTPUT_STRENGTH		{return TOK_OUTPUT_STRENGTH;}
OUTPUT_TYPE		{return TOK_OUTPUT_TYPE;}
OUTPUT_CHANGED		{return TOK_OUTPUT_CHANGED;}

"("			{return TOK_LPAREN;}
")"			{return TOK_RPAREN;}
"["			{return TOK_LBRACKET;}
"]"			{return TOK_RBRACKET;}
","			{return TOK_COMMA;}

{I}+{Z}*		{return TOK_IDENTIFIER;}
[ \t]			ECHO; /* just eat non-newline whitespace */
\n			ECHO; /* echo newlines */
.			{return TOK_MISC_C;}

%%

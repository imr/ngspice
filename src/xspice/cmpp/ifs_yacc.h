/* A Bison parser, made by GNU Bison 1.875a.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     TOK_ALLOWED_TYPES = 258,
     TOK_ARRAY = 259,
     TOK_ARRAY_BOUNDS = 260,
     TOK_BOOL_NO = 261,
     TOK_BOOL_YES = 262,
     TOK_COMMA = 263,
     TOK_PORT_NAME = 264,
     TOK_PORT_TABLE = 265,
     TOK_CTYPE_D = 266,
     TOK_CTYPE_G = 267,
     TOK_CTYPE_GD = 268,
     TOK_CTYPE_H = 269,
     TOK_CTYPE_HD = 270,
     TOK_CTYPE_I = 271,
     TOK_CTYPE_ID = 272,
     TOK_CTYPE_V = 273,
     TOK_CTYPE_VD = 274,
     TOK_CTYPE_VNAM = 275,
     TOK_C_FUNCTION_NAME = 276,
     TOK_DASH = 277,
     TOK_DATA_TYPE = 278,
     TOK_DEFAULT_TYPE = 279,
     TOK_DEFAULT_VALUE = 280,
     TOK_DESCRIPTION = 281,
     TOK_DIRECTION = 282,
     TOK_DIR_IN = 283,
     TOK_DIR_INOUT = 284,
     TOK_DIR_OUT = 285,
     TOK_DTYPE_BOOLEAN = 286,
     TOK_DTYPE_COMPLEX = 287,
     TOK_DTYPE_INT = 288,
     TOK_DTYPE_POINTER = 289,
     TOK_DTYPE_REAL = 290,
     TOK_DTYPE_STRING = 291,
     TOK_IDENTIFIER = 292,
     TOK_STATIC_VAR_NAME = 293,
     TOK_STATIC_VAR_TABLE = 294,
     TOK_INT_LITERAL = 295,
     TOK_LANGLE = 296,
     TOK_LBRACKET = 297,
     TOK_LIMITS = 298,
     TOK_NAME_TABLE = 299,
     TOK_NULL_ALLOWED = 300,
     TOK_PARAMETER_NAME = 301,
     TOK_PARAMETER_TABLE = 302,
     TOK_RANGLE = 303,
     TOK_RBRACKET = 304,
     TOK_REAL_LITERAL = 305,
     TOK_SPICE_MODEL_NAME = 306,
     TOK_STRING_LITERAL = 307
   };
#endif
#define TOK_ALLOWED_TYPES 258
#define TOK_ARRAY 259
#define TOK_ARRAY_BOUNDS 260
#define TOK_BOOL_NO 261
#define TOK_BOOL_YES 262
#define TOK_COMMA 263
#define TOK_PORT_NAME 264
#define TOK_PORT_TABLE 265
#define TOK_CTYPE_D 266
#define TOK_CTYPE_G 267
#define TOK_CTYPE_GD 268
#define TOK_CTYPE_H 269
#define TOK_CTYPE_HD 270
#define TOK_CTYPE_I 271
#define TOK_CTYPE_ID 272
#define TOK_CTYPE_V 273
#define TOK_CTYPE_VD 274
#define TOK_CTYPE_VNAM 275
#define TOK_C_FUNCTION_NAME 276
#define TOK_DASH 277
#define TOK_DATA_TYPE 278
#define TOK_DEFAULT_TYPE 279
#define TOK_DEFAULT_VALUE 280
#define TOK_DESCRIPTION 281
#define TOK_DIRECTION 282
#define TOK_DIR_IN 283
#define TOK_DIR_INOUT 284
#define TOK_DIR_OUT 285
#define TOK_DTYPE_BOOLEAN 286
#define TOK_DTYPE_COMPLEX 287
#define TOK_DTYPE_INT 288
#define TOK_DTYPE_POINTER 289
#define TOK_DTYPE_REAL 290
#define TOK_DTYPE_STRING 291
#define TOK_IDENTIFIER 292
#define TOK_STATIC_VAR_NAME 293
#define TOK_STATIC_VAR_TABLE 294
#define TOK_INT_LITERAL 295
#define TOK_LANGLE 296
#define TOK_LBRACKET 297
#define TOK_LIMITS 298
#define TOK_NAME_TABLE 299
#define TOK_NULL_ALLOWED 300
#define TOK_PARAMETER_NAME 301
#define TOK_PARAMETER_TABLE 302
#define TOK_RANGLE 303
#define TOK_RBRACKET 304
#define TOK_REAL_LITERAL 305
#define TOK_SPICE_MODEL_NAME 306
#define TOK_STRING_LITERAL 307




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 487 "ifs_yacc.y"
typedef union YYSTYPE {
   Ctype_List_t 	*ctype_list;
   Dir_t		dir;
   Boolean_t		bool;
   Range_t		range;
   Data_Type_t		dtype;
   My_Port_Type_t	ctype;
   My_Value_t		value;
   char			*str;
   Bound_t		bound;
   int			ival;
   double		rval;
   Complex_t		cval;
} YYSTYPE;
/* Line 1240 of yacc.c.  */
#line 156 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;




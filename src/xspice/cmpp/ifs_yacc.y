%{

/*============================================================================
FILE  ifs_yacc.y

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Steve Tynor

MODIFICATIONS

    12/31/91  Bill Kuhn  Fix bug in usage of strcmp in check_default_type()

SUMMARY

    This file contains the BNF specification of the language used in
    the ifspec.ifs file together with various support functions,
    and parses the ifspec.ifs file to get the information from it
    and place this information into a data structure
    of type Ifs_Table_t.

INTERFACES

    yyparse()     -    Generated automatically by UNIX 'yacc' utility.

REFERENCED FILES

    ifs_lex.l

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "ifs_yacc_y.h"

#define yymaxdepth ifs_yymaxdepth
#define yyparse ifs_yyparse
#define yylex   ifs_yylex
#define yyerror ifs_yyerror
#define yylval  ifs_yylval
#define yychar  ifs_yychar
#define yydebug ifs_yydebug
#define yypact  ifs_yypact
#define yyr1    ifs_yyr1
#define yyr2    ifs_yyr2
#define yydef   ifs_yydef
#define yychk   ifs_yychk
#define yypgo   ifs_yypgo
#define yyact   ifs_yyact
#define yyexca  ifs_yyexca
#define yyerrflag ifs_yyerrflag
#define yynerrs ifs_yynerrs
#define yyps    ifs_yyps
#define yypv    ifs_yypv
#define yys ifs_yys
#define yy_yys  ifs_yyyys
#define yystate ifs_yystate
#define yytmp   ifs_yytmp
#define yyv ifs_yyv
#define yy_yyv  ifs_yyyyv
#define yyval   ifs_yyval
#define yylloc  ifs_yylloc
#define yyreds  ifs_yyreds
#define yytoks  ifs_yytoks
#define yylhs   ifs_yyyylhs
#define yylen   ifs_yyyylen
#define yydefred ifs_yyyydefred
#define yydgoto ifs_yyyydgoto
#define yysindex ifs_yyyysindex
#define yyrindex ifs_yyyyrindex
#define yygindex ifs_yyyygindex
#define yytable  ifs_yyyytable
#define yycheck  ifs_yyyycheck
#define yyname   ifs_yyyyname
#define yyrule   ifs_yyyyrule

extern int yylineno;
extern int yyival;
extern double yydval;
extern char *ifs_yytext;
 extern int ifs_yylex(void);

bool parser_just_names;
static bool saw_model_name;
static bool saw_function_name;

static char *dtype_to_str[] = {
   "BOOLEAN", "INTEGER", "REAL", "COMPLEX", "STRING", "POINTER"
   };

static bool did_default_type;
static bool did_allowed_types;

static int num_items;
static int item;
static int item_offset;
static bool num_items_fixed;

Ifs_Table_t *parser_ifs_table;
#define TBL parser_ifs_table

int ifs_num_errors;

static size_t alloced_size [4];

/*
 * !!!!! Make sure these are large enough so that they never get realloced
 * !!!!! since that will cause garbage uninitialized data...
 * !!!!! (FIX THIS!)
 */
#define DEFAULT_SIZE_CONN   100
#define DEFAULT_SIZE_PARAM  100
#define DEFAULT_SIZE_INST_VAR   100
#define GROW_SIZE       10

typedef enum {
   TBL_NAME,
   TBL_PORT,
   TBL_PARAMETER,
   TBL_STATIC_VAR,
} Table_t;

typedef struct {
   Table_t table;
   int record;
} Context_t;

Context_t context;

#define ITEM_BUFFER_SIZE 20 /* number of items that can be put in a table
                 * before requiring a new xxx_TABLE: keyword
                 */
#define FOR_ITEM(i) for (i = item_offset; i < num_items; i++)
#define ITEM_BUF(i) item_buffer[i-item_offset]

#define ASSIGN_BOUNDS(struct_name, i) \
  if (ITEM_BUF(i).range.is_named) {\
    TBL->struct_name[i].has_conn_ref = true;\
    TBL->struct_name[i].conn_ref = find_conn_ref (ITEM_BUF(i).range.u.name);\
  } else {\
    TBL->struct_name[i].has_conn_ref = false;\
    TBL->struct_name[i].has_lower_bound =\
       ITEM_BUF(i).range.u.bounds.lower.has_bound;\
    TBL->struct_name[i].has_upper_bound =\
       ITEM_BUF(i).range.u.bounds.upper.has_bound;\
    if (TBL->struct_name[i].has_lower_bound) {\
      assert (ITEM_BUF(i).range.u.bounds.lower.bound.kind == CMPP_INTEGER);\
       TBL->struct_name[i].lower_bound =\
    ITEM_BUF(i).range.u.bounds.lower.bound.u.ivalue;\
    }\
    if (TBL->struct_name[i].has_upper_bound) {\
       assert (ITEM_BUF(i).range.u.bounds.upper.bound.kind == CMPP_INTEGER);\
       TBL->struct_name[i].upper_bound =\
      ITEM_BUF(i).range.u.bounds.upper.bound.u.ivalue;\
    }\
  }

/*---------------------------------------------------------------------------*/
static void fatal (char *str)
{
   yyerror (str);
   exit(1);
}

/*---------------------------------------------------------------------------*/
static int
find_conn_ref (char *name)
{
   int i;
   char str[130];
   
   for (i = 0; i < TBL->num_conn; i++) {
      if (strcmp (name, TBL->conn[i].name) == 0) {
     return i;
      }
   }
   sprintf (str, "Port `%s' not found", name);
   yyerror (str);
   ifs_num_errors++;

   return 0;
}

typedef enum {C_DOUBLE, C_BOOLEAN, C_POINTER, C_UNDEF} Ctype_Class_t;

/*---------------------------------------------------------------------------*/
static Ctype_Class_t get_ctype_class (Port_Type_t type)
{
   switch (type) {
   case USER_DEFINED:
      return C_POINTER;
      break;
   case DIGITAL:
      return C_BOOLEAN;
      break;
   default:
      return C_DOUBLE;
      break;
   }
}

/*---------------------------------------------------------------------------*/
static void check_port_type_direction (Dir_t dir, Port_Type_t port_type)
{
   switch (port_type) {
   case VOLTAGE:
   case DIFF_VOLTAGE:
   case CURRENT:
   case DIFF_CURRENT:
      if (dir == CMPP_INOUT) {
         yyerror ("Port types `v', `vd', `i', `id' are not valid for `inout' ports");
         ifs_num_errors++;
      }
       break;
   case DIGITAL:
   case USER_DEFINED:
      /*
       * anything goes
       */
      break;
   case VSOURCE_CURRENT:
      if (dir != CMPP_IN) {
         yyerror ("Port type `vnam' is only valid for `in' ports");
         ifs_num_errors++;
      }
      break;
   case CONDUCTANCE:
   case DIFF_CONDUCTANCE:
   case RESISTANCE:
   case DIFF_RESISTANCE:
      if (dir != CMPP_INOUT) {
         yyerror ("Port types `g', `gd', `h', `hd' are only valid for `inout' ports");
         ifs_num_errors++;
      }
      break;
   default:
      assert (0);
   }
}

/*---------------------------------------------------------------------------*/
static void check_dtype_not_pointer (Data_Type_t dtype)
{
   if (dtype == CMPP_POINTER) {
      yyerror("Invalid parameter type - POINTER type valid only for STATIC_VARs");
      ifs_num_errors++;
   }
}

/*---------------------------------------------------------------------------*/
static void check_default_type (Conn_Info_t conn)
{
   int i;
   
   for (i = 0; i < conn.num_allowed_types; i++) {
      if (conn.default_port_type == conn.allowed_port_type[i]) {
         if ((conn.default_port_type != USER_DEFINED) ||
            (strcmp (conn.default_type, conn.allowed_type[i]) == 0)) {
         return;
         }
      }
   }
   yyerror ("Port default type is not an allowed type");
   ifs_num_errors++;
}

/*---------------------------------------------------------------------------*/
static void
assign_ctype_list (Conn_Info_t  *conn, Ctype_List_t *ctype_list )
{
   int i;
   Ctype_List_t *p;
   Ctype_Class_t ctype_class = C_UNDEF;
   
   conn->num_allowed_types = 0;
   for (p = ctype_list; p; p = p->next) {
      conn->num_allowed_types++;
   }
   conn->allowed_type = (char**) calloc ((size_t) conn->num_allowed_types,
                     sizeof (char*));
   conn->allowed_port_type = (Port_Type_t*) calloc ((size_t) conn->num_allowed_types,
                            sizeof (Port_Type_t));
   if (! (conn->allowed_type && conn->allowed_port_type)) {
      fatal ("Could not allocate memory");
   }
   for (i = conn->num_allowed_types-1, p = ctype_list; p; i--, p = p->next) {
      if (ctype_class == C_UNDEF) {
         ctype_class = get_ctype_class (p->ctype.kind);
      }
      if (ctype_class != get_ctype_class (p->ctype.kind)) {
        yyerror ("Incompatible port types in `allowed_types' clause");
        ifs_num_errors++;
      }
      check_port_type_direction (conn->direction, p->ctype.kind);
      
      conn->allowed_port_type[i] = p->ctype.kind;
      conn->allowed_type[i] = p->ctype.id;
   } 
}

/*---------------------------------------------------------------------------*/
static void
assign_value (Data_Type_t type, Value_t *dest_value, My_Value_t src_value)
{
   char str[200];
   if ((type == CMPP_REAL) && (src_value.kind == CMPP_INTEGER)) {
      dest_value->rvalue = src_value.u.ivalue;
      return;
   } else if (type != src_value.kind) {
      sprintf (str, "Invalid parameter type (saw %s - expected %s)",
           dtype_to_str[src_value.kind],
           dtype_to_str[type] );
      yyerror (str);
      ifs_num_errors++;
   } 
   switch (type) {
   case CMPP_BOOLEAN:
      dest_value->bvalue = src_value.u.bvalue;
      break;
   case CMPP_INTEGER:
      dest_value->ivalue = src_value.u.ivalue;
      break;
   case CMPP_REAL:
      dest_value->rvalue = src_value.u.rvalue;
      break;
   case CMPP_COMPLEX:
      dest_value->cvalue = src_value.u.cvalue;
      break;
   case CMPP_STRING:
      dest_value->svalue = src_value.u.svalue;
      break;
   default:
      yyerror ("INTERNAL ERROR - unexpected data type in `assign_value'");
      ifs_num_errors++;
   }
}   

/*---------------------------------------------------------------------------*/
static void
assign_limits (Data_Type_t type, Param_Info_t *param, Range_t range)
{
   if (range.is_named) {
      yyerror ("Named range not allowed for limits");
      ifs_num_errors++;
   }
   param->has_lower_limit = range.u.bounds.lower.has_bound;
   if (param->has_lower_limit) {
      assign_value (type, &param->lower_limit, range.u.bounds.lower.bound);
   }
   param->has_upper_limit = range.u.bounds.upper.has_bound;
   if (param->has_upper_limit) {
      assign_value (type, &param->upper_limit, range.u.bounds.upper.bound);
   }
}

/*---------------------------------------------------------------------------*/
static void
check_item_num (void)
{
   if (item-item_offset >= ITEM_BUFFER_SIZE) {
      fatal ("Too many items in table - split into sub-tables");
   }
   if (item > (int) alloced_size [context.table] ) {
      switch (context.table) {
      case TBL_NAME:
         break;
      case TBL_PORT:
         alloced_size[context.table] += GROW_SIZE;
         TBL->conn = (Conn_Info_t*)
               realloc (TBL->conn,
               alloced_size [context.table] * sizeof (Conn_Info_t));
         if (! TBL->conn) {
            fatal ("Error allocating memory for port definition");
         }
         break;
      case TBL_PARAMETER:
         alloced_size [context.table] += GROW_SIZE;
         TBL->param = (Param_Info_t*)
               realloc (TBL->param,
               alloced_size [context.table] * sizeof (Param_Info_t));
         if (! TBL->param) {
            fatal ("Error allocating memory for parameter definition");
         }
         break;
      case TBL_STATIC_VAR:
         alloced_size [context.table] += GROW_SIZE;
         TBL->inst_var = (Inst_Var_Info_t*)
               realloc (TBL->inst_var,
               alloced_size [context.table] * sizeof (Inst_Var_Info_t));
        if (! TBL->inst_var) {
           fatal ("Error allocating memory for static variable definition");
        }
        break;
      }
   }
   item++;
}

/*---------------------------------------------------------------------------*/
static void
check_end_item_num (void)
{
   if (num_items_fixed) {
      if (item != num_items) {
         char buf[200];
         sprintf
               (buf,
               "Wrong number of elements in sub-table (saw %d - expected %d)",
               item - item_offset,
               num_items - item_offset);
         fatal (buf);
      }
   } else {
      num_items = item;
      num_items_fixed = true;
      switch (context.table) {
      case TBL_NAME:
         break;
      case TBL_PORT:
         TBL->num_conn = num_items;
         break;
      case TBL_PARAMETER:
         TBL->num_param = num_items;
         break;
      case TBL_STATIC_VAR:
         TBL->num_inst_var = num_items;
         break;
      }
   }
   item = item_offset;
}

#define INIT(n) item = (n); item_offset = (n); num_items = (n); num_items_fixed = false
#define ITEM check_item_num()
#define END  check_end_item_num()
   
%}

%token TOK_ALLOWED_TYPES
%token TOK_ARRAY
%token TOK_ARRAY_BOUNDS
%token TOK_BOOL_NO
%token TOK_BOOL_YES
%token TOK_COMMA
%token TOK_PORT_NAME
%token TOK_PORT_TABLE
%token TOK_CTYPE_D
%token TOK_CTYPE_G
%token TOK_CTYPE_GD
%token TOK_CTYPE_H
%token TOK_CTYPE_HD
%token TOK_CTYPE_I
%token TOK_CTYPE_ID
%token TOK_CTYPE_V
%token TOK_CTYPE_VD
%token TOK_CTYPE_VNAM
%token TOK_C_FUNCTION_NAME
%token TOK_DASH
%token TOK_DATA_TYPE
%token TOK_DEFAULT_TYPE
%token TOK_DEFAULT_VALUE
%token TOK_DESCRIPTION
%token TOK_DIRECTION
%token TOK_DIR_IN
%token TOK_DIR_INOUT
%token TOK_DIR_OUT
%token TOK_DTYPE_BOOLEAN
%token TOK_DTYPE_COMPLEX
%token TOK_DTYPE_INT
%token TOK_DTYPE_POINTER
%token TOK_DTYPE_REAL
%token TOK_DTYPE_STRING
%token TOK_IDENTIFIER
%token TOK_STATIC_VAR_NAME
%token TOK_STATIC_VAR_TABLE
%token TOK_INT_LITERAL
%token TOK_LANGLE
%token TOK_LBRACKET
%token TOK_LIMITS
%token TOK_NAME_TABLE
%token TOK_NULL_ALLOWED
%token TOK_PARAMETER_NAME
%token TOK_PARAMETER_TABLE
%token TOK_RANGLE
%token TOK_RBRACKET
%token TOK_REAL_LITERAL
%token TOK_SPICE_MODEL_NAME
%token TOK_STRING_LITERAL

%union {
   Ctype_List_t     *ctype_list;
   Dir_t        dir;
   bool         btype;
   Range_t      range;
   Data_Type_t      dtype;
   My_Port_Type_t   ctype;
   My_Value_t       value;
   char         *str;
   Bound_t      bound;
   int          ival;
   double       rval;
   Complex_t        cval;
}

%type   <ctype_list>    ctype_list delimited_ctype_list
%type   <dir>       direction
%type   <ctype>     ctype
%type   <dtype>     dtype
%type   <range>     range int_range
%type   <value>     value number integer_value value_or_dash
%type   <str>       identifier string
%type   <btype>     btype
%type   <bound>     int_or_dash number_or_dash
%type   <ival>      integer
%type   <rval>      real
%type   <cval>      complex

%start ifs_file

%{
/*
 * resuse the Yacc union for our buffer:
 */
YYSTYPE item_buffer [ITEM_BUFFER_SIZE];

/*
 * Shorthand for refering to the current element of the item buffer:
 */
#define BUF ITEM_BUF(item-1)

%}

%%

ifs_file : {TBL->num_conn = 0;
               TBL->num_param = 0;
               TBL->num_inst_var = 0;

               saw_function_name = false;
               saw_model_name = false;

               alloced_size [TBL_PORT] = DEFAULT_SIZE_CONN;
               alloced_size [TBL_PARAMETER] = DEFAULT_SIZE_PARAM;
               alloced_size [TBL_STATIC_VAR] = DEFAULT_SIZE_INST_VAR;

               TBL->conn = (Conn_Info_t*)
                     calloc(DEFAULT_SIZE_CONN, sizeof (Conn_Info_t));
               TBL->param = (Param_Info_t*)
                    calloc (DEFAULT_SIZE_PARAM, sizeof (Param_Info_t));
               TBL->inst_var = (Inst_Var_Info_t*)
                  calloc (DEFAULT_SIZE_INST_VAR, sizeof (Inst_Var_Info_t));
               if (! (TBL->conn && TBL->param && TBL->inst_var) ) {
                  fatal ("Could not allocate enough memory");
               } 
            }
            list_of_tables
            ;

list_of_tables      : table
            | list_of_tables table
            ;

table           : TOK_NAME_TABLE 
                {context.table = TBL_NAME;} 
              name_table
            | TOK_PORT_TABLE
                {context.table = TBL_PORT;
                 did_default_type = false;
                 did_allowed_types = false;
                 INIT (TBL->num_conn);}
              port_table
                {TBL->num_conn = num_items;}
            | TOK_PARAMETER_TABLE 
                {context.table = TBL_PARAMETER;
                 INIT (TBL->num_param);}
              parameter_table   
                {TBL->num_param = num_items;}
            | TOK_STATIC_VAR_TABLE 
                {context.table = TBL_STATIC_VAR;
                 INIT (TBL->num_inst_var);}
              static_var_table
                {TBL->num_inst_var = num_items;}
            ;

name_table      : /* empty */
            | name_table name_table_item
            ;

name_table_item     : TOK_C_FUNCTION_NAME identifier
              {TBL->name.c_fcn_name =strdup (ifs_yytext);
               saw_function_name = true;
               if (parser_just_names && saw_model_name) return 0;}
            | TOK_SPICE_MODEL_NAME identifier
              {TBL->name.model_name = strdup (ifs_yytext);
               saw_model_name = true;
               if (parser_just_names && saw_function_name) return 0;}
            | TOK_DESCRIPTION string
              {TBL->name.description = strdup (ifs_yytext);}
            ;

port_table  : /* empty */
            | port_table port_table_item
            ;

port_table_item : TOK_PORT_NAME list_of_ids 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->conn[i].name = ITEM_BUF(i).str;
               }}
            | TOK_DESCRIPTION list_of_strings
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->conn[i].description = ITEM_BUF(i).str;
               }}
            | TOK_DIRECTION list_of_directions 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->conn[i].direction = ITEM_BUF(i).dir;
               }}
            | TOK_DEFAULT_TYPE list_of_ctypes 
              {int i;
               END;
               did_default_type = true;
               FOR_ITEM (i) {
                  TBL->conn[i].default_port_type = 
                 ITEM_BUF(i).ctype.kind;
                  TBL->conn[i].default_type = ITEM_BUF(i).ctype.id;
                  if (did_allowed_types) {
                 check_default_type (TBL->conn[i]);
                  }
               }}
            | TOK_ALLOWED_TYPES list_of_ctype_lists 
              {int i;
               END;
               did_allowed_types = true;
               FOR_ITEM (i) {
                  assign_ctype_list (&TBL->conn[i],
                         ITEM_BUF(i).ctype_list);
                  if (did_default_type) {
                 check_default_type (TBL->conn[i]);
                  }
               }}
            | TOK_ARRAY list_of_bool 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->conn[i].is_array = ITEM_BUF(i).btype;
               }}
            | TOK_ARRAY_BOUNDS list_of_array_bounds 
              {int i;
               END;
               FOR_ITEM (i) {
                  ASSIGN_BOUNDS (conn, i);
                  assert (!TBL->conn[i].has_conn_ref);
               }}
            | TOK_NULL_ALLOWED list_of_bool  
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->conn[i].null_allowed = ITEM_BUF(i).btype;
               }}
            ;

parameter_table     : /* empty */
            | parameter_table parameter_table_item
            ;

parameter_table_item    : TOK_PARAMETER_NAME list_of_ids 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->param[i].name = ITEM_BUF(i).str;
               }}
            | TOK_DESCRIPTION list_of_strings 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->param[i].description = ITEM_BUF(i).str;
               }}
            | TOK_DATA_TYPE list_of_dtypes 
              {int i;
               END;
               FOR_ITEM (i) {
                  check_dtype_not_pointer (ITEM_BUF(i).dtype);
                  TBL->param[i].type = ITEM_BUF(i).dtype;
               }}
            | TOK_DEFAULT_VALUE list_of_values 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->param[i].has_default = 
                 ITEM_BUF(i).value.has_value;
                  if (TBL->param[i].has_default) {
                     assign_value (TBL->param[i].type,
                           &TBL->param[i].default_value,
                           ITEM_BUF(i).value);
                  }
               }}
            | TOK_LIMITS list_of_ranges 
              {int i;
               END;
               FOR_ITEM (i) {
                  assign_limits (TBL->param[i].type, 
                         &TBL->param[i],
                         ITEM_BUF(i).range);
               }}
            | TOK_ARRAY list_of_bool 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->param[i].is_array = ITEM_BUF(i).btype;
               }}
            | TOK_ARRAY_BOUNDS list_of_array_bounds 
              {int i;
               END;
               FOR_ITEM (i) {
                  ASSIGN_BOUNDS (param, i);
               }}
            | TOK_NULL_ALLOWED list_of_bool  
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->param[i].null_allowed = ITEM_BUF(i).btype;
               }}
            ;

static_var_table    : /* empty */
            | static_var_table static_var_table_item
            ;

static_var_table_item   : TOK_STATIC_VAR_NAME list_of_ids 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->inst_var[i].name = ITEM_BUF(i).str;
               }}
            | TOK_DESCRIPTION list_of_strings 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->inst_var[i].description = ITEM_BUF(i).str;
               }}
            | TOK_DATA_TYPE list_of_dtypes 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->inst_var[i].type = ITEM_BUF(i).dtype;
               }}
            | TOK_ARRAY list_of_bool 
              {int i;
               END;
               FOR_ITEM (i) {
                  TBL->inst_var[i].is_array = ITEM_BUF(i).btype;
               }}
            ;

list_of_ids     : /* empty */
            | list_of_ids identifier {ITEM; BUF.str = $2;}
            ;

list_of_array_bounds    : /* empty */
            | list_of_array_bounds int_range
               {ITEM; 
                BUF.range = $2;}
            | list_of_array_bounds identifier 
               {ITEM; 
                BUF.range.is_named = true;
                BUF.range.u.name = $2;}
            ;

list_of_strings     : /* empty */
            | list_of_strings string {ITEM; BUF.str = $2;}
            ;

list_of_directions  : /* empty */
            | list_of_directions direction {ITEM; BUF.dir = $2;}
            ;

direction       : TOK_DIR_IN    {$$ = CMPP_IN;}
            | TOK_DIR_OUT   {$$ = CMPP_OUT;}
            | TOK_DIR_INOUT {$$ = CMPP_INOUT;}
            ;

list_of_bool        : /* empty */
            | list_of_bool btype {ITEM; BUF.btype = $2;}
            ;

list_of_ctypes      : /* empty */
            | list_of_ctypes ctype {ITEM; BUF.ctype = $2;}
            ;

ctype           : TOK_CTYPE_V   {$$.kind = VOLTAGE;}
            | TOK_CTYPE_VD  {$$.kind = DIFF_VOLTAGE;}
            | TOK_CTYPE_VNAM {$$.kind = VSOURCE_CURRENT;}
            | TOK_CTYPE_I   {$$.kind = CURRENT;}
            | TOK_CTYPE_ID  {$$.kind = DIFF_CURRENT;}
            | TOK_CTYPE_G   {$$.kind = CONDUCTANCE;}
            | TOK_CTYPE_GD  {$$.kind = DIFF_CONDUCTANCE;}
            | TOK_CTYPE_H   {$$.kind = RESISTANCE;}
            | TOK_CTYPE_HD  {$$.kind = DIFF_RESISTANCE;}
            | TOK_CTYPE_D   {$$.kind = DIGITAL;}
            | identifier    {$$.kind = USER_DEFINED;
                     $$.id   = $1;}
            ;

list_of_dtypes      : /* empty */
            | list_of_dtypes dtype {ITEM; BUF.dtype = $2;}
            ;

dtype           : TOK_DTYPE_REAL    {$$ = CMPP_REAL;}
            | TOK_DTYPE_INT     {$$ = CMPP_INTEGER;}
            | TOK_DTYPE_BOOLEAN {$$ = CMPP_BOOLEAN;}
            | TOK_DTYPE_COMPLEX {$$ = CMPP_COMPLEX;}
            | TOK_DTYPE_STRING  {$$ = CMPP_STRING;}
            | TOK_DTYPE_POINTER {$$ = CMPP_POINTER;}
            ;

list_of_ranges      : /* empty */
            | list_of_ranges range {ITEM; BUF.range = $2;}
            ;

int_range       : TOK_DASH {$$.is_named = false; 
                    $$.u.bounds.lower.has_bound = false;
                    $$.u.bounds.upper.has_bound = false;}
            | TOK_LBRACKET int_or_dash maybe_comma int_or_dash
              TOK_RBRACKET
               {$$.is_named = false;
                $$.u.bounds.lower = $2;
                $$.u.bounds.upper = $4;}
            ;

maybe_comma     : /* empty */
            | TOK_COMMA
            ;

int_or_dash     : TOK_DASH {$$.has_bound = false;}
            | integer_value   {$$.has_bound = true; 
                           $$.bound = $1;}
            ;

range           : TOK_DASH {$$.is_named = false; 
                    $$.u.bounds.lower.has_bound = false;
                    $$.u.bounds.upper.has_bound = false;}
            | TOK_LBRACKET number_or_dash maybe_comma 
              number_or_dash TOK_RBRACKET 
               {$$.is_named = false;
                $$.u.bounds.lower = $2;
                $$.u.bounds.upper = $4;}
            ;

number_or_dash      : TOK_DASH {$$.has_bound = false;}
            | number {$$.has_bound = true; 
                  $$.bound = $1;}
            ;

list_of_values      : /* empty */
            | list_of_values value_or_dash {ITEM; BUF.value = $2;}
            ;

value_or_dash       : TOK_DASH {$$.has_value = false;}
            | value
            ;

value           : string    {$$.has_value = true;
                     $$.kind = CMPP_STRING;
                     $$.u.svalue = $1;}
            | btype     {$$.has_value = true;
                     $$.kind = CMPP_BOOLEAN;
                     $$.u.bvalue = $1;}
            | complex   {$$.has_value = true;
                     $$.kind = CMPP_COMPLEX;
                     $$.u.cvalue = $1;}
            | number
            ;

complex         : TOK_LANGLE real maybe_comma real TOK_RANGLE
              {$$.real = $2;
               $$.imag = $4;}
            ;

list_of_ctype_lists : /* empty */
            | list_of_ctype_lists delimited_ctype_list 
                {ITEM; BUF.ctype_list = $2;}
            ;

delimited_ctype_list    : TOK_LBRACKET ctype_list TOK_RBRACKET {$$ = $2;}
            ;

ctype_list      : ctype 
               {$$ = (Ctype_List_t*)calloc (1,
                            sizeof (Ctype_List_t));
                if (!$$) {
                   fatal ("Error allocating memory");
                }
                $$->ctype = $1;
                $$->next = (Ctype_List_t*)0;}
            | ctype_list maybe_comma ctype
               {$$ = (Ctype_List_t*)calloc (1, 
                            sizeof (Ctype_List_t));
                if (!$$) {
                   fatal ("Error allocating memory");
                }
                $$->ctype = $3;
                $$->next = $1;
                /*$$->next = (Ctype_List_t*)0;
                assert ($1);
                $1->next = $$;*/}
            ;

btype           : TOK_BOOL_YES  {$$ = true;}
            | TOK_BOOL_NO   {$$ = false;}
            ;

string          : TOK_STRING_LITERAL {$$ = strdup(ifs_yytext);}
            ;

identifier      : TOK_IDENTIFIER {$$ = strdup(ifs_yytext);}
            ;

number          : real  {$$.has_value = true;
                 $$.kind = CMPP_REAL;
                 $$.u.rvalue = $1;}
            | integer_value
            ;

integer_value       : integer   {$$.has_value = true;
                     $$.kind = CMPP_INTEGER;
                     $$.u.ivalue = $1;}
            ;

real            : TOK_REAL_LITERAL {$$ = yydval;}
            ;

integer         : TOK_INT_LITERAL  {$$ = yyival;}
            ;

%%

/*************
 * Header file for inpcom.c
 * 1999 E. Rouat
 ************/

#ifndef INPCOM_H_INCLUDED
#define INPCOM_H_INCLUDED

FILE * inp_pathopen(char *name, char *mode);
void inp_readall(FILE *fp, struct line **data, int, char *dirname);
void inp_casefix(register char *string);

/* globals -- wanted to avoid complicating inp_readall interface */
static char *library_file[1000];
static char *library_name[1000][1000];
struct line *library_ll_ptr[1000][1000];
struct line *libraries[1000];
int         num_libraries;
int         num_lib_names[1000];
static      char *global;
static char *subckt_w_params[1000];
static int  num_subckt_w_params;
static char *func_names[1000];
static char *func_params[1000][1000];
static char *func_macro[5000];
static int  num_functions;
static int  num_parameters[1000];
#endif

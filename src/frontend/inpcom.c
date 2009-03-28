/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * For dealing with spice input decks and command scripts
 */

/* h_vogt 20 April 2008
 * For xspice and num_pram compatibility .cmodel added
 * .cmodel will be replaced by .model in inp_fix_param_values()
 * and then the entire line is skipped (will not be changed by this function).
 * Usage of numparam requires {} around the parameters in the .cmodel line.
 */

/*
 * SJB 21 April 2005
 * Added support for end-of-line comments that begin with any of the following:
 *   ';'  (for PSpice compatability)
 *   '$ ' (for HSpice compatability)
 *   '//' (like in c++ and as per the numparam code)
 *   '--' (as per the numparam code)
 * Any following text to the end of the line is ignored.
 * Note requirement for $ to be followed by a space. This is to avoid conflict
 * with use in front of a variable.
 * Comments on a contunuation line (i.e. line begining with '+') are allowed
 * and are removed before lines are stitched.
 * Lines that contain only an end-of-line comment with or withou leading white
 * space are also allowed.
 */

/*
 * SJB 22 May 2001
 * Fixed memory leaks in inp_readall() when first(?) line of input begins with a '@'.
 * Fixed memory leaks in inp_readall() when .include lines have errors
 * Fixed crash where a NULL pointer gets freed in inp_readall()
 */

#include "ngspice.h"

#ifdef HAVE_LIBGEN_H /* dirname */
#include <libgen.h>
#define HAVE_DECL_BASENAME 1
#endif

#ifdef HAVE_LIBIBERTY_H /* asprintf etc. */
#include <libiberty.h>
#undef AND /* obsolete macro in ansidecl.h */
#endif

#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "fteinp.h"

#include "inpcom.h"
#include "variable.h"
#include "../misc/util.h" /* dirname() */
#include "../misc/stringutil.h"
#include <wordlist.h>

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include new stuff */
#include "ipctiein.h"
#include "enh.h"
/* gtri - end - 12/12/90 */
#endif

/* SJB - Uncomment this line for debug tracing */
/*#define TRACE*/

/* uncomment this line for getting deck output after all
   manipulations into debug-out.txt */
/* #define OUTDECK */

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

/* Collect information for dynamic allocation of numparam arrays */
/* number of lines in input deck */
int dynmaxline;  /* inpcom.c 1529 */
/* max. line length in input deck */
int dynLlen; /* inpcom.c 1526 */
 /* number of lines in deck after expansion */
int dynMaxckt = 0; /* subckt.c 307 */
/* number of parameter substitutions */
long dynsubst; /* spicenum.c 221 */

/* static declarations */
static char * readline(FILE *fd);
static int  get_number_terminals( char *c );
static void inp_stripcomments_deck(struct line *deck);
static void inp_stripcomments_line(char * s);
static void inp_fix_for_numparam(struct line *deck);
static void inp_remove_excess_ws(struct line *deck);
static void inp_determine_libraries(struct line *deck, char *lib_name);
static void inp_init_lib_data();
static void inp_grab_func(struct line *deck);
static void inp_fix_inst_calls_for_numparam( struct line *deck);
static void inp_expand_macros_in_func();
static void inp_expand_macros_in_deck( struct line *deck );
static void inp_fix_param_values( struct line *deck );
static void inp_reorder_params( struct line *deck, struct line *list_head, struct line *end );
static int  inp_split_multi_param_lines( struct line *deck, int line_number );
static void inp_sort_params( struct line *start_card, struct line *end_card, struct line *card_bf_start, struct line *s_c, struct line *e_c );

/*-------------------------------------------------------------------------*
 *  This routine reads a line (of arbitrary length), up to a '\n' or 'EOF' *
 *  and returns a pointer to the resulting null terminated string.         *
 *  The '\n' if found, is included in the returned string.                 *
 *  From: jason@ucbopal.BERKELEY.EDU (Jason Venner)                        *
 *  Newsgroups: net.sources                                                *
 *-------------------------------------------------------------------------*/
#define STRGROW 256

static char *
readline(FILE *fd)
{
   int c;
   int memlen;
   char *strptr;
   int strlen;
    
   strptr = NULL;
   strlen = 0;
   memlen = STRGROW; 
   strptr = tmalloc(memlen);
   memlen -= 1;          /* Save constant -1's in while loop */
   while((c = getc(fd)) != EOF) {
      if (strlen == 0 && (c == '\t' || c == ' ')) /* Leading spaces away */
         continue;
      strptr[strlen] = c;
      strlen++;
      if( strlen >= memlen ) {
         memlen += STRGROW;
         if( !(strptr = trealloc(strptr, memlen + 1))) {
            return (NULL);
         }
      }
      if (c == '\n') {
         break;
      }
   }
   if (!strlen) {
      tfree(strptr);
      return (NULL);
   }
//   strptr[strlen] = '\0'; 
   /* Trim the string */
   strptr = trealloc(strptr, strlen + 1);
   strptr[strlen] = '\0'; 
   return (strptr);
}


/*-------------------------------------------------------------------------*
 * Look up the variable sourcepath and try everything in the list in order *
 * if the file isn't in . and it isn't an abs path name.                   *
 *-------------------------------------------------------------------------*/
FILE *
inp_pathopen(char *name, char *mode)
{
    FILE *fp;
    char buf[BSIZE_SP];
    struct variable *v;

    /* If this is an abs pathname, or there is no sourcepath var, just
     * do an fopen.
     */
    if (index(name, DIR_TERM)
	    || !cp_getvar("sourcepath", VT_LIST, (char *) &v))
        return (fopen(name, mode));

    while (v) {
        switch (v->va_type) {
            case VT_STRING:
		cp_wstrip(v->va_string);
		(void) sprintf(buf, "%s%s%s", v->va_string, DIR_PATHSEP, name);
		break;
            case VT_NUM:
		(void) sprintf(buf, "%d%s%s", v->va_num, DIR_PATHSEP, name);
		break;
            case VT_REAL:   /* This is foolish */
		(void) sprintf(buf, "%g%s%s", v->va_real, DIR_PATHSEP, name);
		break;
        }
        if ((fp = fopen(buf, mode)))
            return (fp);
        v = v->va_next;
    }
    return (NULL);
}

static void
inp_fix_gnd_name( struct line *deck ) {
  struct line *c = deck;
  char *gnd;

  while ( c != NULL ) {
    gnd = c->li_line;
    if ( *gnd == '*' ) { c = c->li_next; continue; }
    while ( (gnd = strstr( gnd, "gnd " ) ) ) {
      if ( isspace(*(gnd-1)) ) {
	memcpy( gnd, "0   ", 4 );
      }
      gnd += 4;
    }
    c = c->li_next;
  }
}

static struct line*
create_new_card( char *card_str, int *line_number ) {
  char        *str     = strdup(card_str);
  struct line *newcard = alloc(struct line);
	
  newcard->li_line    = str;
  newcard->li_linenum = *line_number;
  newcard->li_error   = NULL;
  newcard->li_actual  = NULL;

  *line_number = *line_number + 1;

  return newcard;
}

static void
inp_chk_for_multi_in_vcvs( struct line *deck, int *line_number ) {
  struct line *c, *a_card, *model_card, *next_card;
  char *line, *bool_ptr, *str_ptr1, *str_ptr2, keep, *comma_ptr, *xy_values1[5], *xy_values2[5];
  char *node_str, *ctrl_node_str, *xy_str1, *model_name, *fcn_name;
  char big_buf[1000];
  int  xy_count1 = 0, xy_count2 = 0;

  for ( c = deck; c != NULL; c = c->li_next ) {
    str_ptr1 = line = c->li_line;
    if ( *line == 'e' ) {
      if ( (bool_ptr = strstr( line, "nand(" )) ||
	   (bool_ptr = strstr( line, "and("  )) ||
	   (bool_ptr = strstr( line, "nor("  )) ||
	   (bool_ptr = strstr( line, "or("   )) ) {
	while ( !isspace(*str_ptr1) ) str_ptr1++;
	keep       = *str_ptr1;	*str_ptr1  = '\0';
	model_name = strdup(line); *str_ptr1  = keep;

	str_ptr2 = bool_ptr - 1;
	while ( isspace(*str_ptr1) ) str_ptr1++;
	while ( isspace(*str_ptr2) ) str_ptr2--;
	str_ptr2++;
	keep      = *str_ptr2; *str_ptr2 = '\0';
	node_str  = strdup(str_ptr1); *str_ptr2 = keep;

	str_ptr1 = bool_ptr + 1;
	while ( *str_ptr1 != '(' ) str_ptr1++;
	*str_ptr1 = '\0'; fcn_name = strdup(bool_ptr);
	*str_ptr1 = '(';

	str_ptr1  = strstr( str_ptr1, ")" );
	str_ptr1++;
	comma_ptr = str_ptr2 = strstr( line, "," );
	str_ptr2--;
	while( isspace(*str_ptr2) ) str_ptr2--;

	while ( isspace(*str_ptr1) ) str_ptr1++;
	if ( *str_ptr2 == '}' ) {
	  while ( *str_ptr2 != '{' ) str_ptr2--;
	  xy_str1 = str_ptr2; str_ptr2--;
	  while ( isspace(*str_ptr2) ) str_ptr2--;
	  str_ptr2++;
	} else {
	  while ( !isspace(*str_ptr2) ) str_ptr2--;
	  xy_str1 = str_ptr2 + 1;
	  while ( isspace(*str_ptr2) ) str_ptr2--;
	  str_ptr2++;
	}
	keep = *str_ptr2; *str_ptr2 = '\0';
	ctrl_node_str = strdup(str_ptr1); *str_ptr2 = keep;

	str_ptr1 = comma_ptr + 1;
	while ( isspace(*str_ptr1) ) str_ptr1++;
	if ( *str_ptr1 == '{' ) {
	  while ( *str_ptr1 != '}' ) str_ptr1++;
	  str_ptr1++;
	} else {
	  while ( !isspace(*str_ptr1) ) str_ptr1++;
	}
	keep = *str_ptr1; *str_ptr1 = '\0';
	xy_count1 = get_comma_separated_values( xy_values1, xy_str1 );
	*str_ptr1 = keep;

	while ( isspace(*str_ptr1) ) str_ptr1++;
	xy_count2 = get_comma_separated_values( xy_values2, str_ptr1 );

	// place restrictions on only having 2 point values; this can change later
	if ( xy_count1 != 2 && xy_count2 != 2 ) {
	  fprintf(stderr,"ERROR: only expecting 2 pair values for multi-input vcvs!\n");
	}

	sprintf( big_buf, "%s %%vd[ %s ] %%vd( %s ) %s", model_name, ctrl_node_str, node_str, model_name );
	a_card = create_new_card( big_buf, line_number );
	*a_card->li_line = 'a';

	sprintf( big_buf, ".model %s multi_input_pwl ( x = [%s %s] y = [%s %s] model = \"%s\" )", model_name, xy_values1[0], xy_values2[0],
		 xy_values1[1], xy_values2[1], fcn_name );
	model_card = create_new_card( big_buf, line_number );

	tfree(model_name); tfree(node_str); tfree(fcn_name); tfree(ctrl_node_str);
	tfree(xy_values1[0]); tfree(xy_values1[1]); tfree(xy_values2[0]); tfree(xy_values2[1]);

	*c->li_line = '*';
	next_card   = c->li_next;
	c->li_next  = a_card;
	a_card->li_next     = model_card;
	model_card->li_next = next_card;
      }
    }
  }
}

static void
inp_add_control_section( struct line *deck, int *line_number ) {
  struct line *c, *newcard, *prev_card = NULL;
  bool        found_control = FALSE, found_run = FALSE;
  bool        found_end = FALSE;
  char        *op_line  = NULL, rawfile[1000], *line;

  for ( c = deck; c != NULL; c = c->li_next ) {
    if ( *c->li_line == '*' ) continue;
    if ( ciprefix( ".op ",  c->li_line ) ) {
      *c->li_line = '*';
      op_line = c->li_line + 1;
    }
    if ( ciprefix( ".end", c->li_line ) ) found_end = TRUE;
    if ( found_control && ciprefix( "run",  c->li_line ) ) found_run = TRUE;

    if ( ciprefix( ".control", c->li_line ) ) found_control = TRUE;
    if ( ciprefix( ".endc",    c->li_line ) ) {
      found_control = FALSE;

      if ( !found_run ) {
	newcard            = create_new_card( "run", line_number );
	prev_card->li_next = newcard;
	newcard->li_next   = c;
	prev_card          = newcard;
	found_run          = TRUE;
      }
      if ( cp_getvar( "rawfile", VT_STRING, rawfile ) ) {
	line = tmalloc( strlen("write") + strlen(rawfile) + 2 );
	sprintf(line, "write %s", rawfile);
	newcard            = create_new_card( line, line_number );
	prev_card->li_next = newcard;
	newcard->li_next   = c;
	prev_card          = newcard;
	tfree(line);
      }
    }
    prev_card = c;
  }
  // check if need to add control section
  if ( !found_run && found_end ) {
    prev_card        = deck->li_next;
    newcard          = create_new_card( ".endc",    line_number );
    deck->li_next    = newcard;
    newcard->li_next = prev_card;

    if ( cp_getvar( "rawfile", VT_STRING, rawfile ) ) {
      line = tmalloc( strlen("write") + strlen(rawfile) + 2 );
      sprintf(line, "write %s", rawfile);
      prev_card        = deck->li_next;
      newcard          = create_new_card( line, line_number );
      deck->li_next    = newcard;
      newcard->li_next = prev_card;
      tfree(line);
    }
    if ( op_line != NULL ) {
      prev_card        = deck->li_next;
      newcard          = create_new_card( op_line,      line_number );
      deck->li_next    = newcard;
      newcard->li_next = prev_card;
    }

    prev_card        = deck->li_next;
    newcard          = create_new_card( "run",      line_number );
    deck->li_next    = newcard;
    newcard->li_next = prev_card;

    prev_card        = deck->li_next;
    newcard          = create_new_card( ".control", line_number );
    deck->li_next    = newcard;
    newcard->li_next = prev_card;
  }
}

// look for shell-style end-of-line continuation '\\'
static bool
chk_for_line_continuation( char *line ) {
  char *ptr = line + strlen(line) - 1;

  if ( *line != '*' && *line != '$' ) {
    while ( ptr >= line && *ptr && isspace(*ptr) ) ptr--;

    if ( (ptr-1) >= line && *ptr == '\\' && *(ptr-1) && *(ptr-1) == '\\' ) {
      *ptr = ' '; *(ptr-1) = ' ';
      return TRUE;
    }
  }
  
  return FALSE;
}

//
// change .macro --> .subckt
//        .eom   --> .ends
//        .subckt (1 2 3) --> .subckt 1 2 3
//        v(1,11)         --> 0
//        x1 (1 2 3)      --> x1 1 2 3
//        .param func1(x,y) = {x*y} --> .func func1(x,y) {x*y}
//
static void
inp_fix_macro_param_func_paren_io( struct line *begin_card ) {
  struct line *card;
  char        *str_ptr, *new_str, *open_paren_ptr, *search_ptr, *fcn_name;
  bool        is_func = FALSE;

  for ( card = begin_card; card != NULL; card = card->li_next ) {

    if ( *card->li_line == '*' ) continue;

    // zero out any voltage node references on .param lines
    if ( ciprefix( ".param", card->li_line ) ) {
      search_ptr = card->li_line;
      while( ( open_paren_ptr = strstr( search_ptr, "(" ) ) ) {
	fcn_name = open_paren_ptr - 1;
	while ( *fcn_name != '\0' && fcn_name != search_ptr && (isalnum(*fcn_name) || *fcn_name == '_' ) ) fcn_name--;
	if ( fcn_name != search_ptr ) fcn_name++;
	*open_paren_ptr = '\0';
	if ( strcmp( fcn_name, "v" ) == 0 ) {
	  *open_paren_ptr = ' ';
	  *fcn_name       = '0';
	  fcn_name++;
	  while ( *fcn_name != ')' ) { *fcn_name = ' '; fcn_name++; }
	  *fcn_name = ' ';
	}
	else {
	  *open_paren_ptr = '(';
	}
	search_ptr = open_paren_ptr + 1;
      }
    }

    if ( ciprefix( ".macro", card->li_line ) || ciprefix( ".eom", card->li_line ) ) {
      str_ptr = card->li_line;
      while( !isspace(*str_ptr) ) str_ptr++;

      if ( ciprefix( ".macro", card->li_line ) ) {
	new_str = tmalloc( strlen(".subckt") + strlen(str_ptr) + 1 );
	sprintf( new_str, ".subckt%s", str_ptr );
      } else {
	new_str = tmalloc( strlen(".ends") + strlen(str_ptr) + 1 );
	sprintf( new_str, ".ends%s", str_ptr );
      }

      tfree( card->li_line );
      card->li_line = new_str;
    }
    if ( ciprefix( ".subckt", card->li_line ) || ciprefix( "x", card->li_line ) ) {
      str_ptr = card->li_line;
      while( !isspace(*str_ptr) ) str_ptr++;  // skip over .subckt, instance name
      while( isspace(*str_ptr)  ) str_ptr++;
      if ( ciprefix( ".subckt", card->li_line ) ) {
	while( !isspace(*str_ptr) ) str_ptr++;  // skip over subckt name
	while( isspace(*str_ptr)  ) str_ptr++;
      }
      if ( *str_ptr == '(' ) {
	*str_ptr = ' ';
	while ( *str_ptr && *str_ptr != '\0' ) {
	  if ( *str_ptr == ')' ) { *str_ptr = ' '; break; }
	  str_ptr++;
	}
      }
    }
    is_func = FALSE;
    if ( ciprefix( ".param", card->li_line ) ) {
      str_ptr = card->li_line;
      while ( !isspace( *str_ptr ) ) str_ptr++;  // skip over .param
      while ( isspace( *str_ptr )  ) str_ptr++;
      while ( !isspace( *str_ptr ) && *str_ptr != '=' ) {
	if ( *str_ptr == '(' ) is_func = TRUE;
	str_ptr++;
      }

      if ( is_func ) {
	if ( ( str_ptr = strstr( card->li_line, "=" ) ) ) *str_ptr = ' ';

	str_ptr = card->li_line + 1;
	*str_ptr = 'f'; *(str_ptr+1) = 'u'; *(str_ptr+2) = 'n';
	*(str_ptr+3) = 'c'; *(str_ptr+4) = ' ';
      }
    }
  }
}

static char *
get_instance_subckt( char *line )
{
  char *equal_ptr = NULL, *end_ptr = line + strlen(line) - 1, *inst_name_ptr = NULL, *inst_name = NULL;
  char keep = ' ';

  // see if instance has parameters
  if ( ( equal_ptr = strstr( line, "=" ) ) ) {
    end_ptr = equal_ptr - 1;
    while ( isspace(*end_ptr)  ) end_ptr--;
    while ( !isspace(*end_ptr) ) end_ptr--;
    while ( isspace(*end_ptr)  ) end_ptr--;
    end_ptr++;
    keep     = *end_ptr;
    *end_ptr = '\0';
  }
  inst_name_ptr = end_ptr;
  while ( !isspace(*inst_name_ptr) ) inst_name_ptr--;
  inst_name_ptr++;

  inst_name = strdup(inst_name_ptr);

  if ( equal_ptr ) *end_ptr = keep;

  return inst_name;
}

static char*
get_subckt_model_name( char *line )
{
  char *name = line, *end_ptr = NULL, *subckt_name;
  char keep;

  while ( !isspace( *name ) ) name++;   // eat .subckt|.model
  while ( isspace( *name )  ) name++;

  end_ptr = name;
  while ( !isspace( *end_ptr ) ) end_ptr++;
  keep     = *end_ptr;
  *end_ptr = '\0';

  subckt_name = strdup(name);
  *end_ptr    = keep;

  return subckt_name; 
}

static char*
get_model_name( char *line, int num_terminals )
{
  char *beg_ptr = line, *end_ptr, keep, *model_name = NULL;
  int  i = 0;

  while ( !isspace( *beg_ptr ) && *beg_ptr != '\0' ) beg_ptr++;   // eat device name
  while ( isspace( *beg_ptr )  && *beg_ptr != '\0' ) beg_ptr++;

  for ( i = 0; i < num_terminals; i++ ) {
    while ( !isspace( *beg_ptr ) && *beg_ptr != '\0' ) beg_ptr++;
    while ( isspace( *beg_ptr )  && *beg_ptr != '\0' ) beg_ptr++;
  }
  end_ptr = beg_ptr;
  while ( *end_ptr != '\0' && !isspace( *end_ptr ) ) end_ptr++;
  keep = *end_ptr;
  *end_ptr = '\0';

  model_name = strdup( beg_ptr );

  *end_ptr = keep;

  return model_name;
}

static char *
get_adevice_model_name( char *line )
{
  char *model_name, *ptr_end = line + strlen(line), *ptr_beg, keep;

  while ( isspace( *(ptr_end-1) ) ) ptr_end--;
  ptr_beg = ptr_end - 1;

  while ( !isspace(*ptr_beg) ) ptr_beg--;
  ptr_beg++;
  keep = *ptr_end; *ptr_end = '\0';
  model_name = strdup(ptr_beg);
  *ptr_end = keep;

  return model_name;
}

static void
get_subckts_for_subckt( struct line *start_card, char *subckt_name,
			char *used_subckt_names[], int *num_used_subckt_names,
			char *used_model_names[], int *num_used_model_names,
			bool has_models )
{
  struct line *card;
  char *line = NULL, *curr_subckt_name, *inst_subckt_name, *model_name, *new_names[100];
  bool found_subckt = FALSE, have_subckt = FALSE, found_model = FALSE;
  int  i, num_terminals = 0, tmp_cnt = 0;

  for ( card = start_card; card != NULL; card = card->li_next ) {
    line = card->li_line;

    if ( *line == '*' ) continue;

    if ( ( ciprefix( ".ends",   line ) || ciprefix( ".eom",   line ) ) && found_subckt )
      break;

    if ( ciprefix( ".subckt", line ) || ciprefix( ".macro", line ) ) {
      curr_subckt_name = get_subckt_model_name( line );

      if ( strcmp( curr_subckt_name, subckt_name ) == 0 ) {
	found_subckt = TRUE;
      }

      tfree(curr_subckt_name);
    }
    if ( found_subckt ) {
      if ( *line == 'x' ) {
	inst_subckt_name = get_instance_subckt( line );
	have_subckt = FALSE;
	for ( i = 0; i < *num_used_subckt_names; i++ )
	  if ( strcmp( used_subckt_names[i], inst_subckt_name ) == 0 )
	    have_subckt = TRUE;
	if ( !have_subckt ) {
	  new_names[tmp_cnt++] = used_subckt_names[*num_used_subckt_names] = inst_subckt_name;
	  *num_used_subckt_names = *num_used_subckt_names + 1;
	}
	else  tfree( inst_subckt_name );
      }
      else if ( *line == 'a' ) {
	model_name = get_adevice_model_name( line );
	found_model = FALSE;
	for ( i = 0; i < *num_used_model_names; i++ )
	  if ( strcmp( used_model_names[i], model_name ) == 0 ) found_model = TRUE;
	if ( !found_model ) {
	  used_model_names[*num_used_model_names] = model_name;
	  *num_used_model_names = *num_used_model_names + 1;
	}
	else  tfree( model_name );
      }
      else if ( has_models ) {
	num_terminals = get_number_terminals( line );

	if ( num_terminals != 0 ) {
	  model_name = get_model_name( line, num_terminals );
	  
	  if ( isalpha( *model_name ) ) {
	    found_model = FALSE;
	    for ( i = 0; i < *num_used_model_names; i++ )
	      if ( strcmp( used_model_names[i], model_name ) == 0 ) found_model = TRUE;
	    if ( !found_model ) {
	      used_model_names[*num_used_model_names] = model_name;
	      *num_used_model_names = *num_used_model_names + 1;
	    }
	    else  tfree( model_name );
	  }
	  else {
	    tfree( model_name );
	  }
	}
      }
    }
  }
  // now make recursive call on instances just found above
  for ( i = 0; i < tmp_cnt; i++ )
    get_subckts_for_subckt( start_card, new_names[i], used_subckt_names, num_used_subckt_names,
			    used_model_names, num_used_model_names, has_models );
}

/*
  check if current token matches model bin name -- <token>.[0-9]+
 */
static bool
model_bin_match( char* token, char* model_name )
{
  char* dot_char;
  bool  flag = FALSE;

  if ( strncmp( model_name, token, strlen(token) ) == 0 ) {
    if ( (dot_char = strstr( model_name, "." )) ) {
      flag = TRUE;
      dot_char++;
      while( *dot_char != '\0' ) {
	if ( !isdigit( *dot_char ) ) {
	  flag = FALSE;
	  break;
	}
	dot_char++;
      }
    }
  }
  return flag;
}

/*
  iterate through the deck and comment out unused subckts, models
  (don't want to waste time processing everything)
  also comment out .param lines with no parameters defined
 */
static void
comment_out_unused_subckt_models( struct line *start_card )
{
  struct line *card;
  char *used_subckt_names[1000], *used_model_names[1000], *line = NULL, *subckt_name, *model_name;
  int  num_used_subckt_names = 0, num_used_model_names = 0, i = 0, num_terminals = 0, tmp_cnt = 0;
  bool processing_subckt = FALSE, found_subckt = FALSE, remove_subckt = FALSE, found_model = FALSE, has_models = FALSE;

  for ( card = start_card; card != NULL; card = card->li_next ) {
    if ( ciprefix( ".model", card->li_line ) ) has_models = TRUE;
    if ( ciprefix( ".cmodel", card->li_line ) ) has_models = TRUE;    
    if ( ciprefix( ".param", card->li_line ) && !strstr( card->li_line, "=" ) ) *card->li_line = '*';
  }
  
  for ( card = start_card; card != NULL; card = card->li_next ) {
    line = card->li_line;

    if ( *line == '*' ) continue;

    if ( ciprefix( ".subckt", line ) || ciprefix( ".macro", line ) ) processing_subckt = TRUE;
    if ( ciprefix( ".ends",   line ) || ciprefix( ".eom",   line ) ) processing_subckt = FALSE;
    if ( !processing_subckt ) {
      if ( *line == 'x' ) {
	subckt_name = get_instance_subckt( line );

	found_subckt = FALSE;
	for ( i = 0; i < num_used_subckt_names; i++ )
	  if ( strcmp( used_subckt_names[i], subckt_name ) == 0 ) found_subckt = TRUE;
	if ( !found_subckt ) {
	  used_subckt_names[num_used_subckt_names++] = subckt_name;
	  tmp_cnt++;
	}
	else  tfree( subckt_name );
      }
      else if ( *line == 'a' ) {
	model_name = get_adevice_model_name( line );
	found_model = FALSE;
	for ( i = 0; i < num_used_model_names; i++ )
	  if ( strcmp( used_model_names[i], model_name ) == 0 ) found_model = TRUE;
	if ( !found_model ) used_model_names[num_used_model_names++] = model_name;
	else  tfree( model_name );
      }
      else if ( has_models ) {
	num_terminals = get_number_terminals( line );

	if ( num_terminals != 0 ) {
	  model_name = get_model_name( line, num_terminals );

	  if ( isalpha( *model_name ) ) {
	    found_model = FALSE;
	    for ( i = 0; i < num_used_model_names; i++ )
	      if ( strcmp( used_model_names[i], model_name ) == 0 ) found_model = TRUE;
	    if ( !found_model ) used_model_names[num_used_model_names++] = model_name;
	    else  tfree( model_name );
	  } else {
	    tfree( model_name );
	  }
	}
      }
    }
  }
  for ( i = 0; i < tmp_cnt; i++ )
    get_subckts_for_subckt( start_card, used_subckt_names[i], used_subckt_names,
			    &num_used_subckt_names, used_model_names, &num_used_model_names, has_models );

  // comment out any unused subckts
  for ( card = start_card; card != NULL; card = card->li_next ) {
    line = card->li_line;

    if ( *line == '*' ) continue;

    if ( ciprefix( ".subckt", line ) || ciprefix( ".macro", line ) ) {
      subckt_name = get_subckt_model_name( line );
      remove_subckt = TRUE;
      for ( i = 0; i < num_used_subckt_names; i++ )
	if ( strcmp( used_subckt_names[i], subckt_name ) == 0 ) remove_subckt = FALSE;
      tfree(subckt_name);
    }
    if ( ciprefix( ".ends", line ) || ciprefix( ".eom", line ) ) {
      if ( remove_subckt ) *line = '*';
      remove_subckt = FALSE;
    }
    if ( remove_subckt ) *line = '*';
    else if ( has_models && (ciprefix( ".model", line ) || ciprefix( ".cmodel", line )) ) {
      model_name = get_subckt_model_name( line );

      found_model = FALSE;
      for ( i = 0; i < num_used_model_names; i++ )
	if ( strcmp( used_model_names[i], model_name ) == 0 || model_bin_match( used_model_names[i], model_name ) ) found_model = TRUE;
      if ( !found_model ) *line = '*';
      tfree(model_name);
    }
  }
  for ( i = 0; i < num_used_subckt_names; i++ ) tfree(used_subckt_names[i]);
  for ( i = 0; i < num_used_model_names;  i++ ) tfree(used_model_names[i]);
}

static char*
inp_fix_ternary_operator_str( char *line )
{
  char *conditional, *if_str, *else_str, *question, *colon, keep, *str_ptr, *str_ptr2, *new_str;
  char *paren_ptr = NULL, *end_str = NULL, *beg_str = NULL;
  int  count = 0;

  if ( !strstr( line, "?" ) && !strstr( line, ":" ) ) return line;

  str_ptr = line;
  if ( ciprefix( ".param", line ) || ciprefix( ".func", line ) || ciprefix( ".meas", line ) ) {

    str_ptr = line;
    if ( ciprefix( ".param", line ) || ciprefix( ".meas", line ) ) str_ptr = strstr( line, "=" );
    else                                                           str_ptr = strstr( line, ")" );

    str_ptr++;
    while( isspace(*str_ptr) ) str_ptr++;
    if ( *str_ptr == '{' ) { str_ptr++; while( isspace(*str_ptr) ) str_ptr++; }

    question  = strstr( str_ptr, "?" );
    paren_ptr = strstr( str_ptr, "(" );

    if ( paren_ptr != NULL && paren_ptr < question ) {
      str_ptr = question;
      while ( *str_ptr != '(' ) str_ptr--;
      *str_ptr = '\0';
      beg_str = strdup(line);
      *str_ptr = '(';
      str_ptr++;
      paren_ptr = NULL;
    }
    else {
      keep = *str_ptr; *str_ptr = '\0';
      beg_str = strdup(line);
      *str_ptr = keep;
    }
  }

  // get conditional
  str_ptr2 = question = strstr( str_ptr, "?" );
  str_ptr2--;
  while ( isspace(*str_ptr2) ) str_ptr2--;
  if ( *str_ptr2 == ')' ) {
    while ( *str_ptr != '(' ) str_ptr--;
  }
  str_ptr2++; keep = *str_ptr2; *str_ptr2 = '\0';
  conditional = strdup(str_ptr);
  *str_ptr2 = keep;

  // get if
  str_ptr = question + 1;
  while ( isspace(*str_ptr) ) str_ptr++;
  if ( *str_ptr == '(' ) {
    // find closing paren
    count    = 1;
    str_ptr2 = str_ptr + 1;
    while ( count != 0 && *str_ptr2 != '\0' ) {
      str_ptr2++;
      if ( *str_ptr2 == '(' ) count++;
      if ( *str_ptr2 == ')' ) count--;
    }
    if ( count != 0 ) {
      fprintf(stderr, "ERROR: problem parsing 'if' of ternary string %s!\n", line);
      exit (-1);
    }
    colon = str_ptr2 + 1;
    while ( *colon != ':' && *colon != '\0' ) colon++;
    if ( *colon != ':' ) {
      fprintf(stderr,"ERROR: problem parsing ternary string (finding ':') %s!\n", line);
      exit(-1);
    }
  }
  else if ( ( colon = strstr( str_ptr, ":" ) ) ) {
    str_ptr2 = colon - 1;
    while ( isspace(*str_ptr2) ) str_ptr2--;
  }
  else {
    fprintf(stderr,"ERROR: problem parsing ternary string (missing ':') %s!\n", line);
    exit(-1);
  }
  str_ptr2++; keep = *str_ptr2; *str_ptr2 = '\0';
  if_str = inp_fix_ternary_operator_str(strdup(str_ptr));
  *str_ptr2 = keep;

  // get else
  str_ptr = colon + 1;
  while ( isspace(*str_ptr) ) str_ptr++;
  if ( paren_ptr != NULL ) {
    // find end paren ')'
    bool found_paren = FALSE;
    count = 0; str_ptr2 = str_ptr;
    while ( *str_ptr2 != '\0' ) {
      if ( *str_ptr2 == '(' ) { count++; found_paren = TRUE; }
      if ( *str_ptr2 == ')' ) count--;
      str_ptr2++;
      if ( found_paren && count == 0 ) break;
    }
    if ( found_paren && count != 0 ) {
      fprintf( stderr, "ERROR: problem parsing ternary line %s!\n", line );
      exit(-1);
    }
    keep = *str_ptr2;
    *str_ptr2 = '\0';
    else_str = inp_fix_ternary_operator_str(strdup(str_ptr));
    if ( keep != '}' ) {
      end_str  = inp_fix_ternary_operator_str(strdup(str_ptr2+1));
    } else {
      *str_ptr2 = keep;
      end_str = strdup(str_ptr2);
    }
    *str_ptr2 = keep;
  }
  else {
    if ( ( str_ptr2 = strstr( str_ptr, "}" ) ) ) {
      *str_ptr2 = '\0';
      else_str = inp_fix_ternary_operator_str(strdup(str_ptr));
      *str_ptr2 = '}';
      end_str = strdup(str_ptr2);
    } else {
      else_str = inp_fix_ternary_operator_str(strdup(str_ptr));
    }
  }

  if ( end_str != NULL ) {
    if ( beg_str != NULL ) {
      new_str = tmalloc( strlen(beg_str) + strlen("ternary_fcn") + strlen(conditional) + strlen(if_str) + strlen(else_str) + strlen(end_str) + 5 );
      sprintf( new_str, "%sternary_fcn(%s,%s,%s)%s", beg_str, conditional, if_str, else_str, end_str );
    } else {
      new_str = tmalloc( strlen("ternary_fcn") + strlen(conditional) + strlen(if_str) + strlen(else_str) + strlen(end_str) + 5 );
      sprintf( new_str, "ternary_fcn(%s,%s,%s)%s", conditional, if_str, else_str, end_str );
    }
  }
  else {
    if ( beg_str != NULL ) {
      new_str = tmalloc( strlen(beg_str) + strlen("ternary_fcn") + strlen(conditional) + strlen(if_str) + strlen(else_str) + 5 );
      sprintf( new_str, "%sternary_fcn(%s,%s,%s)", beg_str, conditional, if_str, else_str );
    } else {
      new_str = tmalloc( strlen("ternary_fcn") + strlen(conditional) + strlen(if_str) + strlen(else_str) + 5 );
      sprintf( new_str, "ternary_fcn(%s,%s,%s)", conditional, if_str, else_str );
    }
  }

  tfree(line);
  tfree(conditional); tfree(if_str); tfree(else_str);
  if ( beg_str != NULL ) tfree(beg_str);
  if ( end_str != NULL ) tfree(end_str);

  return new_str;
}

static void
inp_fix_ternary_operator( struct line *start_card )
{
  struct line *card;
  char        *line;

  for ( card = start_card; card != NULL; card = card->li_next ) {
    line = card->li_line;

    if ( *line == '*' ) continue;
    if ( strstr( line, "?" ) && strstr( line, ":" ) ) {
      card->li_line = inp_fix_ternary_operator_str( line );
    }
  }
}

/*-------------------------------------------------------------------------
 * Read the entire input file and return  a pointer to the first line of   
 * the linked list of 'card' records in data.  The pointer is stored in
 * *data.
 *-------------------------------------------------------------------------*/
void
inp_readall(FILE *fp, struct line **data, int call_depth, char *dir_name)
{
   struct line *end = NULL, *cc = NULL, *prev = NULL, *working, *newcard, *start_lib, *global_card, *tmp_ptr = NULL, *tmp_ptr2 = NULL;
   char *buffer = NULL, *s, *t, *y, *z, c;
   /* segfault fix */
#ifdef XSPICE
   char big_buff[5000];
   int line_count = 0;
   Ipc_Status_t    ipc_status;
   char            ipc_buffer[1025];  /* Had better be big enough */
   int             ipc_len;
#endif
   char *copys=NULL, big_buff2[5000];
   char *global_copy = NULL, keep_char;
   int line_number = 1; /* sjb - renamed to avoid confusion with struct line */
   int line_number_orig = 1, line_number_lib = 1, line_number_inc = 1;
   int no_braces = 0; /* number of '{' */
   FILE *newfp;
#if defined(TRACE) || defined(OUTDECK)
   FILE *fdo;
#endif    
   struct line *tmp_ptr1 = NULL;    

   int i, j;
   bool found_library, found_lib_name, found_end = FALSE, shell_eol_continuation = FALSE;
   bool dir_name_flag = FALSE;

   struct variable *v;
   char *s_ptr, *s_lower;

   /*   Must set this to NULL or non-tilde includes segfault. -- Tim Molteno   */
   /* copys = NULL; */   /*  This caused a parse error with gcc 2.96.  Why???  */

   if ( call_depth == 0 ) {
      num_subckt_w_params = 0;
      num_libraries       = 0;
      num_functions       = 0;
      global              = NULL;
      found_end           = FALSE;
   }

/*   gtri - modify - 12/12/90 - wbk - read from mailbox if ipc enabled   */
#ifdef XSPICE

   /* First read in all lines & put them in the struct cc */
   while (1) {
      /* If IPC is not enabled, do equivalent of what SPICE did before */
      if(! g_ipc.enabled) {
         if ( call_depth == 0 && line_count == 0 ) {
            line_count++;
            if ( fgets( big_buff, 5000, fp ) ) {
               buffer = tmalloc( strlen(big_buff) + 1 ); 
               strcpy(buffer, big_buff);
            }
         }
         else {
            buffer = readline(fp);
            if(! buffer) {
               break;
            }
         }
      }
      else {
         /* else, get the line from the ipc channel. */
         /* We assume that newlines are not sent by the client */
         /* so we add them here */
         ipc_status = ipc_get_line(ipc_buffer, &ipc_len, IPC_WAIT);
         if(ipc_status == IPC_STATUS_END_OF_DECK) {
            buffer = NULL;
            break;
         }
         else if(ipc_status == IPC_STATUS_OK) {
            buffer = (void *) tmalloc(strlen(ipc_buffer) + 3);
            strcpy(buffer, ipc_buffer);
            strcat(buffer, "\n");
         }
         else {  /* No good way to report this so just die */
            exit(1);
         }
      }

/* gtri - end - 12/12/90 */
#else
   while ((buffer = readline(fp))) {
#endif

#ifdef TRACE
      /* SDB debug statement */
      printf ("in inp_readall, just read   %s", buffer); 
#endif

      if ( !buffer ) {
         continue;
      }
      /* OK -- now we have loaded the next line into 'buffer'.  Process it. */
      /* If input line is blank, ignore it & continue looping.  */
      if ( (strcmp(buffer,"\n") == 0) || (strcmp(buffer,"\r\n") == 0) ) {
         if ( call_depth != 0 || (call_depth == 0 && cc != NULL) ) {
            line_number_orig++;
            tfree(buffer);      /* was allocated by readline() */
            continue;
         }
      }

      if (*buffer == '@') {
         tfree(buffer);		/* was allocated by readline() */
         break;
      }

      /* now handle .lib statements */
      if (ciprefix(".lib", buffer)) {
         for   ( s = buffer; *s && !isspace(*s); s++ );         /* skip over .lib           */
         while ( isspace(*s) || isquote(*s) ) s++;              /* advance past space chars              */
         if    ( !*s ) {                                        /* if at end of line, error              */
            fprintf(cp_err, "Error: .lib filename missing\n");
            tfree(buffer);		                           /* was allocated by readline()           */
            continue;
         }                                                      /* Now s points to first char after .lib */
         for   ( t = s; *t && !isspace(*t) && !isquote(*t); t++ );         /* skip to end of word      */
         y = t;
         while ( isspace(*y) || isquote(*y) ) y++;              /* advance past space chars */
         // check if rest of line commented out
         if    ( *y && *y != '$' ) {                            /* .lib <file name> <lib name> */
            for ( z = y; *z && !isspace(*z) && !isquote(*z); z++ );
            c  = *t;
            *t = '\0';
            *z = '\0';

            if ( *s == '~' ) {
               copys = cp_tildexpand(s); /* allocates memory, but can also return NULL */
               if( copys != NULL ) {
                  s = copys;		  /* reuse s, but remember, buffer still points to allocated memory */
               }
            }
	      
	         /* lower case the file name for later string compares */
	         s_ptr = strdup(s);
	         s_lower = strdup(s);
	         for(s_ptr = s_lower; *s_ptr && (*s_ptr != '\n'); s_ptr++) *s_ptr = tolower(*s_ptr);

            found_library = FALSE;
            for ( i = 0; i < num_libraries; i++ ) {
               if ( strcmp( library_file[i], s_lower ) == 0 ) {
                  found_library = TRUE;
                  break;
               }
            }
            if ( found_library ) {
               if(copys) tfree(copys);   /* allocated by the cp_tildexpand() above */
            } else {
               if ( dir_name != NULL ) sprintf( big_buff2, "%s/%s", dir_name, s );
               else                    sprintf( big_buff2, "./%s", s );
               dir_name_flag = FALSE;
               if ( !( newfp = inp_pathopen( s, "r" ) ) ) {
                  dir_name_flag = TRUE;
                  if ( !( newfp = inp_pathopen( big_buff2, "r" ) ) ) { 
                     perror(s);
                     if(copys) tfree(copys);   /* allocated by the cp_tildexpand() above */
                     tfree(buffer);	    /* allocated by readline() above */
                     continue;
                  }
               }
               if(copys) tfree(copys);     /* allocated by the cp_tildexpand() above */

               library_file[num_libraries++] = strdup(s_lower);

               if ( dir_name_flag == FALSE ) {
                  char *s_dup = strdup(s);
                  inp_readall(newfp, &libraries[num_libraries-1], call_depth+1, dirname(s_dup));
                  tfree(s_dup);
               }
               else
                  inp_readall(newfp, &libraries[num_libraries-1], call_depth+1, dir_name);
            
               fclose(newfp);
            }
            *t = c;
            tfree(s_lower);

            /* Make the .lib a comment */
            *buffer = '*';
         }
      }   /*  end of .lib handling  */

      /* now handle .include statements */
      if (ciprefix(".include", buffer) || ciprefix(".inc", buffer)) {
         for (s = buffer; *s && !isspace(*s); s++) /* advance past non-space chars */
            ;
         while (isspace(*s) || isquote(*s))        /* now advance past space chars */
            s++;
         if (!*s) {                                /* if at end of line, error */
            fprintf(cp_err,  "Error: .include filename missing\n");
            tfree(buffer);		/* was allocated by readline() */
            continue;
         }                           /* Now s points to first char after .include */
         for (t = s; *t && !isspace(*t) && !isquote(*t); t++)     /* now advance past non-space chars */
            ;
         *t = '\0';                         /* place \0 and end of file name in buffer */

         if (*s == '~') {
            copys = cp_tildexpand(s); /* allocates memory, but can also return NULL */
            if(copys != NULL) {
               s = copys;		/* reuse s, but remember, buffer still points to allocated memory */
            }
         }
	    
         /* open file specified by  .include statement */
         if ( dir_name != NULL ) sprintf( big_buff2, "%s/%s", dir_name, s );
         else                    sprintf( big_buff2, "./%s", s );
         dir_name_flag = FALSE;
         if (!(newfp = inp_pathopen(s, "r"))) { 
            dir_name_flag = TRUE;
            if ( !( newfp = inp_pathopen( big_buff2, "r" ) ) ) { 
               perror(s);
               if(copys) {
                  tfree(copys);	/* allocated by the cp_tildexpand() above */
               }
               tfree(buffer);		/* allocated by readline() above */
               continue;
            }
         }
	    
         if(copys) {
            tfree(copys);		/* allocated by the cp_tildexpand() above */
         }  

         if ( dir_name_flag == FALSE ) {
            char *s_dup = strdup(s);
            inp_readall(newfp, &newcard, call_depth+1, dirname(s_dup));  /* read stuff in include file into netlist */
            tfree(s_dup);
         }
         else
            inp_readall(newfp, &newcard, call_depth+1, dir_name);  /* read stuff in include file into netlist */

         (void) fclose(newfp);

         /* Make the .include a comment */
         *buffer = '*';

         /* now check if this is the first pass (i.e. end points to null) */
         if (end) {                            /* end already exists */
            end->li_next = alloc(struct line);  /* create next card */
            end = end->li_next;                 /* make end point to next card */
         } else {
            end = cc = alloc(struct line);   /* create the deck & end.  cc will
						point to beginning of deck, end to 
						the end */
         }

         /* now fill out rest of struct end. */
         end->li_next = NULL;
         end->li_error = NULL;
         end->li_actual = NULL;
         end->li_line = copy(buffer);
         end->li_linenum = end->li_linenum_orig = line_number++;
         if (newcard) {
            end->li_next = newcard;
            /* Renumber the lines */
            line_number_inc = 1;
            for (end = newcard; end && end->li_next; end = end->li_next) {
               end->li_linenum = line_number++;
               end->li_linenum_orig = line_number_inc++;
            }
            end->li_linenum = line_number++;	/* SJB - renumber the last line */
            end->li_linenum_orig = line_number_inc++;	/* SJB - renumber the last line */
         }

         /* Fix the buffer up a bit. */
         (void) strncpy(buffer + 1, "end of:", 7);
      }   /*  end of .include handling  */

      /* loop through 'buffer' until end is reached.  Then test for
	   premature end.  If premature end is reached, spew
	   error and zap the line. */
      if ( !ciprefix( "write", buffer ) ) {    // exclude 'write' command so filename case preserved
         for (s = buffer; *s && (*s != '\n') && (*s != '\0'); s++) *s = tolower(*s);
         if (!*s) {
         //fprintf(cp_err, "Warning: premature EOF\n");
         }
         *s = '\0';      /* Zap the newline. */
	
         if((s-1) >= buffer && *(s-1) == '\r') /* Zop the carriage return under windows */
            *(s-1) = '\0';
      }

      if (ciprefix(".end", buffer) && strlen(buffer) == 4 ) {
         found_end = TRUE;
         *buffer   = '*';
      }

      if ( ciprefix( ".global", buffer ) ) {
         for ( s = buffer; *s && !isspace(*s); s++ );

         if ( global == NULL ) {
            global = strdup(buffer);
         } else {
            global_copy = tmalloc( strlen(global) + strlen(s) + 1 );
            sprintf( global_copy, "%s%s", global, s );
            tfree(global);
            global = global_copy;
         }
         *buffer = '*';
      }

      if ( shell_eol_continuation ) {
         char *new_buffer = tmalloc( strlen(buffer) + 2);
         sprintf( new_buffer, "+%s", buffer );

         tfree(buffer);
         buffer = new_buffer;
      }

      shell_eol_continuation = chk_for_line_continuation( buffer );

      /* now check if this is the first pass (i.e. end points to null) */
      if (end) {                              /* end already exists */
         end->li_next = alloc(struct line);    /* create next card */
         end = end->li_next;                   /* point to next card */
      } else {                              /* End doesn't exist.  Create it. */
         end = cc = alloc(struct line);   /* note that cc points to beginning
					      of deck, end to the end */
      }
	
      /* now put buffer into li */
      end->li_next = NULL;
      end->li_error = NULL;
      end->li_actual = NULL;
      end->li_line = copy(buffer);
      end->li_linenum = line_number++;
      end->li_linenum_orig = line_number_orig++;
      tfree(buffer);
   }  /* end while ((buffer = readline(fp))) */

   if (!end) { /* No stuff here */
      *data = NULL;
      return;
   }

   if ( call_depth == 0 && found_end == TRUE) {
      if ( global == NULL ) {
         global = tmalloc( strlen(".global gnd") + 1 );
         sprintf( global, ".global gnd" );
      }
      global_card = alloc(struct line);
      global_card->li_error   = NULL;
      global_card->li_actual  = NULL;
      global_card->li_line    = global;
      global_card->li_linenum = 1;

      prev                 = cc->li_next;
      cc->li_next          = global_card;
      global_card->li_next = prev;

      inp_init_lib_data();
      inp_determine_libraries(cc, NULL);
   }

   /*
      add libraries
   */
   found_lib_name = FALSE;
   if ( call_depth == 0 ) {
      for( i = 0; i < num_libraries; i++ ) {
         working = libraries[i];
         while ( working ) {
            buffer = working->li_line;

            if ( found_lib_name && ciprefix(".endl", buffer) ) {
               /* Make the .endl a comment */
               *buffer = '*';
               found_lib_name = FALSE;
 
               /* set pointer and continue to avoid deleting below */
               tmp_ptr2         = working->li_next;
               working->li_next = tmp_ptr;
               working          = tmp_ptr2;

               /* end          = working;
               * working      = working->li_next;
               * end->li_next = NULL; */

               continue;
            }   /* for ... */

            if ( ciprefix(".lib", buffer) ) {
               if ( found_lib_name == TRUE ) {
                  fprintf( stderr, "ERROR: .lib is missing .endl!\n" );
                  exit(-1);
               }

               for   ( s = buffer; *s && !isspace(*s); s++ );            /* skip over .lib                        */
               while ( isspace(*s) || isquote(*s) ) s++;                 /* advance past space chars              */
               for   ( t = s; *t && !isspace(*t) && !isquote(*t); t++ ); /* skip to end of word                   */
               keep_char = *t;
               *t = '\0';
               /* see if library we want to copy */
               found_lib_name = FALSE;
               for( j = 0; j < num_lib_names[i]; j++ ) {
                  if ( strcmp( library_name[i][j], s ) == 0 ) {
                     found_lib_name = TRUE;
                     start_lib      = working;

                     /* make the .lib a comment */
                     *buffer = '*';

                     tmp_ptr = library_ll_ptr[i][j]->li_next;
                     library_ll_ptr[i][j]->li_next = working;

                     /* renumber lines */
                     line_number_lib = 1;
                     for ( start_lib = working; !ciprefix(".endl", start_lib->li_line); start_lib = start_lib->li_next ) {
                        start_lib->li_linenum = line_number++;
                        start_lib->li_linenum_orig = line_number_lib++;
                     }
                     start_lib->li_linenum = line_number++;  // renumber endl line
                     start_lib->li_linenum_orig = line_number_lib++; 
                     break;
                  }
               }
               *t = keep_char;
            }
            prev = working;
            working = working->li_next;

            if ( found_lib_name == FALSE ) {
               tfree(prev->li_line);
               tfree(prev);
            }
         } /* end while */
      } /* end for */

      if ( found_end == TRUE ) {
         end->li_next = alloc(struct line);    /* create next card */
         end = end->li_next;                   /* point to next card */
  
         buffer = tmalloc( strlen( ".end" ) + 1 );
         sprintf( buffer, ".end" );

         /* now put buffer into li */
         end->li_next = NULL;
         end->li_error = NULL;
         end->li_actual = NULL;
         end->li_line = buffer;
         end->li_linenum = end->li_linenum_orig = line_number++;
         end->li_linenum_orig = line_number_orig++;
      }
   }

   /* Now clean up li: remove comments & stitch together continuation lines. */
   working = cc->li_next;      /* cc points to head of deck.  Start with the
				   next card. */

   /* sjb - strip or convert end-of-line comments.
       This must be cone before stitching continuation lines.
       If the line only contains an end-of-line comment then it is converted
       into a normal comment with a '*' at the start.  This will then get
       stripped in the following code. */			   		   
   inp_stripcomments_deck(working);	

   while (working) {
      for (s = working->li_line; (c = *s) && c <= ' '; s++)
         ;

#ifdef TRACE
	/* SDB debug statement */
	printf("In inp_readall, processing linked list element line = %d, s = %s . . . \n", working->li_linenum,s); 
#endif

      switch (c) {
         case '#':
         case '$':
         case '*':
         case '\0':
         /* this used to be commented out.  Why? */
         /*  prev = NULL; */
            working = working->li_next;  /* for these chars, go to next card */
            break;

         case '+':   /* handle continuation */
            if (!prev) {
               working->li_error = copy(
                  "Illegal continuation line: ignored.");
               working = working->li_next;
               break;
            }

            /* create buffer and write last and current line into it. */
            buffer = tmalloc(strlen(prev->li_line) + strlen(s) + 2);
            (void) sprintf(buffer, "%s %s", prev->li_line, s + 1); 

            s = prev->li_line;
            prev->li_line = buffer;
            prev->li_next = working->li_next;
            working->li_next = NULL;
            if (prev->li_actual) {
               for (end = prev->li_actual; end->li_next; end = end->li_next)
                  ;
               end->li_next = working;
               tfree(s);
            } else {
               newcard = alloc(struct line);
               newcard->li_linenum = prev->li_linenum;
               newcard->li_line = s;
               newcard->li_next = working;
               newcard->li_error = NULL;
               newcard->li_actual = NULL;
               prev->li_actual = newcard;
            }
            working = prev->li_next;
            break;

         default:  /* regular one-line card */
            prev = working;
            working = working->li_next;
            break;
      }
   }

   working = cc->li_next;

   inp_fix_for_numparam(working);
   inp_remove_excess_ws(working);

   if ( call_depth == 0 ) {
      comment_out_unused_subckt_models(working);

      line_number = inp_split_multi_param_lines(working, line_number);

      inp_fix_macro_param_func_paren_io(working);
      inp_fix_ternary_operator(working);
      inp_grab_func(working);

      inp_expand_macros_in_func();
      inp_expand_macros_in_deck(working);
      inp_fix_param_values(working);

      /* get end card as last card in list; end card pntr does not appear to always
         be correct at this point */
      for(newcard = working; newcard != NULL; newcard = newcard->li_next)
         end = newcard;

      inp_reorder_params(working, cc, end);
      inp_fix_inst_calls_for_numparam(working);
      inp_fix_gnd_name(working);
      inp_chk_for_multi_in_vcvs(working, &line_number);

      if (cp_getvar("addcontrol", VT_BOOL, (char *) &v))
         inp_add_control_section(working, &line_number);
   }
   *data = cc;
    
   /* get max. line length and number of lines in input deck,
      and renumber the lines,
      count the number of '{' per line as an upper estimate of the number
      of parameter substitutions in a line*/
   dynmaxline = 0;
   dynLlen = 0;
   for(tmp_ptr1 = cc; tmp_ptr1 != NULL; tmp_ptr1 = tmp_ptr1->li_next) {
      char *s;
      int braces_per_line = 0;
      /* count number of lines */
      dynmaxline++;
      /* renumber the lines of the processed input deck */
      tmp_ptr1->li_linenum = dynmaxline;
      if (dynLlen < strlen(tmp_ptr1->li_line))
         dynLlen = strlen(tmp_ptr1->li_line);
      /* count '{' */
      for (s = tmp_ptr1->li_line; *s; s++)
         if (*s == '{') braces_per_line++;
      if (no_braces <  braces_per_line) no_braces = braces_per_line;
   }

#if defined(TRACE) || defined(OUTDECK)
   /*debug: print into file*/
   fdo = fopen("debug-out.txt", "w");
   for(tmp_ptr1 = cc; tmp_ptr1 != NULL; tmp_ptr1 = tmp_ptr1->li_next)
      fprintf(fdo, "%d  %d  %s\n", tmp_ptr1->li_linenum_orig, tmp_ptr1->li_linenum, tmp_ptr1->li_line);        

   (void) fclose(fdo);  
   fprintf(stdout, "max line length %d, max subst. per line %d, number of lines %d\n", 
      dynLlen, no_braces, dynmaxline);
#endif
   /* max line length increased by maximum number of parameter substitutions per line 
      times parameter string length (25) */
   dynLlen += no_braces * 25;
   /* several times a string of length dynLlen is used for messages, thus give it a
      minimum length */
   if (dynLlen < 512) dynLlen = 512;
   return;
}

/*-------------------------------------------------------------------------*
 *                                                                         *
 *-------------------------------------------------------------------------*/
void
inp_casefix(char *string)
{
#ifdef HAVE_CTYPE_H
    if (string)
	while (*string) {
	    /* Let's make this really idiot-proof. */
#ifdef HAS_ASCII
	    *string = strip(*string);
#endif
	    if (*string == '"') {
		*string++ = ' ';
		while (*string && *string != '"')
		    string++;
		if (*string == '"')
		    *string = ' ';
	    }
	    if (!isspace(*string) && !isprint(*string))
		*string = '_';
	    if (isupper(*string))
		*string = tolower(*string);
	    string++;
	}
    return;
#endif
}
 
/* Strip all end-of-line comments from a deck */
static void
inp_stripcomments_deck(struct line *deck)
{
    struct line *c=deck;
    while( c!=NULL) {
      inp_stripcomments_line(c->li_line);
	c= c->li_next;
    }
}

/* Strip end of line comment from a string and remove trailing white space
   supports comments that begin with single characters ';'
   or double characters '$ ' or '//' or '--'
   If there is only white space before the end-of-line comment the
   the whole line is converted to a normal comment line (i.e. one that
   begins with a '*').
   BUG: comment characters in side of string literals are not ignored. */  
static void
inp_stripcomments_line(char * s)
{
    char c = ' '; /* anything other than a comment character */
    char * d = s;
    if(*s=='\0') return;	/* empty line */
    
    /* look for comments */
    while((c=*d)!='\0') {
	d++;
	if (*d==';') {	
	    break;
	} else if ((c=='$') && (*d==' ')) {
	    d--; /* move d back to first comment character */
	    break;
	} else if( (*d==c) && ((c=='/') || (c=='-'))) {
	    d--; /* move d back to first comment character */
	    break;
	}
    }
    /* d now points to the first comment character of the null at the string end */
    
    /* check for special case of comment at start of line */
    if(d==s) {
	*s = '*'; /* turn into normal comment */
	return;
    }
    
    if(d>s) {
	d--;
	/* d now points to character just before comment */
	
	/* eat white space at end of line */
	while(d>=s) {
	    if( (*d!=' ') && (*d!='\t' ) )
		break;
	    d--;
	}
	d++;
	/* d now points to the first white space character before the
	   end-of-line or end-of-line comment, or it points to the first
	   end-of-line comment character, or to the begining of the line */
    }
       
    /* Check for special case of comment at start of line 
       with or without preceeding white space */
    if(d<=s) {
	*s = '*'; /* turn the whole line into normal comment */
	return;
    }
    
    *d='\0'; /* terminate line in new location */    
}

static void
inp_change_quotes( char *s )
{
  bool first_quote = FALSE;

  while ( *s ) {
    if ( *s == '\'' ) {
      if ( first_quote == FALSE ) {
	*s = '{';
	first_quote = TRUE;
      } else {
	*s = '}';
	first_quote = FALSE;
      }
    } 
    s++;
  }
}

static char*
inp_fix_subckt( char *s )
{
  struct line *head=NULL, *newcard=NULL, *start_card=NULL, *end_card=NULL, *prev_card=NULL, *c=NULL;
  char *equal = strstr( s, "=" );
  char *beg, *buffer, *ptr1, *ptr2, *str, *new_str = NULL;
  char keep;
  int  num_params = 0, i = 0;

  if ( !strstr( s, "params:" ) && equal != NULL ) {
    /* get subckt name (ptr1 will point to name) */
    for ( ptr1 = s; *ptr1 && !isspace(*ptr1); ptr1++ );
    while ( isspace(*ptr1) ) ptr1++;
    for ( ptr2 = ptr1; *ptr2 && !isspace(*ptr2) && !isquote(*ptr2); ptr2++ );

    keep  = *ptr2;
    *ptr2 = '\0';

    subckt_w_params[num_subckt_w_params++] = strdup(ptr1);

    *ptr2 = keep;

    /* go to beginning of first parameter word  */
    /* s    will contain only subckt definition */
    /* beg  will point to start of param list   */
    for ( beg = equal-1; *beg && isspace(*beg); beg-- );
    for (              ; *beg && !isspace(*beg); beg-- );
    *beg = '\0';
    beg++;

    head = alloc(struct line);
    /* create list of parameters that need to get sorted */
    while ( ( ptr1 = strstr( beg, "=" ) ) ) {
      ptr2 = ptr1+1;
      ptr1--;
      while ( isspace(*ptr1)  ) ptr1--;
      while ( !isspace(*ptr1) && *ptr1 != '\0' ) ptr1--;
      ptr1++;                                    /* ptr1 points to beginning of parameter */

      while ( isspace(*ptr2) ) ptr2++;
      while ( *ptr2 && !isspace(*ptr2) ) ptr2++; /* ptr2 points to end of parameter       */

      keep  = *ptr2;
      *ptr2 = '\0';
      beg   = ptr2+1;

      newcard = alloc(struct line);
      str     = strdup(ptr1);

      newcard->li_line = str;
      newcard->li_next = NULL;

      if ( start_card == NULL  ) head->li_next = start_card = newcard;
      else                       prev_card->li_next = newcard;

      prev_card = end_card = newcard;
      num_params++;
    }

    /* now sort parameters in order of dependencies */
    inp_sort_params( start_card, end_card, head, start_card, end_card );

    /* create new ordered parameter string for subckt call */
    c=head->li_next;
    tfree(head);
    for( i = 0; i < num_params && c!= NULL; i++ ) {
      if ( new_str == NULL ) new_str = strdup(c->li_line);
      else {
	str     = new_str;
	new_str = tmalloc( strlen(str) + strlen(c->li_line) + 2 );
	sprintf( new_str, "%s %s", str, c->li_line );
	tfree(str);
      }
      tfree(c->li_line);
      head = c;
      c = c->li_next;
      tfree(head);
    }

    /* create buffer and insert params: */ 
    buffer = tmalloc( strlen(s) + 9 + strlen(new_str) + 1 );
    sprintf( buffer, "%s params: %s", s, new_str );

    tfree(s); tfree(new_str);

    s = buffer;
  }
  return s;
}

static char*
inp_remove_ws( char *s )
{
   char *big_buff;
   int  big_buff_index = 0;
   char *buffer, *curr;
   bool is_expression = FALSE;
  
   big_buff = tmalloc( strlen(s) + 1 );
   curr = s;

   while ( *curr != '\0' ) {
      if ( *curr == '{' ) is_expression = TRUE;
      if ( *curr == '}' ) is_expression = FALSE;

      big_buff[big_buff_index++] = *curr;
      if ( *curr == '=' || (is_expression && (is_arith_char(*curr) || *curr == ',')) ) {
         curr++;
         while ( isspace(*curr) ) curr++;

         if ( *curr == '{' ) is_expression = TRUE;
         if ( *curr == '}' ) is_expression = FALSE;

         big_buff[big_buff_index++] = *curr;
      }
      curr++;
      if ( isspace(*curr) ) {
         while ( isspace(*curr) ) curr++;
         if ( is_expression ) {
            if ( *curr != '=' && !is_arith_char(*curr) && *curr != ',' )    big_buff[big_buff_index++] = ' ';
         } else {
            if ( *curr != '=' ) big_buff[big_buff_index++] = ' ';
         }
      }
   }
//  big_buff[big_buff_index++] = *curr;
   big_buff[big_buff_index] = '\0';

   buffer = copy(big_buff);

   tfree(s);
   tfree(big_buff);

   return buffer;
}

static void
inp_fix_for_numparam(struct line *deck)
{
   bool found_control = FALSE;
   struct line *c=deck;
   while( c!=NULL) {
      if ( ciprefix( ".modif", c->li_line ) ) *c->li_line = '*';
      if ( ciprefix( "*lib", c->li_line ) ) {
         c = c->li_next;
         continue;
      }
 
      /* exclude plot line between .control and .endc from getting quotes changed */
      if ( ciprefix( ".control", c->li_line ) ) found_control = TRUE;
      if ( ciprefix( ".endc",    c->li_line ) ) found_control = FALSE; 
      if ((found_control) && (ciprefix( "plot", c->li_line ))) {
         c = c->li_next;
         continue;
      }      	 
      
      if ( !ciprefix( "*lib", c->li_line ) && !ciprefix( "*inc", c->li_line ) )
         inp_change_quotes(c->li_line);

      if ( ciprefix( ".subckt", c->li_line ) ) {
         c->li_line = inp_fix_subckt(c->li_line);
      }

      c = c->li_next;
   }
}

static void
inp_remove_excess_ws(struct line *deck )
{
  struct line *c = deck;
  while ( c != NULL ) {
    if ( *c->li_line == '*' ) { c = c->li_next; continue; }
    c->li_line = inp_remove_ws(c->li_line); /* freed in fcn */
    c = c->li_next;
  }
}

static void
inp_determine_libraries( struct line *deck, char *lib_name )
{
  struct line *c = deck;
  char *line, *s, *t, *y, *z, *copys, keep_char1, keep_char2;
  int i, j;
  bool found_lib_name = FALSE;
  bool read_line = FALSE;

  if ( lib_name == NULL ) read_line = TRUE;

  while ( c != NULL ) {
    line = c->li_line;

    if ( ciprefix( ".endl", line ) && lib_name != NULL ) read_line = FALSE;

    if ( ciprefix( "*lib", line ) || ciprefix( ".lib", line ) ) {
      for ( s = line; *s && !isspace(*s); s++);
      while ( isspace(*s) || isquote(*s) ) s++;
      for ( t = s; *t && !isspace(*t) && !isquote(*t); t++ );
      y = t;
      while ( isspace(*y) || isquote(*y) ) y++;

      /* .lib <lib name> */
      if ( !*y ) {
	keep_char1 = *t;
	*t = '\0';

	if ( lib_name != NULL && strcmp( lib_name, s ) == 0 ) read_line = TRUE;
	*t = keep_char1;
      }
      /* .lib <file name> <lib name> */
      else if ( read_line == TRUE ) {
	for ( z = y; *z && !isspace(*z) && !isquote(*z); z++ );
	keep_char1 = *t; keep_char2 = *z;
	*t = '\0'; *z = '\0';

	if ( *s == '~' ) {
	  copys = cp_tildexpand(s);
	  if ( copys != NULL ) {
	    s = copys;
	  }
	}
	for ( i = 0; i < num_libraries; i++ )
	  if ( strcmp( library_file[i], s ) == 0 ) {
	    found_lib_name = FALSE;
	    for ( j = 0; j < num_lib_names[i] && found_lib_name == FALSE; j++ )
	      if ( strcmp( library_name[i][j], y ) == 0 ) found_lib_name = TRUE;

	    if ( found_lib_name == FALSE ) {
	      library_ll_ptr[i][num_lib_names[i]] = c;
	      library_name[i][num_lib_names[i]++] = strdup(y);
	      /* see if other libraries referenced */
	      inp_determine_libraries( libraries[i], y );
	    }
	  }
	*line = '*';  /* comment out .lib line */
	*t = keep_char1; *z = keep_char2;
	}
      }
    c = c->li_next;
  }
}

static void
inp_init_lib_data()
{
  int i;

  for ( i = 0; i < num_libraries; i++ )
    num_lib_names[i] = 0;
}

static char*
inp_get_subckt_name( char *s )
{
  char *end_ptr = strstr( s, "=" );
  char *subckt_name, *subckt_name_copy;
  char keep;

  if ( end_ptr != NULL ) {
    end_ptr--;
    while ( isspace(*end_ptr) ) end_ptr--;
    for( ;*end_ptr && !isspace(*end_ptr); end_ptr--);
  } else {
    end_ptr = s + strlen(s);
  }

  subckt_name = end_ptr;
  while ( isspace( *subckt_name ) ) subckt_name--;
  for( ; !isspace(*subckt_name); subckt_name-- );
  subckt_name++;

  keep     = *end_ptr;
  *end_ptr = '\0';

  subckt_name_copy = strdup( subckt_name );

  *end_ptr = keep;

  return subckt_name_copy;
}

static int
inp_get_params( char *line, char *param_names[], char *param_values[] )
{
  char *equal_ptr = strstr( line, "=" );
  char *end, *name, *value;
  int  num_params = 0;
  char tmp_str[1000];
  char keep;
  bool is_expression = FALSE;

  while ( ( equal_ptr = strstr( line, "=" ) ) ) {

    // check for equality '=='
    if ( *(equal_ptr+1) == '=' ) { line = equal_ptr+2; continue; }
    // check for '!=', '<=', '>='
    if ( *(equal_ptr-1) == '!' || *(equal_ptr-1) == '<' || *(equal_ptr-1) == '>' ) { line = equal_ptr+1; continue; }

    is_expression = FALSE;

    /* get parameter name */
    name = equal_ptr - 1;
    while ( *name && isspace(*name)  ) name--;
    end  = name + 1;
    while ( *name && !isspace(*name) ) name--;
    name++;

    keep = *end;
    *end = '\0';
    param_names[num_params++] = strdup(name);
    *end = keep;

    /* get parameter value */
    value = equal_ptr + 1;
    while ( *value && isspace(*value) ) value++;

    if ( *value == '{' ) is_expression = TRUE;
    end = value;
    if ( is_expression ) {
      while ( *end && *end != '}' ) end++;
    } else {
      while ( *end && !isspace(*end)    ) end++;
    }
    if ( is_expression ) end++;
    keep = *end;
    *end = '\0';

    if ( *value != '{' && 
	 !( isdigit( *value ) || ( *value == '.' && isdigit(*(value+1)) ) ) ) {
      sprintf( tmp_str, "{%s}", value );
      value = tmp_str;
    }
    param_values[num_params-1] = strdup(value);
    *end = keep;
    
    line = end;
  }

  return num_params;
}

static char*
inp_fix_inst_line( char *inst_line,
		   int num_subckt_params, char *subckt_param_names[], char *subckt_param_values[],
		   int num_inst_params,   char *inst_param_names[],   char *inst_param_values[] )
{
  char *end = strstr( inst_line, "=" ), *inst_name, *inst_name_end = inst_line;
  char *curr_line = inst_line, *new_line = NULL;
  char keep;
  int i, j;

  while ( !isspace(*inst_name_end) ) inst_name_end++;
  keep            = *inst_name_end;
  *inst_name_end  = '\0';
  inst_name       = strdup( inst_line );
  *inst_name_end  = keep;

  if ( end != NULL ) {
    end--;
    while ( isspace( *end )  ) end--;
    while ( !isspace( *end ) ) end--;
    *end = '\0';
  }

  for ( i = 0; i < num_subckt_params; i++ ) {
    for ( j = 0; j < num_inst_params; j++ ) {
      if ( strcmp( subckt_param_names[i], inst_param_names[j] ) == 0 ) {
	tfree( subckt_param_values[i] );
	subckt_param_values[i] = strdup( inst_param_values[j] );
      }
    }
  }

  for ( i = 0; i < num_subckt_params; i++ ) {
    new_line = tmalloc( strlen( curr_line ) + strlen( subckt_param_values[i] ) + 2 );
    sprintf( new_line, "%s %s", curr_line, subckt_param_values[i] );

    tfree( curr_line );
    tfree( subckt_param_names[i] );
    tfree( subckt_param_values[i] );

    curr_line = new_line;
  }

  for ( i = 0; i < num_inst_params; i++ ) {
    tfree( inst_param_names[i] );
    tfree( inst_param_values[i] );
  }
  
  tfree( inst_name );

  return curr_line;
}

static bool
found_mult_param( int num_params, char *param_names[] )
{
  bool found_mult = FALSE;
  int i;

  for ( i = 0; i < num_params; i++ ) {
    if ( strcmp( param_names[i], "m" ) == 0 ) {
      found_mult = TRUE;
    }
  }
  return found_mult;
}
 
static int
inp_fix_subckt_multiplier( struct line *subckt_card,
			   int num_subckt_params, char *subckt_param_names[], char *subckt_param_values[] )
{
  struct line *card;
  char *new_str;

  subckt_param_names[num_subckt_params]  = strdup("m");
  subckt_param_values[num_subckt_params] = strdup("1");
  num_subckt_params = num_subckt_params + 1;

  if ( !strstr( subckt_card->li_line, "params:" ) ) {
    new_str = tmalloc( strlen( subckt_card->li_line ) + 13 );
    sprintf( new_str, "%s params: m=1", subckt_card->li_line );
    subckt_w_params[num_subckt_w_params++] = get_subckt_model_name( subckt_card->li_line );
  } else {
    new_str = tmalloc( strlen( subckt_card->li_line ) + 5 );
    sprintf( new_str, "%s m=1", subckt_card->li_line );
  }

  tfree( subckt_card->li_line );
  subckt_card->li_line = new_str;
  
  for ( card = subckt_card->li_next;
	card != NULL && !ciprefix( ".ends", card->li_line );
	card = card->li_next ) {
    new_str = tmalloc( strlen( card->li_line ) + 7 );
    sprintf( new_str, "%s m={m}", card->li_line );
    
    tfree( card->li_line );
    card->li_line = new_str;
  }

  return num_subckt_params;
}

static void
inp_fix_inst_calls_for_numparam(struct line *deck)
{
  struct line *c = deck;
  struct line *d, *p = NULL;
  char *inst_line;
  char *subckt_line;
  char *subckt_name;
  char *subckt_param_names[1000];
  char *subckt_param_values[1000];
  char *inst_param_names[1000];
  char *inst_param_values[1000];
  char name_w_space[1000];
  int  num_subckt_params = 0;
  int  num_inst_params   = 0;
  int i,j,k;
  bool flag = FALSE;
  bool found_subckt = FALSE;
  bool found_param_match = FALSE;

  // first iterate through instances and find occurences where 'm' multiplier needs to be
  // added to the subcircuit -- subsequent instances will then need this parameter as well
  for ( c = deck; c != NULL; c = c->li_next ) {
    inst_line = c->li_line;

    if ( *inst_line == '*' ) { continue; }
    if ( ciprefix( "x", inst_line ) ) {
      num_inst_params = inp_get_params( inst_line,   inst_param_names,   inst_param_values   );
      subckt_name     = inp_get_subckt_name( inst_line );

      if ( found_mult_param( num_inst_params, inst_param_names ) ) {
	flag = FALSE;
	// iterate through the deck to find the subckt (last one defined wins)
	d = deck;
	while ( d != NULL ) {
	  subckt_line = d->li_line;
	  if ( ciprefix( ".subckt", subckt_line ) ) {
	    for ( ; *subckt_line && !isspace(*subckt_line); subckt_line++ );
	    while ( isspace(*subckt_line) ) subckt_line++;

	    sprintf( name_w_space, "%s ", subckt_name );
	    if ( strncmp( subckt_line, name_w_space, strlen(name_w_space) ) == 0 ) {
	      p = d;
	      flag = TRUE;
	    }
	  }
	  d = d->li_next;
	}
        if ( flag ) {
	  num_subckt_params = inp_get_params( p->li_line, subckt_param_names, subckt_param_values );

	  if ( num_subckt_params == 0 || !found_mult_param( num_subckt_params, subckt_param_names ) ) {
	    inp_fix_subckt_multiplier( p, num_subckt_params, subckt_param_names, subckt_param_values );
	  }
	
        }
      }
      tfree(subckt_name );
    }
  }

  c = deck;
  while ( c != NULL ) {
    inst_line = c->li_line;

    if ( *inst_line == '*' ) { c = c->li_next; continue; }
    if ( ciprefix( "x", inst_line ) ) {
      subckt_name = inp_get_subckt_name( inst_line );
      for ( i = 0; i < num_subckt_w_params; i++ ) {
	if ( strcmp( subckt_w_params[i], subckt_name ) == 0 ) {
	  sprintf( name_w_space, "%s ", subckt_name );

	  /* find .subckt line */
	  found_subckt = FALSE;

	  d = deck;
	  while ( d != NULL ) {
	    subckt_line = d->li_line;
	    if ( ciprefix( ".subckt", subckt_line ) ) {
	      for ( ; *subckt_line && !isspace(*subckt_line); subckt_line++ );
	      while ( isspace(*subckt_line) ) subckt_line++;

	      if ( strncmp( subckt_line, name_w_space, strlen(name_w_space) ) == 0 ) {
		num_subckt_params = inp_get_params( subckt_line, subckt_param_names, subckt_param_values );
		num_inst_params   = inp_get_params( inst_line,   inst_param_names,   inst_param_values   );

		// make sure that if have inst params that one matches subckt
		found_param_match = FALSE;
		if ( num_inst_params == 0 ) found_param_match = TRUE;
		else {
		  for ( j = 0; j < num_inst_params; j++ ) {
		    for ( k = 0; k < num_subckt_params; k++ ) {
		      if ( strcmp( subckt_param_names[k], inst_param_names[j] ) == 0 ) {
			found_param_match = TRUE;
			break;
		      }
		    }
		    if ( found_param_match ) break;
		  }
		}

		if ( !found_param_match ) {
		  // comment out .subckt and continue
		  while ( d != NULL && !ciprefix( ".ends", d->li_line ) ) { *(d->li_line) = '*'; d = d->li_next; }
		  *(d->li_line) = '*'; d = d->li_next; continue;
		}

		c->li_line = inp_fix_inst_line( inst_line,
						num_subckt_params, subckt_param_names, subckt_param_values,
						num_inst_params,   inst_param_names,   inst_param_values );
		found_subckt = TRUE;
	      }
	    }
	    if ( found_subckt ) break;
	    d = d->li_next;
	  }
	  break;
	}
      }
      tfree(subckt_name);
    }
    c = c->li_next;
  }
}

static void
inp_get_func_from_line( char *line )
{
  char *ptr, *end;
  char keep;
  char temp_buf[5000];
  int  num_params = 0;
  int  str_len = 0;
  int  i = 0;

  /* get function name */
  while ( !isspace( *line ) ) line++;
  while ( isspace( *line )  ) line++;
  end = line;
  while ( !isspace( *end ) && *end != '(' ) end++;
  keep = *end;
  *end = '\0';
  
  /* see if already encountered this function */
  for ( i = 0; i < num_functions; i++ )
    if ( strcmp( func_names[i], line ) == 0 ) break;
  
  func_names[num_functions++] = strdup(line);
  *end = keep;
  
  num_params = 0;
  
  /* get function parameters */
  while ( *end != '(' ) end++;
  while ( *end != ')' ) {
    end++;
    ptr = end;
    while ( isspace( *ptr ) ) ptr++;
    end = ptr;
    while ( !isspace( *end ) && *end != ',' && *end != ')' ) end++;
    keep = *end;
    *end = '\0';
    func_params[num_functions-1][num_params++] = strdup(ptr);
    *end = keep;
  }
  num_parameters[num_functions-1] = num_params;
  
  /* get function macro */
  str_len = 0;
  while ( *end != '{' ) end++;
  end++;
  while ( *end != '}' ) {
    while ( isspace( *end ) ) end++;
    if ( *end != '}' ) temp_buf[str_len++] = *end;
    end++;
  }
  temp_buf[str_len++] = '\0';
  
  func_macro[num_functions-1] = strdup(temp_buf);
}

//
// only grab global functions; skip subckt functions
//
static void
inp_grab_func( struct line *deck )
{
  struct line *c        = deck;
  bool        is_subckt = FALSE;

  while ( c != NULL ) {
    if ( *c->li_line == '*' ) { c = c->li_next; continue; }
    if ( ciprefix( ".subckt", c->li_line ) ) is_subckt = TRUE;
    if ( ciprefix( ".ends",   c->li_line ) ) is_subckt = FALSE;

    if ( !is_subckt && ciprefix( ".func", c->li_line ) ) {
      inp_get_func_from_line( c->li_line );
      *c->li_line = '*';
    }
    c = c->li_next;
  }
}

static void
inp_grab_subckt_func( struct line *subckt )
{
  struct line *c = subckt;

  while ( !ciprefix( ".ends", c->li_line ) ) {
    if ( ciprefix( ".func", c->li_line ) ) {
      inp_get_func_from_line( c->li_line );
      *c->li_line = '*';
    }
    c = c->li_next;
  }
}

static char*
inp_do_macro_param_replace( int fcn_number, char *params[] )
{
  char *param_ptr, *curr_ptr, *new_str, *curr_str = NULL, *search_ptr;
  char keep, before, after;
  int  i;

  for ( i = 0; i < num_parameters[fcn_number]; i++ ) {
    if ( curr_str == NULL )
      search_ptr = curr_ptr = func_macro[fcn_number];
    else {
      search_ptr = curr_ptr = curr_str;
      curr_str = NULL;
    }
    while ( ( param_ptr = strstr( search_ptr, func_params[fcn_number][i] ) ) ) {

      /* make sure actually have the parameter name */
      before = *(param_ptr-1);
      after  = *(param_ptr+strlen(func_params[fcn_number][i]));
      if ( !(is_arith_char(before) || isspace(before) || before == ',' || before == '=' || (param_ptr-1) < curr_ptr ) ||
	   !(is_arith_char(after)  || isspace(after)  || after  == ',' || after  == '=' || after  == '\0' ) ) {
	search_ptr = param_ptr + 1;
	continue;
      }

      keep       = *param_ptr;
      *param_ptr = '\0';

      if ( curr_str != NULL ) {
	if ( str_has_arith_char( params[i] ) ) {
	  new_str = tmalloc( strlen(curr_str) + strlen(curr_ptr) + strlen(params[i]) + 3 );
	  sprintf( new_str, "%s%s(%s)", curr_str, curr_ptr, params[i] );
	} else {
	  new_str = tmalloc( strlen(curr_str) + strlen(curr_ptr) + strlen(params[i]) + 1 );
	  sprintf( new_str, "%s%s%s", curr_str, curr_ptr, params[i] );
	}

	tfree( curr_str );
      } else {
	if ( str_has_arith_char( params[i] ) ) {
	  new_str = tmalloc( strlen(curr_ptr) + strlen(params[i]) + 3 );
	  sprintf( new_str, "%s(%s)", curr_ptr, params[i] );
	} else {
	  new_str = tmalloc( strlen(curr_ptr) + strlen(params[i]) + 1 );
	  sprintf( new_str, "%s%s", curr_ptr, params[i] );
	}
      }
      curr_str = new_str;

      *param_ptr = keep;
      search_ptr = curr_ptr = param_ptr + strlen(func_params[fcn_number][i]);
    }
    if ( param_ptr == NULL ) {
      if ( curr_str == NULL ) {
	curr_str = curr_ptr;
      } else {
	new_str = tmalloc( strlen(curr_str) + strlen(curr_ptr) + 1 );
	sprintf( new_str, "%s%s", curr_str, curr_ptr );
	tfree(curr_str);
	curr_str = new_str;
      } 
    }
  }
  return curr_str;
}

static char*
inp_expand_macro_in_str( char *str )
{
  int  i;
  char *c;
  char *open_paren_ptr, *close_paren_ptr, *fcn_name, *comma_ptr, *params[1000];
  char *curr_ptr, *new_str, *macro_str, *curr_str = NULL;
  int  num_parens, num_params;
  char *orig_ptr = str, *search_ptr = str, *orig_str = strdup(str);
  char keep;

  while ( ( open_paren_ptr = strstr( search_ptr, "(" ) ) ) {
    fcn_name = open_paren_ptr - 1;
    while ( fcn_name != search_ptr && (isalnum(*fcn_name) || *fcn_name == '_') ) fcn_name--;
    if ( fcn_name != search_ptr ) fcn_name++;

    *open_paren_ptr = '\0';
    close_paren_ptr = NULL;

    if ( open_paren_ptr != fcn_name ) {
      for ( i = 0; i < num_functions; i++ ) {
	if ( strcmp( func_names[i], fcn_name ) == 0 ) {

	  /* find the closing paren */
	  close_paren_ptr = NULL;
	  num_parens      = 0;
	  for ( c = open_paren_ptr + 1; *c && *c != '\0'; c++ ) {
	    if ( *c == '(' ) num_parens++;
	    if ( *c == ')' ) {
	      if ( num_parens != 0 ) num_parens--;
	      else {
		close_paren_ptr = c;
		break;
	      }
	    }
	  }
	  if ( close_paren_ptr == NULL ) {
	    fprintf( stderr, "ERROR: did not find closing parenthesis for function call in str: %s\n", orig_str );
	    exit( -1 );
	  }
	  *close_paren_ptr = '\0';
	  
	  /* get the parameters */
	  curr_ptr   = open_paren_ptr+1;
	  while ( isspace(*curr_ptr) ) curr_ptr++;
	  num_params = 0;
	  if ( ciprefix( "v(", curr_ptr ) ) {
	    // look for any commas and change to ' '
	    char *str_ptr = curr_ptr;
	    while ( *str_ptr != '\0' && *str_ptr != ')' ) { if ( *str_ptr == ',' || *str_ptr == '(' ) *str_ptr = ' '; str_ptr++; }
	    if ( *str_ptr == ')' ) *str_ptr = ' ';
	  }
	  while ( ( comma_ptr = strstr( curr_ptr, "," ) ) ) {
	    while ( isspace(*curr_ptr) ) curr_ptr++;
	    *comma_ptr = '\0';
	    params[num_params++] = inp_expand_macro_in_str( strdup( curr_ptr ) );
	    *comma_ptr = ',';
	    curr_ptr = comma_ptr+1;
	  }
	  while ( isspace(*curr_ptr) ) curr_ptr++;
	  /* get the last parameter */
	  params[num_params++] = inp_expand_macro_in_str( strdup( curr_ptr ) );
	  
	  if ( num_parameters[i] != num_params ) {
	    fprintf( stderr, "ERROR: parameter mismatch for function call in str: %s\n", orig_ptr );
	    exit( -1 );
	  }

	  macro_str = inp_do_macro_param_replace( i, params );
	  keep      = *fcn_name;
	  *fcn_name = '\0';

	  if ( curr_str == NULL ) {
	    new_str = tmalloc( strlen(str) + strlen(macro_str) + strlen(close_paren_ptr+1) + 3 );
	    sprintf( new_str, "%s(%s)", str, macro_str ); 
	    curr_str = new_str;
	  } else {
	    new_str = tmalloc( strlen(curr_str) + strlen(str) + strlen(macro_str) + strlen(close_paren_ptr+1) + 3 );
	    sprintf( new_str, "%s%s(%s)", curr_str, str, macro_str ); 
	    tfree(curr_str);
	    curr_str = new_str;
	  }

	  *fcn_name        = keep;
	  *close_paren_ptr = ')';

	  search_ptr = str = close_paren_ptr+1;
	  break;
	} /* if strcmp */
      } /* for loop over function names */
    } 
    *open_paren_ptr = '(';
    search_ptr = open_paren_ptr + 1;
  }

  if ( curr_str == NULL ) {
    curr_str = orig_ptr;
  }
  else {
    if ( str != NULL ) {
      new_str = tmalloc( strlen(curr_str) + strlen(str) + 1 );
      sprintf( new_str, "%s%s", curr_str, str );
      tfree(curr_str);
      curr_str = new_str;
    }
    tfree(orig_ptr);
  }

  tfree(orig_str);

  return curr_str;
}

static void
inp_expand_macros_in_func()
{
  int  i;

  for ( i = 0; i < num_functions; i++ ) {
    func_macro[i] = inp_expand_macro_in_str( func_macro[i] );
  }
}

static void
inp_expand_macros_in_deck( struct line *deck )
{
  struct line *c = deck;
  int         prev_num_functions = 0, i, j;

  while ( c != NULL ) {
    if ( *c->li_line == '*' ) { c = c->li_next; continue; }
    if ( ciprefix( ".subckt", c->li_line ) ) {
      prev_num_functions = num_functions;
      inp_grab_subckt_func( c );
      if ( prev_num_functions != num_functions ) inp_expand_macros_in_func();
    }
    if ( ciprefix( ".ends", c->li_line ) ) {
      if ( prev_num_functions != num_functions ) {
	for ( i = prev_num_functions; i < num_functions; i++ ) {
	  tfree( func_names[i] );
	  tfree( func_macro[i] );
	  for ( j = 0; j < num_parameters[i]; j++ ) tfree( func_params[i][j] );
	  num_functions = prev_num_functions;
	}
      }
    }

    if ( *c->li_line != '*' ) c->li_line = inp_expand_macro_in_str( c->li_line );
    c = c->li_next;
  }
}

static void
inp_fix_param_values( struct line *deck )
{
  struct line *c = deck;
  char *line, *beg_of_str, *end_of_str, *old_str, *equal_ptr, *new_str;
  char *vec_str, *natok, *buffer, *newvec, *whereisgt;
  bool control_section = FALSE, has_paren = FALSE;
  int n = 0;
  wordlist *wl, *nwl;
  
  while ( c != NULL ) {
    line = c->li_line;

    if ( *line == '*' || (ciprefix( ".param", line ) && strstr( line, "{" )) ) { c = c->li_next; continue; }

    if ( ciprefix( ".control", line ) ) { control_section = TRUE;  c = c->li_next; continue; }
    if ( ciprefix( ".endc", line ) )    { control_section = FALSE; c = c->li_next; continue; }
    if ( control_section || ciprefix( ".option", line ) ) { c = c->li_next; continue; } /* no handling of params in "option" lines */
    if ( ciprefix( "set", line ) ) { c = c->li_next; continue; } /* no handling of params in "set" lines */
    if ( *line == 'b' ) { c = c->li_next; continue; } /* no handling of params in B source lines */
    
    /* for xspice .cmodel: replace .cmodel with .model and skip entire line) */
    if ( ciprefix( ".cmodel", line ) ) { 
    	*(++line) = 'm';
    	*(++line) = 'o';
    	*(++line) = 'd';
    	*(++line) = 'e';
    	*(++line) = 'l';
    	*(++line) = ' ';
    	c = c->li_next; continue; }

    /* exclude CIDER models */
    if ( ciprefix( ".model", line ) && ( strstr(line, "numos") || strstr(line, "numd") || strstr(line, "nbjt") || 
		strstr(line, "nbjt2") || strstr(line, "numd2") ) )
	   { c = c->li_next; continue; }
    /* exclude CIDER devices with ic.file parameter */
    if ( strstr(line, "ic.file")) { c = c->li_next; continue; }

    while ( ( equal_ptr = strstr( line, "=" ) ) ) {

      // skip over equality '=='
      if ( *(equal_ptr+1) == '=' ) { line += 2; continue; }
      // check for '!=', '<=', '>='
      if ( *(equal_ptr-1) == '!' || *(equal_ptr-1) == '<' || *(equal_ptr-1) == '>' ) { line += 1; continue; }

      beg_of_str = equal_ptr + 1;
      while ( isspace(*beg_of_str) ) beg_of_str++;
      /* all cases where no {} have to be put around selected token */
      if ( isdigit(*beg_of_str) || *beg_of_str == '{' || *beg_of_str == '.' ||
         *beg_of_str == '"' || ( *beg_of_str == '-' && isdigit(*(beg_of_str+1)) ) ||
         ciprefix("true", beg_of_str) || ciprefix("false", beg_of_str) ) {
         line = equal_ptr + 1;
      } else if (*beg_of_str == '[') {
      	/* A vector following the '=' token: code to put curly brackets around all params 
      	inside a pair of square brackets */
      	end_of_str = beg_of_str;
      	n = 0;
      	while (*end_of_str != ']') {
      	  end_of_str++;
          n++;
        }     
      	vec_str = tmalloc(n); /* string xx yyy from vector [xx yyy] */
      	*vec_str = '\0';
      	strncat(vec_str, beg_of_str + 1, n - 1);

      	/* work on vector elements inside [] */
        nwl = NULL;
        for (;;) {
           natok = gettok(&vec_str);
           if (!natok) break;
           wl = alloc(struct wordlist);

           buffer = tmalloc(strlen(natok) + 4);  
           if ( isdigit(*natok) || *natok == '{' || *natok == '.' ||
            *natok == '"' || ( *natok == '-' && isdigit(*(natok+1)) ) ||
            ciprefix("true", natok) || ciprefix("false", natok) ||
            eq(natok, "<") || eq(natok, ">")) {
              (void) sprintf(buffer, "%s", natok);
           /* A complex value found inside a vector [< x1 y1> <x2 y2>] */ 
           /* < xx and yy > have been dealt with before */
           /* <xx */             	
           } else if (*natok == '<') {  
              if (isdigit(*(natok + 1)) || ( *(natok + 1) == '-' && 
                isdigit(*(natok + 2)) )) {
                 (void) sprintf(buffer, "%s", natok);
              } else {
              	*natok = '{';
                (void) sprintf(buffer, "<%s}", natok);
              } 
           /* yy> */
           } else if (strchr(natok, '>')) {   
              if (isdigit(*natok) || ( *natok == '-' && isdigit(*(natok+1))) ) {
                 (void) sprintf(buffer, "%s", natok);
              } else {
                 whereisgt = strchr(natok, '>');
                 *whereisgt = '}';
                 (void) sprintf(buffer, "{%s>", natok);
              }
           /* all other tokens */
           } else { 	 
              (void) sprintf(buffer, "{%s}", natok);       	       
           }
           tfree(natok);
           wl->wl_word = copy(buffer);
           tfree(buffer);
           wl->wl_next = nwl;
           if (nwl)
              nwl->wl_prev = wl;
           nwl = wl;   
        }
        nwl = wl_reverse(nwl);
        /* new vector elements */
        newvec = wl_flatten(nwl);
        wl_free(nwl);
        /* insert new vector into actual line */
	*equal_ptr  = '\0';	
	new_str = tmalloc( strlen(c->li_line) + strlen(newvec) + strlen(end_of_str+1) + 5 );
	sprintf( new_str, "%s=[%s] %s", c->li_line, newvec, end_of_str+1 );        
        tfree(newvec);
        
	old_str    = c->li_line;
	c->li_line = new_str;
	line = new_str + strlen(old_str) + 1;
	tfree(old_str);
      } else if (*beg_of_str == '<') {
      	/* A complex value following the '=' token: code to put curly brackets around all params 
      	inside a pair < > */
      	end_of_str = beg_of_str;
      	n = 0;
      	while (*end_of_str != '>') {
      	  end_of_str++;
          n++;
        }     
      	vec_str = tmalloc(n); /* string xx yyy from vector [xx yyy] */
      	*vec_str = '\0';
      	strncat(vec_str, beg_of_str + 1, n - 1);

      	/* work on tokens inside <> */
        nwl = NULL;
        for (;;) {
           natok = gettok(&vec_str);
           if (!natok) break;
           wl = alloc(struct wordlist);

           buffer = tmalloc(strlen(natok) + 4);  
           if ( isdigit(*natok) || *natok == '{' || *natok == '.' ||
            *natok == '"' || ( *natok == '-' && isdigit(*(natok+1)) ) ||
            ciprefix("true", natok) || ciprefix("false", natok)) {
              (void) sprintf(buffer, "%s", natok);
           } else { 	 
              (void) sprintf(buffer, "{%s}", natok);       	       
           }
           tfree(natok);
           wl->wl_word = copy(buffer);
           tfree(buffer);
           wl->wl_next = nwl;
           if (nwl)
              nwl->wl_prev = wl;
           nwl = wl;   
        }
        nwl = wl_reverse(nwl);
        /* new elements of complex variable */
        newvec = wl_flatten(nwl);
        wl_free(nwl);
        /* insert new complex value into actual line */
	*equal_ptr  = '\0';	
	new_str = tmalloc( strlen(c->li_line) + strlen(newvec) + strlen(end_of_str+1) + 5 );
	sprintf( new_str, "%s=<%s> %s", c->li_line, newvec, end_of_str+1 );        
        tfree(newvec);
        
	old_str    = c->li_line;
	c->li_line = new_str;
	line = new_str + strlen(old_str) + 1;
	tfree(old_str);        
      } else {
      /* put {} around token to be accepted as numparam */	
	end_of_str = beg_of_str;
	while ( *end_of_str != '\0' && (!isspace(*end_of_str) || has_paren) ) {
	  if ( *end_of_str == '(' ) has_paren = TRUE; 
	  if ( *end_of_str == ')' ) has_paren = FALSE;
	  end_of_str++;
	}

	*equal_ptr  = '\0';

	if ( *end_of_str == '\0' ) {
	  new_str = tmalloc( strlen(c->li_line) + strlen(beg_of_str) + 4 );
	  sprintf( new_str, "%s={%s}", c->li_line, beg_of_str );

	} else {
	  *end_of_str = '\0';
	
	  new_str = tmalloc( strlen(c->li_line) + strlen(beg_of_str) + strlen(end_of_str+1) + 5 );
	  sprintf( new_str, "%s={%s} %s", c->li_line, beg_of_str, end_of_str+1 );
	}
	old_str    = c->li_line;
	c->li_line = new_str;

	line = new_str + strlen(old_str) + 1;
	tfree(old_str);
      }
    }
    c = c->li_next;
  }
}

static char*
get_param_name( char *line )
{
  char *name, *equal_ptr, *beg;
  char keep;

  if ( ( equal_ptr = strstr( line, "=" ) ) )
    {
      equal_ptr--;
      while ( isspace(*equal_ptr)  ) equal_ptr--;
      equal_ptr++;

      beg = equal_ptr-1;
      while ( !isspace(*beg) && beg != line ) beg--;
      if ( beg != line ) beg++;
      keep       = *equal_ptr;
      *equal_ptr = '\0';
      name       = strdup(beg);
      *equal_ptr = keep;
    }
  else
    {
      fprintf( stderr, "ERROR: could not find '=' on parameter line '%s'!\n", line );
      exit(-1);
    }
  return name;
}

static char*
get_param_str( char *line )
{
  char *equal_ptr;

  if ( ( equal_ptr = strstr( line, "=" ) ) ) {
    equal_ptr++;
    while ( isspace(*equal_ptr) ) equal_ptr++;
    return equal_ptr;
  }
  return line;
}

static int
//inp_get_param_level( int param_num, char *depends_on[12000][100], char *param_names[12000], char *param_strs[12000], int total_params, int *level )
inp_get_param_level( int param_num, char ***depends_on, char **param_names, char **param_strs, int total_params, int *level )
{
  int index1 = 0, comp_level = 0, temp_level = 0;
  int index2 = 0;

  if ( *(level+param_num) != -1 ) return *(level+param_num);

  while ( depends_on[param_num][index1] != NULL )
    {
      index2 = 0;
      while ( index2 <= total_params && param_names[index2] != depends_on[param_num][index1] ) index2++;

      if ( index2 > total_params )
	{
	  fprintf( stderr, "ERROR: unable to find dependency parameter for %s!\n", param_names[param_num] );
	  exit( -1 );
	}
      temp_level = inp_get_param_level( index2, depends_on, param_names, param_strs, total_params, level );
      temp_level++;
      
      if ( temp_level > comp_level ) comp_level = temp_level;
      index1++;
    }
  *(level+param_num) = comp_level;
  return comp_level;
}

static int
get_number_terminals( char *c )
{
  int i, j, k;
  char *name[10];
  char nam_buf[33];
  bool area_found = FALSE;

  switch (*c) {
  case 'r': case 'c': case 'l': case 'k': case 'f': case 'h': case 'b':
  case 'v': case 'i': case 'w': case 'd':
    return 2;
    break;
  case 'u': case 'j': case 'z':
    return 3;
    break;
  case 't': case 'o': case 'g': case 'e': case 's': case 'y':
    return 4;
    break;
  case 'm': /* recognition of 4, 5, 6, or 7 nodes for SOI devices needed */
    i = 0;
    /* find the first token with "off" or "=" in the line*/
    while ( (i < 20) && (*c != '\0') ) {
      char *inst = gettok_instance(&c);
      strncpy(nam_buf, inst, 32);
      txfree(inst);
      if (strstr(nam_buf, "off") || strstr(nam_buf, "=")) break;
      i++;
    }
    return i-2;
    break;
  case 'p': /* recognition of up to 100 cpl nodes */
    i = j = 0;
    /* find the last token in the line*/
    while ( (i < 100) && (*c != '\0') ) {
      strncpy(nam_buf, gettok_instance(&c), 32);
      if (strstr(nam_buf, "=")) j++;      
      i++;
    }
    if (i == 100) return 0;    
    return i-j-2;
    break;    
  case 'q': /* recognition of 3/4 terminal bjt's needed */
    i = j = 0;
    while ( (i < 10) && (*c != '\0') ) {
      name[i] = gettok_instance(&c);
      if (strstr(name[i], "off") || strstr(name[i], "=")) j++;
      i++;
    }
    i--;
    area_found = FALSE;
    for (k = i; k > i-j-1; k--) {
      if (isdigit(*name[k])) area_found = TRUE;
    }
    if (area_found) {
        return i-j-2;
    } else {
        return i-j-1;
    }
    break;
  default:
    return 0;
    break;
  }
}

/* sort parameters based on parameter dependencies */
static void
inp_sort_params( struct line *start_card, struct line *end_card, struct line *card_bf_start, struct line *s_c, struct line *e_c )
{
  char *param_name = NULL, *param_str = NULL, *param_ptr = NULL;
  int  i, j, num_params = 0, ind = 0, max_level = 0, num_terminals = 0;
  bool in_control = FALSE;

  bool found_in_list = FALSE;

  struct line *ptr;
  char *curr_line;
  char *str_ptr, *beg, *end, *new_str;
  int  skipped = 0;
  int arr_size = 12000;

  int *level;
  int *param_skip;
  char **param_names;
  char **param_strs;
  char ***depends_on;
  struct line **ptr_array;
  struct line **ptr_array_ordered;
  
  if ( start_card == NULL ) return;    

  /* determine the number of lines with .param */
  ptr = start_card;
  while ( ptr != NULL )
    {
      if ( strstr( ptr->li_line, "=" ) ) {
	num_params++;
      }
      ptr = ptr->li_next;
    }

  arr_size = num_params;
  num_params = 0; /* This is just to keep the code in row 2907ff. */
  
  // dynamic memory allocation        
  level = (int *) tmalloc(arr_size*sizeof(int));
  param_skip = (int *) tmalloc(arr_size*sizeof(int));
  param_names = (char **) tmalloc(arr_size*sizeof(char*));
  param_strs = (char **) tmalloc(arr_size*sizeof(char*));
  
  /* array[row][column] -> depends_on[array_size][100] */
  /* rows */
  depends_on = (char ***) tmalloc(sizeof(char **) * arr_size);
  /* columns */
   for (i = 0; i < arr_size; i++)
   {
       depends_on[i] = (char **) tmalloc(sizeof(char *) * 100);
   }  

  ptr_array = (struct line **) tmalloc(arr_size*sizeof(struct line *));
  ptr_array_ordered = (struct line **) tmalloc(arr_size*sizeof(struct line *));

  ptr = start_card;
  while ( ptr != NULL )
    {
      // ignore .param lines without '='
      if ( strstr( ptr->li_line, "=" ) ) {
	depends_on[num_params][0] = NULL;
	level[num_params]         = -1;
	param_names[num_params]   = get_param_name( ptr->li_line ); /* strdup in fcn */
	param_strs[num_params]    = strdup( get_param_str( ptr->li_line )  );

	ptr_array[num_params++]   = ptr;
      }
      ptr = ptr->li_next;
    }
  // look for duplicately defined parameters and mark earlier one to skip
  // param list is ordered as defined in netlist
  for ( i = 0; i < num_params; i++ )
      param_skip[i] = 0;
  for ( i = 0; i < num_params; i++ )
    for ( j = num_params-1; j >= 0; j-- )
      {
	if ( i != j && i < j && strcmp( param_names[i], param_names[j] ) == 0 ) {
	  // skip earlier one in list
	  param_skip[i] = 1;
	  skipped++;
	}
      }

  for ( i = 0; i < num_params; i++ )
    {
      if ( param_skip[i] == 1 ) continue;

      param_name = param_names[i];
      for ( j = 0; j < num_params; j++ )
	{
	  if ( j == i ) continue;

	  param_str = param_strs[j];

	  while ( ( param_ptr = strstr( param_str, param_name ) ) )
	    {
	      if ( !isalnum( *(param_ptr-1) )                  && *(param_ptr-1) != '_'     &&
		   !isalnum( *(param_ptr+strlen(param_name)) ) && *(param_ptr+strlen(param_name)) != '_' )
		{
		  ind = 0;
		  found_in_list = FALSE;
		  while ( depends_on[j][ind] != NULL ) {
		    if ( strcmp( param_name, depends_on[j][ind] ) == 0 ) { found_in_list = TRUE; break; }
		    ind++;
		  }
		  if ( !found_in_list ) {
		    depends_on[j][ind++] = param_name;
		    depends_on[j][ind]   = NULL;
		  }
		  break;
		}
	      param_str = param_ptr + strlen(param_name);
	    }
	}
    }

  for ( i = 0; i < num_params; i++ )
    {
      level[i] = inp_get_param_level( i, depends_on, param_names, param_strs, num_params, level );
      if ( level[i] > max_level ) max_level = level[i];
    }

  /* look for unquoted parameters and quote them */
  ptr = s_c;
  in_control = FALSE;
  while ( ptr != NULL && ptr != e_c )
    {
      curr_line = ptr->li_line;

      if ( ciprefix( ".control", curr_line ) ) { in_control = TRUE; ptr = ptr->li_next; continue; }
      if ( ciprefix( ".endc", curr_line ) ) { in_control = FALSE; ptr = ptr->li_next; continue; }
      if ( in_control || curr_line[0] == '.' || curr_line[0] == '*' ) { ptr = ptr->li_next; continue; }

      num_terminals = get_number_terminals( curr_line );

      if ( num_terminals <= 0 ) { ptr = ptr->li_next; continue; }

      for ( i = 0; i < num_params; i++ )
	{
	  str_ptr = curr_line;
	  
	  for ( j = 0; j < num_terminals+1; j++ )
	    {
	      while ( !isspace(*str_ptr) && *str_ptr != '\0' ) str_ptr++;
	      while ( isspace(*str_ptr)  && *str_ptr != '\0' ) str_ptr++;
	    }

	  while ( ( str_ptr = strstr( str_ptr, param_names[i] ) ) )
	    {
	      /* make sure actually have the parameter name */
	      char before = *(str_ptr-1);
	      char after  = *(str_ptr+strlen(param_names[i]));
	      if ( !( is_arith_char(before) || isspace(before) || (str_ptr-1) < curr_line ) ||
		   !( is_arith_char(after)  || isspace(after)  || after == '\0' ) ) {
		str_ptr = str_ptr + 1;
		continue;
	      }
	      beg = str_ptr - 1;
	      end = str_ptr + strlen(param_names[i]);
	      if ( ( isspace(*beg) || *beg == '=' ) &&
		   ( isspace(*end) || *end == '\0' ) )
		{
		  *str_ptr = '\0';
		  if ( *end != '\0' )
		    {
		      new_str = tmalloc( strlen(curr_line) + strlen(param_names[i]) + strlen(end) + 3 );
		      sprintf( new_str, "%s{%s}%s", curr_line, param_names[i], end );
		    }
		  else
		    {
		      new_str = tmalloc( strlen(curr_line) + strlen(param_names[i]) + 3 );
		      sprintf( new_str, "%s{%s}", curr_line, param_names[i] );
		    }
		  str_ptr = new_str + strlen(curr_line) + strlen(param_names[i]);

		  tfree(ptr->li_line);
		  curr_line = ptr->li_line = new_str;
		}
	      str_ptr++;
	    }
	}
      ptr = ptr->li_next;
    }

  ind = 0;
  for ( i = 0; i <= max_level; i++ )
    for ( j = num_params-1; j >= 0; j-- )
      {
	if ( level[j] == i ) {
	  if ( param_skip[j] == 0 ) {
	    ptr_array_ordered[ind++] = ptr_array[j];
	  }
	}
      }

  num_params -= skipped;
  if ( ind != num_params )
    {
      fprintf( stderr, "ERROR: found wrong number of parameters during levelization ( %d instead of %d parameter s)!\n", ind, num_params );
      exit(-1);
    }

  /* fix next ptrs */
  ptr                                      = card_bf_start->li_next;
  card_bf_start->li_next                   = ptr_array_ordered[0];
  ptr_array_ordered[num_params-1]->li_next = ptr;
  for ( i = 0; i < num_params-1; i++ )
    ptr_array_ordered[i]->li_next = ptr_array_ordered[i+1];

  // clean up memory
  for ( i = 0; i < num_params; i++ )
    {
      tfree( param_names[i] );
      tfree( param_strs[i] );
    }
    
  tfree(level);
  tfree(param_skip);
  tfree(param_names);
  tfree(param_strs);  

   for (i = 0; i< arr_size; i++)
       tfree(depends_on[i]);
   tfree(depends_on);  
   
   tfree(ptr_array);  
   tfree(ptr_array_ordered);   
  
}

static void
inp_add_params_to_subckt( struct line *subckt_card )
{
  struct line *card        = subckt_card->li_next;
  char        *curr_line   = card->li_line;
  char        *subckt_line = subckt_card->li_line;
  char        *new_line, *param_ptr, *subckt_name, *end_ptr;
  char        keep;

  while ( card != NULL && ciprefix( ".param", curr_line ) ) {
    param_ptr = strstr( curr_line, " " );
    while ( isspace(*param_ptr) ) param_ptr++;

    if ( !strstr( subckt_line, "params:" ) ) {
      new_line = tmalloc( strlen(subckt_line) + strlen("params: ") + strlen(param_ptr) + 2 );
      sprintf( new_line, "%s params: %s", subckt_line, param_ptr );

      subckt_name = subckt_card->li_line;
      while ( !isspace(*subckt_name) ) subckt_name++;
      while ( isspace(*subckt_name) )  subckt_name++;
      end_ptr = subckt_name;
      while ( !isspace(*end_ptr) ) end_ptr++;
      keep     = *end_ptr;
      *end_ptr = '\0';
      subckt_w_params[num_subckt_w_params++] = strdup(subckt_name);
      *end_ptr = keep;
    } else {
      new_line = tmalloc( strlen(subckt_line) + strlen(param_ptr) + 2 );
      sprintf( new_line, "%s %s", subckt_line, param_ptr );
    }

    tfree( subckt_line );
    subckt_card->li_line = subckt_line = new_line;

    *curr_line = '*';

    card      = card->li_next;
    curr_line = card->li_line;
  }
}

static void
inp_reorder_params( struct line *deck, struct line *list_head, struct line *end )
{
  struct line *c = deck, *subckt_card = NULL, *param_card = NULL, *prev_card = list_head;
  struct line *subckt_param_card = NULL, *first_param_card = NULL, *first_subckt_param_card = NULL;
  char *curr_line;
  bool processing_subckt = FALSE;
  
  /* move .param lines to beginning of deck */
  while ( c != NULL ) {
    curr_line = c->li_line;
    if ( *curr_line == '*' ) { c = c->li_next; continue; }
    if ( ciprefix( ".subckt", curr_line ) ) {
      processing_subckt       = TRUE;
      subckt_card             = c;
      first_subckt_param_card = NULL;
    }
    if ( ciprefix( ".ends", curr_line ) && processing_subckt ) {
      processing_subckt          = FALSE;
      if ( first_subckt_param_card != NULL ) {
	inp_sort_params( first_subckt_param_card, subckt_param_card, subckt_card, subckt_card, c );
	inp_add_params_to_subckt( subckt_card );
      }
    }

    if ( ciprefix( ".param", curr_line ) ) {
      if ( !processing_subckt ) {
	if ( first_param_card == NULL ) {
	  first_param_card    = c;
	} else {
	  param_card->li_next = c;
	}
	param_card          = c;
	prev_card->li_next  = c->li_next;
	param_card->li_next = NULL;
	c                   = prev_card;
      } else {
	if ( first_subckt_param_card == NULL ) {
	  first_subckt_param_card    = c;
	} else {
	  subckt_param_card->li_next = c;
	}
	subckt_param_card          = c;
	prev_card->li_next         = c->li_next;
	c                          = prev_card;
	subckt_param_card->li_next = NULL;
      }
    }
    prev_card = c;
    c         = c->li_next;
  }
  inp_sort_params( first_param_card, param_card, list_head, deck, end );
}

// iterate through deck and find lines with multiply defined parameters
//
// split line up into multiple lines and place those new lines immediately
// afetr the current multi-param line in the deck
static int
inp_split_multi_param_lines( struct line *deck, int line_num )
{
  struct line *card = deck, *param_end = NULL, *param_beg = NULL, *prev = NULL, *tmp_ptr;
  char *curr_line, *equal_ptr, *beg_param, *end_param, *new_line;
  char *array[5000];
  int  counter = 0, i;
  bool get_expression = FALSE, get_paren_expression = FALSE;
  char keep;

  while ( card != NULL )
    {
      curr_line = card->li_line;

      if ( *curr_line == '*' ) { card = card->li_next; continue; }

      if ( ciprefix( ".param", curr_line ) )
	{
	  counter = 0;
	  while ( ( equal_ptr = strstr( curr_line, "=" ) ) ) {
	    // check for equality '=='
	    if ( *(equal_ptr+1) == '=' ) { curr_line = equal_ptr+2; continue; }
	    // check for '!=', '<=', '>='
	    if ( *(equal_ptr-1) == '!' || *(equal_ptr-1) == '<' || *(equal_ptr-1) == '>' ) { curr_line = equal_ptr+1; continue; }
	    counter++; curr_line = equal_ptr + 1;
	  }
	  if ( counter <= 1 ) { card = card->li_next; continue; }

	  // need to split multi param line
	  curr_line = card->li_line;
	  counter   = 0;
	  while ( curr_line < card->li_line+strlen(card->li_line) && ( equal_ptr = strstr( curr_line, "=" ) ) )
	    {
	      // check for equality '=='
	      if ( *(equal_ptr+1) == '=' ) { curr_line = equal_ptr+2; continue; }
	      // check for '!=', '<=', '>='
	      if ( *(equal_ptr-1) == '!' || *(equal_ptr-1) == '<' || *(equal_ptr-1) == '>' ) { curr_line = equal_ptr+1; continue; }

	      beg_param = equal_ptr - 1;
	      end_param = equal_ptr + 1;
	      while ( isspace(*beg_param)  ) beg_param--;
	      while ( !isspace(*beg_param) ) beg_param--;
	      while ( isspace(*end_param) ) end_param++;
	      while ( *end_param != '\0' && (!isspace(*end_param) || get_expression || get_paren_expression) ) {
		if ( *end_param == '{' ) get_expression       = TRUE;
		if ( *end_param == '(' ) get_paren_expression = TRUE;
		if ( *end_param == '}' ) get_expression       = FALSE;
		if ( *end_param == ')' ) get_paren_expression = FALSE;
		end_param++;
	      }
	      beg_param++;
	      keep       = *end_param;
	      *end_param = '\0';
	      new_line   = tmalloc( strlen(".param ") + strlen(beg_param) + 1 );
	      sprintf( new_line, ".param %s", beg_param );
	      array[counter++] = new_line;
	      *end_param = keep;
	      curr_line = end_param;
	    }
	  tmp_ptr = card->li_next;
	  for ( i = 0; i < counter; i++ )
	    {
	      if ( param_end )
		{
		  param_end->li_next = alloc(struct line);
		  param_end          = param_end->li_next;
		}
	      else
		{
		  param_end = param_beg = alloc(struct line);
		}
	      param_end->li_next    = NULL;
	      param_end->li_error   = NULL;
	      param_end->li_actual  = NULL;
	      param_end->li_line    = array[i];
	      param_end->li_linenum = line_num++;
	    }
	  // comment out current multi-param line
	  *(card->li_line)   = '*';
	  // insert new param lines immediately after current line
	  tmp_ptr            = card->li_next;
	  card->li_next      = param_beg;
	  param_end->li_next = tmp_ptr;
	  // point 'card' pointer to last in scalar list
	  card               = param_end;

	  param_beg = param_end = NULL;
	} // if ( ciprefix( ".param", curr_line ) )
      prev = card;
      card = card->li_next;
    } // while ( card != NULL )
  if ( param_end )
    {
      prev->li_next = param_beg;
      prev          = param_end;
    }
  return line_num;
}

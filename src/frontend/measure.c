/* Routines to evaluate the .measure cards.
   Entry point is function do_measure(), called by fcn dosim() 
   from runcoms.c:335, after simulation is finished.
   
   In addition it contains the fcn com_meas(), which provide the 
   interactive 'meas' command.
*/
   
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"

#include "rawfile.h"
#include "variable.h"
#include "numparam/numpaif.h"
#include "ngspice/missing_math.h"
#include "com_measure2.h"
#include "com_let.h"
#include "com_commands.h"
#include "com_display.h"

static wordlist *measure_parse_line( char *line ) ;

static bool measure_valid[20000];/* TRUE: if measurement no. [xxx] has been done successfully
                                 (not used anywhere)*/
static bool just_chk_meas;   /* TRUE: only check if measurement can be done successfully,
                             no output generated (if option autostop is set)*/
static bool measures_passed; /* TRUE: stop simulation (if option autostop is set)*/

extern bool ft_batchmode;
extern bool rflag;

/* measure in interactive mode:
   meas command inside .control ... .endc loop or manually entered.
   meas has to be followed by the standard tokens (see measure_extract_variables()).
   The result is put into a vector with name "result"
*/

void
com_meas(wordlist *wl) {
   /* wl: in, input line of meas command */
   char *line_in, *outvar, newvec[1000];
   wordlist * wl_count, *wl_let;

   char *vec_found, *token, *equal_ptr, newval[256];
   wordlist *wl_index;
   struct dvec *d;
   int err=0;

   int fail;
   double result = 0;

   if (!wl) {
      com_display(NULL);
      return;
   }
   wl_count = wl;

   /* check each wl entry, if it contain '=' and if the following token is
      a single valued vector. If yes, replace this vector by its value.
      Vectors may stem from other meas commands, or be generated elsewhere 
      within the .control .endc script. All other right hand side vectors are
      treated in com_measure2.c. */
   wl_index = wl;

   while ( wl_index) {
	   token = wl_index->wl_word;
      /* find the vector vec_found, next token after each '=' sign. 
	     May be in the next wl_word */
      if ( *(token + strlen(token) - 1) == '=' ) {
         wl_index = wl_index->wl_next;
         vec_found = wl_index->wl_word;
         /* token may be already a value, maybe 'LAST', which we have to keep, or maybe a vector */
         if (!cieq(vec_found, "LAST")) {
            INPevaluate( &vec_found, &err, 1 );
            /* if not a valid number */
            if (err) {         
               /* check if vec_found is a valid vector */	
               d = vec_get(vec_found);
               /* Only if we have a single valued vector, replacing
               of the rigt hand side does make sense */
               if (d && (d->v_length == 1) && (d->v_numdims == 1)) {
                  /* get its value */
                  sprintf(newval, "%e", d->v_realdata[0]);
                  tfree(vec_found);
                  wl_index->wl_word = copy(newval);
               }
            }
         } 
      } 
	   /* may be inside the same wl_word */
      else if ( (equal_ptr = strstr( token, "=" )) != NULL ) {
         vec_found = equal_ptr + 1;
         if (!cieq(vec_found, "LAST")) {
            INPevaluate( &vec_found, &err, 1 );
            if (err) {
               d = vec_get(vec_found);
               /* Only if we have a single valued vector, replacing
               of the rigt hand side does make sense */
               if (d && (d->v_length == 1) && (d->v_numdims == 1)) {
                  *equal_ptr = '\0';
                  sprintf(newval, "%s=%e", token, d->v_realdata[0]);
//               memory leak with first part of vec_found ?
                  tfree(token);
                  wl_index->wl_word = copy(newval);
               }
            }
         } 
      } else {
         ;// nothing
      }
      wl_index = wl_index->wl_next;      
   }	  

   line_in = wl_flatten(wl);

   /* get output var name */
   wl_count = wl_count->wl_next;
   outvar = wl_count->wl_word;

   fail = get_measure2(wl, &result, NULL, FALSE) ;

   if (fail) {
      fprintf(stdout, " meas %s failed!\n\n", line_in);
      return;
   }

   sprintf(newvec, "%s = %e", outvar, result);
   wl_let = wl_cons(copy(newvec), NULL);
   com_let(wl_let);
   wl_free(wl_let);
//   fprintf(stdout, "in: %s\n", line_in);
}


static bool
chkAnalysisType( char *an_type ) {
  /* only support tran, dc, ac, sp analysis type for now */
  if ( strcmp( an_type, "tran" ) != 0 && strcmp( an_type, "ac" ) != 0 &&
	  strcmp( an_type, "dc"   ) != 0 && strcmp( an_type, "sp"   ) != 0)
    return FALSE;
//  else if (ft_batchmode == TRUE) return FALSE;
  else return TRUE;
}


/* Gets pointer to double value after 'xxx=' and advances pointer of *line. 
   On error returns FALSE. */ 
static bool
get_double_value( 
   char **line,  /*in|out: pointer to line to be parsed */
   char *name,   /*in: xxx e.g. 'val' from 'val=0.5' */
   double *value /*out: return value (e.g. 0.5) from 'val=0.5'*/
) {
  char *token     = gettok(line);
  bool return_val = TRUE;
  char *equal_ptr, *junk;
  int  err=0;

  if ( name && ( strncmp( token, name, strlen(name) ) != 0 ) ) {
    if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: syntax error for measure statement; expecting next field to be '%s'.\n", name );
    return_val = FALSE;
  } else {
    /* see if '=' is last char of current token -- implies we need to read value in next token */
    if ( *(token + strlen(token) - 1) == '=' ) {
      txfree(token);
      junk = token = gettok(line);

      *value = INPevaluate( &junk, &err, 1 );
    } else {
      if ( (equal_ptr = strstr( token, "=" )) != NULL ) {
	equal_ptr += 1;
	*value = INPevaluate( &equal_ptr, &err, 1 );
      } else {
	if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: syntax error for measure statement; missing '='!\n" );
	return_val = FALSE;
      }
    }
    if ( err ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: Bad value.\n" ); return_val = FALSE; }
  } 
  txfree(token);

  return return_val;
}


/* Entry point for .meas evaluation. 
   Called in fcn dosim() from runcoms.c:335, after simulation is finished
   with chk_only set to FALSE.
   Called from fcn check_autostop()
   with chk_only set to TRUE (no printouts, no params set). */
void
do_measure( 
  char *what,   /*in: analysis type*/
  bool chk_only /*in: TRUE if checking for "autostop", FALSE otherwise*/
                /*global variable measures_passed
                  out: set to FALSE if .meas syntax is violated (used with autostop)*/ 
) {
  struct line *meas_card, *meas_results = NULL, *end = NULL, *newcard;
  char        *line, *an_name, *an_type, *resname, *meastype, *str_ptr, out_line[1000];
  int         idx  = 0, ok = 0;
  int	      fail;
  double      result = 0;
  bool        first_time = TRUE;
  wordlist    *measure_word_list ;
  int         precision = measure_get_precision() ;

  just_chk_meas = chk_only; 

  an_name = strdup( what ); /* analysis type, e.g. "tran" */
  strtolower( an_name );
  measure_word_list = NULL ;

  /* don't allow .meas if batchmode is set by -b and -r rawfile given */
  if (ft_batchmode && rflag) {
     fprintf(cp_err, "\nNo .measure possible in batch mode (-b) with -r rawfile set!\n");
     fprintf(cp_err, "Remove rawfile and use .print or .plot or\n");
     fprintf(cp_err, "select interactive mode (optionally with .control section) instead.\n\n");
     return;
  }

  /* Evaluating the linked list of .meas cards, assembled from the input deck
     by fcn inp_spsource() in inp.c:575.
     A typical .meas card will contain:
     parameter        value
     nameof card      .meas(ure) 
     analysis type    tran        only tran available currently
     result name      myout       defined by user
     measurement type trig|delay|param|expr|avg|mean|max|min|rms|integ(ral)|when

     The measurement type determines how to continue the .meas card. 
     param|expr are skipped in first pass through .meas cards and are treated in second pass, 
     all others are treated in fcn get_measure2() (com_measure2.c).
     */
  
  /* first pass through .meas cards: evaluate everything except param|expr */
  for ( meas_card = ft_curckt->ci_meas; meas_card != NULL; meas_card = meas_card->li_next ) {
    line = meas_card->li_line;

    txfree(gettok(&line)); /* discard .meas */

    an_type = gettok(&line); resname = gettok(&line); meastype = gettok(&line);

    if ( chkAnalysisType( an_type ) != TRUE ) {
      if ( just_chk_meas != TRUE ) {
        fprintf( cp_err, "Error: unrecognized analysis type '%s' for the following .meas statement on line %d:\n", an_type, meas_card->li_linenum );
        fprintf( cp_err, "       %s\n", meas_card->li_line );
      }

      txfree(an_type); txfree(resname); txfree(meastype);
      continue;
    }
    /* print header before evaluating first .meas line */
    else if ( first_time ) {
      first_time = FALSE;

      if ( just_chk_meas != TRUE && strcmp( an_type, "tran" ) == 0 ) {
         fprintf( stdout, "             Transient Analysis\n\n" );
//         plot_cur = setcplot("tran");
      }
    }

    /* skip param|expr measurement types for now -- will be done after other measurements */
    if ( strncmp( meastype, "param", 5 ) == 0 || strncmp( meastype, "expr", 4 ) == 0 ) continue;

    /* skip .meas line, if analysis type from line and name of analysis performed differ */
    if ( strcmp( an_name, an_type ) != 0 ) {
      txfree(an_type); txfree(resname); txfree(meastype);
      continue; 
    }

    /* New way of processing measure statements using common code 
       in fcn get_measure2() (com_measure2.c)*/
    out_line[0] = '\0' ;
    measure_word_list = measure_parse_line( meas_card->li_line) ;
    if( measure_word_list ){
      fail = get_measure2(measure_word_list,&result,out_line,chk_only) ;
      if( fail ){
	measure_valid[idx++] = FALSE;
	measures_passed = FALSE;
      fprintf(stdout, " %s failed!\n\n", meas_card->li_line);
      } else {
	if(!(just_chk_meas)){
	  nupa_add_param( resname, result );
	}
	measure_valid[idx++] = TRUE;
      }
      wl_free( measure_word_list ) ;
    } else {
      measure_valid[idx++] = FALSE;
      measures_passed = FALSE;
    }

    newcard          = alloc(struct line);
    newcard->li_line = strdup(out_line);
    newcard->li_next = NULL;

    if ( meas_results == NULL ) meas_results = end = newcard;
    else {
      end->li_next = newcard;
      end          = newcard;
    }

    txfree(an_type); txfree(resname); txfree(meastype);

    /* see if number of measurements exceeds fixed array size of 20,000 */
    if ( idx >= 20000 ) {
      fprintf( stderr, "ERROR: number of measurements exceeds 20,000!\nAborting...\n" );
      controlled_exit(EXIT_FAILURE);
    }

  } /* end of for loop (first pass through .meas lines) */


  /* second pass through .meas cards: now do param|expr .meas statements */
  newcard = meas_results;
  for ( meas_card = ft_curckt->ci_meas; meas_card != NULL; meas_card = meas_card->li_next ) {
    line = meas_card->li_line;

    txfree(gettok(&line)); /* discard .meas */

    an_type = gettok(&line); resname = gettok(&line); meastype = gettok(&line);

    if ( chkAnalysisType( an_type ) != TRUE ) {
      if ( just_chk_meas != TRUE ) {
        fprintf( cp_err, "Error: unrecognized analysis type '%s' for the following .meas statement on line %d:\n", an_type, meas_card->li_linenum );
        fprintf( cp_err, "       %s\n", meas_card->li_line );
      }

      txfree(an_type); txfree(resname); txfree(meastype);
      continue;
    }
    if ( strcmp( an_name, an_type ) != 0 ) {
      txfree(an_type); txfree(resname); txfree(meastype);
      continue;
    }

    if ( strncmp( meastype, "param", 5 ) != 0 && strncmp( meastype, "expr", 4 ) != 0 ) {

      if ( just_chk_meas != TRUE ) fprintf( stdout, "%s", newcard->li_line );
      end     = newcard;
      newcard = newcard->li_next;

      txfree( end->li_line );
      txfree( end );

      txfree(an_type); txfree(resname); txfree(meastype);
      continue;
    }

    if ( just_chk_meas != TRUE ) fprintf( stdout, "%-20s=", resname );

    if ( just_chk_meas != TRUE ) {
      ok = nupa_eval( meas_card->li_line, meas_card->li_linenum, meas_card->li_linenum_orig );

      if ( ok ) {
        str_ptr = strstr( meas_card->li_line, meastype );
        if ( !get_double_value( &str_ptr, meastype, &result ) ) {
          if ( just_chk_meas != TRUE ) fprintf( stdout, "   failed\n"       );
        }
        else {
          if ( just_chk_meas != TRUE ) fprintf( stdout, "  %.*e\n", precision, result );
          nupa_add_param( resname, result );
        }
      }
      else {
        if ( just_chk_meas != TRUE ) fprintf( stdout, "   failed\n" );
      }
    }
    txfree(an_type); txfree(resname); txfree(meastype);
  }

  if ( just_chk_meas != TRUE ) fprintf( stdout, "\n" );

  txfree(an_name);

  fflush( stdout );

  //nupa_list_params();
}


/* called from dctran.c:470, if timepoint is accepted.
   Returns TRUE if measurement (just a check, no output) has been successful. 
   If TRUE is returned, transient simulation is stopped.
   Returns TRUE if "autostop" has been set as an option and if measures_passed not
   set to FALSE during calling do_measure. 'what' is set to "tran".*/

bool
check_autostop( char* what ) {
  bool flag = FALSE;

  measures_passed = TRUE;
  if ( cp_getvar( "autostop", CP_BOOL, NULL) ) {
    do_measure( what, TRUE );

    if ( measures_passed == TRUE ) flag = TRUE;
  }

  return flag;
}

/* parses the .meas line into a wordlist (without leading .meas) */
static wordlist *measure_parse_line( char *line )
 {
   size_t len ;				/* length of string */
   wordlist *wl ;			/* build a word list - head of list */
   wordlist *new_item ;			/* single item of a list */
   char *item ;				/* parsed item */
   char *long_str ;			/* concatenated string */
   char *extra_item ;			/* extra item */
 
   wl = NULL ;
   (void) gettok(&line) ;
   do { 
     item = gettok(&line) ;
     if(!(item)){
       break ;
     }
     len = strlen(item) ;
     if( item[len-1] == '=' ){
       /* We can't end on an equal append the next piece */
       extra_item = gettok(&line) ;
       if(!(extra_item)){
 	break ;
       }
       len += strlen( extra_item ) + 2 ;
       long_str = TMALLOC(char, len) ;
       sprintf( long_str, "%s%s", item, extra_item ) ;
       txfree( item ) ;
       txfree( extra_item ) ;
       item = long_str ;
     }
     new_item = wl_cons(item, NULL);
     wl = wl_append(wl, new_item) ;
   } while( line && *line ) ;
 
   return(wl) ;
 
} /* end measure_parse_line() */

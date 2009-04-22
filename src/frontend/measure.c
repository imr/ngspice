#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"

#include "rawfile.h"
#include "variable.h"
#include "numparam/numpaif.h"
#include "missing_math.h"

static bool measure_valid[20000];
static bool just_chk_meas;
static bool measures_passed;


static int
get_measure_precision()
{
  char *env_ptr;
  int  precision = 5;
  
  if ( ( env_ptr = getenv("NGSPICE_MEAS_PRECISION") ) ) {
    precision = atoi(env_ptr);
  }

  return precision;
}

static double
interpolate( struct dvec *time, struct dvec *values, int i, int j, double var_value, char x_or_y ) {
  double slope = (values->v_realdata[j] - values->v_realdata[i])/(time->v_realdata[j] - time->v_realdata[i]);
  double yint  = values->v_realdata[i] - slope*time->v_realdata[i];
  double result;
  
  if ( x_or_y == 'x' ) result = (var_value - yint)/slope;
  else                 result = slope*var_value + yint;

  return result;
}

static double
get_volt_time( struct dvec *time, struct dvec *values, double value, char polarity, int index, bool *failed )
{
  int    i = 0, count = 0;
  double comp_time = 0;

  for ( i = 0; i < values->v_length-1; i++ ) {
    if ( polarity == 'r' ) {
      if ( values->v_realdata[i] < value && value <= values->v_realdata[i+1] ) {
	count++;
	if ( count == index ) comp_time = interpolate( time, values, i, i+1, value, 'x' );
      }
    }
    else if ( polarity == 'f' ) {
      if ( values->v_realdata[i] >= value && value > values->v_realdata[i+1] ) {
	count++;
	if ( count == index ) comp_time = interpolate( time, values, i, i+1, value, 'x' );
      }
    }
    else {
      if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: unknown signal polarity '%c'; valid types are 'r' or 'f'.\n", polarity );
      *failed = TRUE;
    }
  }
  if ( AlmostEqualUlps( comp_time, 0, 100 ) ) *failed = TRUE;

  return comp_time;
}

static bool
measure( char *trig_name, double trig_value, char trig_polarity, int trig_index,
  char *targ_name, double targ_value, char targ_polarity, int targ_index, double *result,
  double *trig_time, double *targ_time ) {
  struct dvec *time  = vec_get("time");
  struct dvec *trig  = vec_get(trig_name);
  struct dvec *targ  = vec_get(targ_name);
  bool        failed = FALSE;

  if ( !time ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: problem accessing vector 'time'!\n"          ); return TRUE; }
  if ( !trig ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: problem accessing vector '%s'!\n", trig_name ); return TRUE; }
  if ( !targ ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: problem accessing vector '%s'!\n", targ_name ); return TRUE; }

  *trig_time = get_volt_time( time, trig, trig_value, trig_polarity, trig_index, &failed );
  *targ_time = get_volt_time( time, targ, targ_value, targ_polarity, targ_index, &failed );
  *result    = *targ_time - *trig_time;

  return failed;
}

/*
  avg: (average) calculates the area under the out_var divided by the periods of interest
  rms: (root mean squared) calculates the square root of the area under the out_var^2 curve 
       divided by the period of interest
  integral: calculate the integral
*/
static bool
measure2( char *meas_type, char *vec_name, char vec_type, double from, double to, double *result, double *result_time ) {
   struct dvec *time  = vec_get("time");
   struct dvec *vec;
   int         xy_size = 0;
   double      *x, *y, *width, sum1 = 0, sum2 = 0, sum3 = 0;
   double      init_val;
   char        tmp_vec_name[1000];
   double      prev_result = 0;
   bool        failed = FALSE, first_time = TRUE, constant_y = TRUE;
   int         i, idx, upflag ;

   if ( to < from ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: (measure2) 'to' time (%e) < 'from' time (%e).\n", to, from ); return TRUE; }

   if ( vec_type == 'i' ) {
      if ( strstr( vec_name, ".v" ) ) sprintf( tmp_vec_name, "v.%s#branch", vec_name );
      else                            sprintf( tmp_vec_name, "%s#branch",   vec_name );
   }
   else sprintf( tmp_vec_name, "%s", vec_name );

   vec = vec_get( tmp_vec_name );

   if ( !time ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: problem accessing vector 'time'!\n"             ); return TRUE; }
   if ( !vec  ) { if ( just_chk_meas != TRUE ) fprintf( stderr, "Error: problem accessing vector '%s'!\n", tmp_vec_name ); return TRUE; }

   if ( strcmp( meas_type, "max" ) == 0 || strcmp( meas_type, "min" ) == 0 ) {
      for ( i = 0; i < vec->v_length; i++ ) {
         if ( time->v_realdata[i] >= from && ( i+1 < time->v_length && time->v_realdata[i+1] <= to ) ) {
            prev_result = *result;
            if ( first_time ) {
               first_time   = FALSE;
               *result      = vec->v_realdata[i];
               *result_time = time->v_realdata[i];
            } else {
               *result = ( strcmp( meas_type, "max" ) == 0 ) ? MAX( *result, vec->v_realdata[i] ) : MIN( *result, vec->v_realdata[i] );
               if ( !AlmostEqualUlps( prev_result, *result, 100 ) ) *result_time = time->v_realdata[i];
            }
         }
      }
   }
   else if ( strcmp( meas_type, "avg"      ) == 0 || strcmp( meas_type, "rms"   ) == 0 ||
	    strcmp( meas_type, "integral" ) == 0 || strcmp( meas_type, "integ" ) == 0 ) {
      x     = (double *) tmalloc(time->v_length * sizeof(double));
      y     = (double *) tmalloc(time->v_length * sizeof(double));
      width = (double *) tmalloc(time->v_length * sizeof(double));

      // create new set of values over interval [from, to] -- interpolate if necessary
      for ( i = 0; i < vec->v_length; i++ ) {
         if ( time->v_realdata[i] >= from && time->v_realdata[i] <= to ) {
            *(x+xy_size)   = time->v_realdata[i];
            *(y+xy_size++) = ( strcmp( meas_type, "avg" ) == 0 || ciprefix( "integ", meas_type ) ) ? vec->v_realdata[i] : pow(vec->v_realdata[i],2);
         }
      }
      // evaluate segment width
      for ( i = 0; i < xy_size-1; i++ ) *(width+i) = *(x+i+1) - *(x+i);
      *(width+i++) = 0;
      *(width+i++) = 0;

      // see if y-value constant
      for ( i = 0; i < xy_size-1; i++ )
         if ( !AlmostEqualUlps( *(y+i), *(y+i+1), 100 ) ) constant_y = FALSE;

      // Compute Integral (area under curve)
      i = 0;
      while ( i < xy_size-1 ) {
         // Simpson's 3/8 Rule
         if ( AlmostEqualUlps( *(width+i), *(width+i+1), 100 ) && AlmostEqualUlps( *(width+i), *(width+i+2), 100 ) ) {
            sum1 += 3*(*(width+i))*(*(y+i) + 3*(*(y+i+1) + *(y+i+2)) + *(y+i+3))/8;
            i += 3;
         }
         // Simpson's 1/3 Rule
         else if ( AlmostEqualUlps( *(width+i), *(width+i+1), 100 ) ) {
            sum2 += *(width+i)*(*(y+i) + 4*(*(y+i+1)) + *(y+i+2))/3;
            i += 2;
         }
         // Trapezoidal Rule
         else if ( !AlmostEqualUlps( *(width+i), *(width+i+1), 100 ) ) {
            sum3 += *(width+i)*(*(y+i) + *(y+i+1))/2;
            i++;
         }
      }

      if ( !ciprefix( "integ", meas_type ) ) {
         *result = (sum1 + sum2 + sum3)/(to - from);

         if ( strcmp( meas_type, "rms" ) == 0 ) *result = sqrt(*result);
         if ( strcmp( meas_type, "avg" ) == 0 && constant_y == TRUE ) *result = *y;
      }
      else {
         *result = ( sum1 + sum2 + sum3 );
      }
      txfree(x); txfree(y); txfree(width);
   }
   else if ( strcmp( meas_type, "when"      ) == 0 ){
      init_val = vec->v_realdata[0] ;
      if ( AlmostEqualUlps( init_val, from, 100 ) ){
         /* match right out of the gate. */
         *result      = vec->v_realdata[0];
         *result_time = time->v_realdata[0];
         return failed ;
      }
      if( init_val < from ){
         /* search upward */
         upflag = TRUE ;
      } else {
         /* search downward */
         upflag = FALSE ;
      }
      idx = -1 ;
      for ( i = 0; i < vec->v_length; i++ ) {
         if ( AlmostEqualUlps( vec->v_realdata[i], from, 100 ) ){
            idx = i ;
            break ;
         } else if( upflag && (vec->v_realdata[i] > from)  ){
            idx = i ;
            break ;
         } else if( !(upflag) && (vec->v_realdata[i] < from) ){
            idx = i ;
            break ;
         }
      }
      if( idx < 0 ){
         return failed;
      }
      *result = vec->v_realdata[idx] ;
      *result_time = interpolate( time, vec, idx-1, i, from, 'x' );
   }
   else {
      if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: (measure2) unknown meas function '%s'.\n", meas_type );
      return TRUE;
   }
   return failed;
}

static bool
chkAnalysisType( char *an_type ) {
  /*
    if ( strcmp( an_type, "ac"    ) != 0 && strcmp( an_type, "dc"   ) != 0 &&
       strcmp( an_type, "noise" ) != 0 && strcmp( an_type, "tran" ) != 0 &&
       strcmp( an_type, "fft"   ) != 0 && strcmp( an_type, "four" ) != 0 )
  */
  /* only support tran analysis type for now */
  if ( strcmp( an_type, "tran" ) != 0 )
    return FALSE;
  else return TRUE;
}

static bool
get_int_value( char **line, char *name, int *value ) {
  char *token     = gettok(line);
  bool return_val = TRUE;
  char *equal_ptr;

  if ( strncmp( token, name, strlen(name) ) != 0 ) {
    if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: syntax error for measure statement; expecting next field to be '%s'.\n", name );
    return_val = FALSE;
  } else {
    /* see if '=' is last char of current token -- implies we need to read value in next token */
    if ( *(token + strlen(token) - 1) == '=' ) {
      txfree(token);
      token  = gettok(line);
      *value = atoi(token);
    } else {
      if ( (equal_ptr = strstr( token, "=" )) ) {
	*value = atoi(equal_ptr+1);
      } else {
	if ( just_chk_meas != TRUE ) fprintf( cp_err, "Error: syntax error for measure statement; missing '='!\n" );
	return_val = FALSE;
      }
    }
  } 
  txfree(token);

  return return_val;
}

static bool
get_double_value( char **line, char *name, double *value ) {
  char *token     = gettok(line);
  bool return_val = TRUE;
  char *equal_ptr, *junk;
  int  err;

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
      if ( (equal_ptr = strstr( token, "=" )) ) {
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

 /*-------------------------------------------------------------------------*
  * gettok skips over whitespace and returns the next token found.  This is 
  * the original version.  It does not "do the right thing" when you have 
  * parens or commas anywhere in the nodelist.  Note that I left this unmodified
  * since I didn't want to break any fcns which called it from elsewhere than
  * subckt.c.  -- SDB 12.3.2003.
  * Since gettok doesn't work right try a new version that does. WPS.
  *-------------------------------------------------------------------------*/
 static char *
 gettok_paren(char **s)
 {
     char buf[BSIZE_SP];
     int i = 0;
     char c;
     int paren;
 
     paren = 0;
     while (isspace(**s))
         (*s)++;
     if (!**s)
         return (NULL);
     while ((c = **s) && !isspace(c)) {
 	if (c == '('/*)*/)
 	    paren += 1;
 	else if (c == /*(*/')'){
 	    paren -= 1;
 	    if( paren <= 0 ) 
 	      break ;
 	} else if (c == ',' && paren < 1)
 	    break;
         buf[i++] = *(*s)++;
     }
     buf[i] = '\0';
     while (isspace(**s) || **s == ',')
         (*s)++;
     return (copy(buf));
 }
 
static char*
get_vector_name( char **line ) {
  char *token, *name;

  token = name = gettok_paren(line);

//  *(name + strlen(name) - 1) = '\0';
  name = strdup(name); txfree(token);

  return name;
}

static bool
do_delay_measurement( char *resname, char *out_line, char *line, char *o_line, int meas_index, double *result ) {
  char   *trig_name, *targ_name, *token;
  char   trig_type, targ_type, trig_polarity, targ_polarity;
  double targ_value, trig_value;
  int    trig_index, targ_index;
  double trig_time = 0, targ_time = 0;
  int    precision = get_measure_precision();
  bool   failed;

  measure_valid[meas_index] = FALSE;

  trig_type = *line; line += 2;           /* skip over vector type and open paren */
  trig_name = get_vector_name( &line );
  if ( trig_type != 'v' && trig_type != 'i' ) {
    if ( just_chk_meas != TRUE ) {
      fprintf( cp_err, "Error: unexpected vector type '%c' for .meas!\n", trig_type );
      fprintf( cp_err, "       %s\n", o_line );
    }
    txfree(trig_name); return FALSE;
  }

  if ( !get_double_value( &line, "val", &trig_value ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); return FALSE; }

  if ( strncmp( line, "rise", 4 ) == 0 ) {
    trig_polarity = 'r';
    if ( !get_int_value( &line, "rise", &trig_index ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); return FALSE; }
  }
  else if ( strncmp( line, "fall", 4 ) == 0 ) {
    trig_polarity = 'f';
    if ( !get_int_value( &line, "fall", &trig_index ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); return FALSE; }
  }
  else {
    if ( just_chk_meas != TRUE ) {
      fprintf( cp_err, "Error: expecting next token to be rise|fall for measurement!\n" );
      fprintf( cp_err, "       %s\n", o_line );
    }
    txfree(trig_name); return FALSE;
  }

  token = gettok(&line);
  if ( strcmp(token, "targ" ) != 0 ) {
    if ( just_chk_meas != TRUE ) {
      fprintf( cp_err, "Error: expected 'targ' as next token in .meas statement!\n" );
      fprintf( cp_err, "       %s\n", o_line );
    }
    txfree(token); txfree(trig_name); return FALSE;
  }
  txfree(token);

  targ_type = *line; line += 2;           /* skip over vector type and open paren */
  targ_name = get_vector_name( &line );
  if ( targ_type != 'v' && targ_type != 'i' ) {
    if ( just_chk_meas != TRUE ) {
      fprintf( cp_err, "Error: unexpected vector type '%c' for .meas!\n", targ_type );
      fprintf( cp_err, "       %s\n", o_line );
    }
    txfree(trig_name); txfree(targ_name); return FALSE;
  }

  if ( !get_double_value( &line, "val", &targ_value ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); txfree(targ_name); return FALSE; }

  if ( strncmp( line, "rise", 4 ) == 0 ) {
    targ_polarity = 'r';
    if ( !get_int_value( &line, "rise", &targ_index ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); txfree(targ_name); return FALSE; }
  }
  else if ( strncmp( line, "fall", 4 ) == 0 ) {
    targ_polarity = 'f';
    if ( !get_int_value( &line, "fall", &targ_index ) ) { if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); txfree(trig_name); txfree(targ_name); return FALSE; }
  }
  else {
    if ( just_chk_meas != TRUE ) {
      fprintf( cp_err, "Error: expecting next token to be rise|fall for measurement!\n" );
      fprintf( cp_err, "       %s\n", o_line );
    }
    txfree(trig_name); txfree(targ_name); return FALSE;
  }

  failed = measure( trig_name, trig_value, trig_polarity, trig_index, targ_name, targ_value, targ_polarity,
		    targ_index, result, &trig_time, &targ_time );

  if ( !failed ) {
    sprintf( out_line, "%-15s=   %.*e targ=   %.*e trig=   %.*e\n", resname, precision, *result, precision, targ_time, precision, trig_time );
    measure_valid[meas_index] = TRUE;
  } else {
    measures_passed = FALSE;
    sprintf( out_line, "%-15s=   failed\n", resname );
    measure_valid[meas_index] = FALSE;
  }

  txfree(trig_name); txfree(targ_name);

  return ( failed ) ? FALSE : TRUE;
}

static bool
do_other_measurement( char *resname, char *out_line, char *meas_type, char *line, char *o_line, int meas_index, double *result ) {

   char   *vec_name;
   char   vec_type;
   double from, to, result_time = 0;
   int    precision = get_measure_precision();
   bool   failed;

   vec_type = *line; line += 2;           /* skip over vector type and open paren */
   vec_name = get_vector_name( &line );
   if ( vec_type != 'v' && vec_type != 'i' ) {
      if ( just_chk_meas != TRUE ) {
         fprintf( cp_err, "Error: unexpected vector type '%c' for .meas!\n", vec_type );
         fprintf( cp_err, "       %s\n", o_line );
      }
      txfree(vec_name); 
      return FALSE;
   }
   if ( strcmp( meas_type, "when" ) == 0 ){
      if ( !get_double_value( &line, NULL, &from ) ) { 
         if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); 
         txfree(vec_name); 
         return FALSE; 
      }
      to = from ;
   } else {
      if ( !get_double_value( &line, "from", &from ) ) { 
         if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); 
         txfree(vec_name); 
         return FALSE; 
      }
      if ( !get_double_value( &line, "to",   &to   ) ) { 
         if ( just_chk_meas != TRUE ) fprintf( cp_err, "       %s\n", o_line ); 
         txfree(vec_name); 
         return FALSE; 
      }
   }

   failed = measure2( meas_type, vec_name, vec_type, from, to, result, &result_time );

   if ( !failed ) {
      if ( strcmp( meas_type, "max" ) == 0 || strcmp( meas_type, "min" ) == 0 )
         sprintf( out_line, "%-15s=   %.*e at=   %.*e\n", resname, precision, *result, precision, result_time );
      else if ( strcmp( meas_type, "when" ) == 0 )
         sprintf( out_line, "%-15s=   %.*e\n", resname, precision, result_time ) ;
      else
         sprintf( out_line, "%-15s=   %.*e from=   %.*e to=   %.*e\n", resname, precision, *result, precision, from, precision, to );
      measure_valid[meas_index] = TRUE;
   } else {
      measures_passed = FALSE;
      sprintf( out_line, "%-15s=   failed\n", resname );
      measure_valid[meas_index] = FALSE;
   }

   txfree(vec_name);

   return ( failed ) ? FALSE : TRUE;
}

void
do_measure( char *what, bool chk_only ) {
  struct line *meas_card, *meas_results = NULL, *end = NULL, *newcard;
  char        *line, *an_name, *an_type, *resname, *meastype, *str_ptr, out_line[1000];
  int         index  = 0, ok = 0;
  double      result = 0;
  bool        first_time = TRUE;
  int         precision = get_measure_precision();

  just_chk_meas = chk_only;

  an_name = strdup( what );
  strtolower( an_name );

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
    else if ( first_time ) {
      first_time = FALSE;

      if ( just_chk_meas != TRUE && strcmp( an_type, "tran" ) == 0 ) {
         fprintf( stdout, "             Transient Analysis\n\n" );
//         plot_cur = setcplot("tran");
      }
    }

    /* skip param|expr measurement types for now -- will be done after other measurements */
    if ( strncmp( meastype, "param", 5 ) == 0 || strncmp( meastype, "expr", 4 ) == 0 ) continue;

    if ( strcmp( an_name, an_type ) != 0 ) {
      txfree(an_type); txfree(resname); txfree(meastype);
      continue;
    }

    if      ( strcmp( meastype, "trig"  ) == 0 || strcmp( meastype, "delay" ) == 0 ) {
      if ( do_delay_measurement( resname, out_line, line, meas_card->li_line, index++, &result ) && just_chk_meas != TRUE ) {
        nupa_add_param( resname, result );
      }
    }
    else if ( strcmp( meastype, "avg"   ) == 0 || strcmp( meastype, "mean"  ) == 0 ||
	      strcmp( meastype, "max"   ) == 0 || strcmp( meastype, "min"   ) == 0 ||
	      strcmp( meastype, "rms"   ) == 0 || strcmp( meastype, "integ" ) == 0 ||
	      strcmp( meastype, "integral" ) == 0 || strcmp( meastype, "when" ) == 0 ) {
      if ( do_other_measurement( resname, out_line, meastype, line, meas_card->li_line, index++, &result ) && just_chk_meas != TRUE ) {
        nupa_add_param( resname, result );
      }
    }
    else {
      measures_passed = FALSE;
      sprintf( out_line, "%-15s=   failed\n", resname );
      if ( just_chk_meas != TRUE ) {
        fprintf( cp_err, "Error: unsupported measurement type '%s' on line %d:\n", meastype, meas_card->li_linenum );
        fprintf( cp_err, "       %s\n", meas_card->li_line );
      }
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

    // see if number of measurements exceeds fixed array size of 20,000
    if ( index >= 20000 ) {
      fprintf( stderr, "ERROR: number of measurements exceeds 20,000!\nAborting...\n" );
      winmessage("Fatal error in SPICE");
      exit(-1);
    }
  }

  // now do param|expr .meas statements
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

    if ( just_chk_meas != TRUE ) fprintf( stdout, "%-15s=", resname );

    if ( just_chk_meas != TRUE ) {
      ok = nupa_eval( meas_card->li_line, meas_card->li_linenum );

      if ( ok ) {
        str_ptr = strstr( meas_card->li_line, meastype );
        if ( !get_double_value( &str_ptr, meastype, &result ) ) {
          if ( just_chk_meas != TRUE ) fprintf( stdout, "   failed\n"       );
        }
        else {
          if ( just_chk_meas != TRUE ) fprintf( stdout, "   %.*e\n", precision, result );
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

bool
check_autostop( char* what ) {
  bool flag = FALSE;
  bool autostop;

  measures_passed = TRUE;
  if ( cp_getvar( "autostop", VT_BOOL, (bool *) &autostop ) ) {
    do_measure( what, TRUE );

    if ( measures_passed == TRUE ) flag = TRUE;
  }

  return flag;
}

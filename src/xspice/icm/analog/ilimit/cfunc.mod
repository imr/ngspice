/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE ilimit/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the ilimit code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/


                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_ilimit()

AUTHORS                      

     6 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the ilimit code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_ILIMIT ROUTINE ===*/

void cm_ilimit(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double in_offset,gain,r_out_source,r_out_sink,i_limit_source,
           i_limit_sink,v_pwr_range,i_source_range,i_sink_range,
           r_out_domain,/*out_lower_limit,out_upper_limit,*/veq,pveq_pvin,
           pveq_pvpos,pveq_pvneg,r_out,pr_out_px,i_out,i_threshold_lower,
           i_threshold_upper,i_pos_pwr,pi_out_pvin,pi_pos_pvneg,
           pi_pos_pvpos,pi_pos_pvout,i_neg_pwr,pi_neg_pvin,pi_neg_pvneg,
           pi_neg_pvpos,pi_neg_pvout,vout,pi_out_plimit,pi_out_pvout,
           pi_out_ppos_pwr,pi_out_pneg_pwr,pi_pos_pvin,pi_neg_plimit,
           pi_pos_plimit,pos_pwr_in,neg_pwr_in;

    Mif_Complex_t ac_gain;
                                                   



    /* Retrieve frequently used parameters... */

    in_offset = PARAM(in_offset);
    gain = PARAM(gain);
    r_out_source = PARAM(r_out_source);
    r_out_sink = PARAM(r_out_sink);
    i_limit_source = PARAM(i_limit_source);
    i_limit_sink = PARAM(i_limit_sink);
    v_pwr_range = PARAM(v_pwr_range);
    i_source_range = PARAM(i_source_range);
    i_sink_range = PARAM(i_sink_range);
    r_out_domain = PARAM(r_out_domain);
                                      




    /* Retrieve frequently used inputs... */

    vout = INPUT(out);
                    


    /* Test to see if pos_pwr or neg_pwr are connected...   */
    /* if not, assign large voltage values to the variables */
    /* pos_pwr andneg_pwr...                                */

   if ( PORT_NULL(pos_pwr) ) {
        pos_pwr_in = 1.0e6;  
    }
    else {
        pos_pwr_in = INPUT(pos_pwr);
    }

    if ( PORT_NULL(neg_pwr) ) {
        neg_pwr_in = -1.0e6;  
    }
    else {
        neg_pwr_in = INPUT(neg_pwr);
    }

    
    /* Compute Veq plus derivatives using climit_fcn */

    if(INIT != 1){ 
    /* If reasonable power and voltage values exist (i.e., not INIT)... */
    /* then calculate expected equivalent voltage values and derivs.    */

	cm_climit_fcn(INPUT(in), in_offset, pos_pwr_in, neg_pwr_in, 
                      0.0, 0.0, v_pwr_range, gain, MIF_FALSE, &veq, 
                      &pveq_pvin, &pveq_pvneg, &pveq_pvpos);
    }                                                            
    else {
    /* Initialization pass...set nominal values */

        veq = (pos_pwr_in - neg_pwr_in) / 2.0;
        pveq_pvin = 0.0;
        pveq_pvpos = 0.0;
        pveq_pvneg = 0.0;
    }


    /* Calculate Rout */

    if (r_out_source == r_out_sink) {  
    /* r_out constant => no calculation necessary */

        r_out = r_out_source;
        pr_out_px = 0.0;

    }
    else {    /* Interpolate smoothly between sourcing & sinking values */
    cm_smooth_discontinuity(veq - vout, -r_out_domain, r_out_sink, r_out_domain,
                   r_out_source, &r_out, &pr_out_px);
    }                                      
                           

         
    /* Calculate i_out & derivatives */
                                                      
    i_threshold_lower = -i_limit_sink + i_sink_range;
    i_threshold_upper = i_limit_source - i_source_range;

    i_out = (veq - vout) / r_out;
    pi_out_pvin = (pveq_pvin/r_out - veq*pr_out_px*pveq_pvin/
                   (r_out*r_out));
    pi_out_pvout = (-1.0/r_out - vout*pr_out_px/(r_out*r_out));

    pi_out_ppos_pwr = (pveq_pvpos/r_out - veq*pr_out_px*pveq_pvpos/
                       (r_out*r_out));
    pi_out_pneg_pwr = (pveq_pvneg/r_out - veq*pr_out_px*pveq_pvneg/
                       (r_out*r_out));
                                                           

    /* Preset i_pos_pwr & i_neg_pwr & partials to 0.0 */

    i_pos_pwr = 0.0;
    pi_pos_pvin = 0.0;
    pi_pos_pvneg = 0.0;
    pi_pos_pvpos = 0.0;
    pi_pos_pvout = 0.0;


    i_neg_pwr = 0.0;
    pi_neg_pvin = 0.0;
    pi_neg_pvneg = 0.0;
    pi_neg_pvpos = 0.0;
    pi_neg_pvout = 0.0;




    /* Determine operating point of i_out for limiting */

    if (i_out < 0.0) {                         /* i_out sinking */
        if (i_out < i_threshold_lower) {     
            if (i_out < (-i_limit_sink-i_sink_range)) { /* i_out lower-limited */
                i_out = -i_limit_sink;
                i_neg_pwr = -i_out;
                pi_out_pvin = 0.0;
                pi_out_pvout = 0.0;
                pi_out_ppos_pwr = 0.0;
                pi_out_pneg_pwr = 0.0;
            }
            else {               /* i_out in lower smoothing region */
                cm_smooth_corner(i_out,-i_limit_sink,-i_limit_sink,i_sink_range,
                            0.0,1.0,&i_out,&pi_out_plimit);
                pi_out_pvin = pi_out_pvin * pi_out_plimit;
                pi_out_pvout = pi_out_pvout * pi_out_plimit;
                pi_out_ppos_pwr = pi_out_ppos_pwr * pi_out_plimit;
                pi_out_pneg_pwr = pi_out_pneg_pwr * pi_out_plimit;

                i_neg_pwr = -i_out;
                pi_neg_pvin = -pi_out_pvin;
                pi_neg_pvneg = -pi_out_pneg_pwr;
                pi_neg_pvpos = -pi_out_ppos_pwr;
                pi_neg_pvout = -pi_out_pvout;
            } 
        }
        else {  /* i_out in lower linear region...calculate i_neg_pwr */
            if (i_out > -2.0*i_sink_range) {  /* i_out near 0.0...smooth i_neg_pwr */
                cm_smooth_corner(i_out,-i_sink_range,0.0,i_sink_range,1.0,0.0,
                            &i_neg_pwr,&pi_neg_plimit);
                i_neg_pwr = -i_neg_pwr;
                pi_neg_pvin = -pi_out_pvin * pi_neg_plimit;
                pi_neg_pvneg = -pi_out_pneg_pwr * pi_neg_plimit;
                pi_neg_pvpos = -pi_out_ppos_pwr * pi_neg_plimit;
                pi_neg_pvout = -pi_out_pvout * pi_neg_plimit;
            }
            else {
                i_neg_pwr = -i_out;  /* Not near i_out=0.0 => i_neg_pwr=-i_out */
                pi_neg_pvin = -pi_out_pvin;
                pi_neg_pvneg = -pi_out_pneg_pwr;
                pi_neg_pvpos = -pi_out_ppos_pwr;
                pi_neg_pvout = -pi_out_pvout;
            }
        }
    }
    else {                                     /* i_out sourcing */
        if (i_out > i_threshold_upper) {
            if (i_out > (i_limit_source + i_source_range)) { /* i_out upper-limited */
                i_out = i_limit_source;
                i_pos_pwr = -i_out;
                pi_out_pvin = 0.0;
                pi_out_pvout = 0.0;
                pi_out_ppos_pwr = 0.0;
                pi_out_pneg_pwr = 0.0;
            }
            else {               /* i_out in upper smoothing region */
                cm_smooth_corner(i_out,i_limit_source,i_limit_source,i_sink_range,
                            1.0,0.0,&i_out,&pi_out_plimit);
                pi_out_pvin = pi_out_pvin * pi_out_plimit;
                pi_out_pvout = pi_out_pvout * pi_out_plimit;
                pi_out_ppos_pwr = pi_out_ppos_pwr * pi_out_plimit;
                pi_out_pneg_pwr = pi_out_pneg_pwr * pi_out_plimit;

                i_pos_pwr = -i_out;
                pi_pos_pvin = -pi_out_pvin;
                pi_pos_pvneg = -pi_out_pneg_pwr;
                pi_pos_pvpos = -pi_out_ppos_pwr;
                pi_pos_pvout = -pi_out_pvout;
            }
        }
        else {  /* i_out in upper linear region...calculate i_pos_pwr */
            if (i_out < 2.0*i_source_range) { /* i_out near 0.0...smooth i_pos_pwr */
                cm_smooth_corner(i_out,i_source_range,0.0,i_source_range,0.0,1.0,
                            &i_pos_pwr,&pi_pos_plimit);
                i_pos_pwr = -i_pos_pwr;
                pi_pos_pvin = -pi_out_pvin * pi_pos_plimit;
                pi_pos_pvneg = -pi_out_pneg_pwr * pi_pos_plimit;
                pi_pos_pvpos = -pi_out_ppos_pwr * pi_pos_plimit;
                pi_pos_pvout = -pi_out_pvout * pi_pos_plimit;
            }
            else {                   /* Not near i_out=0.0 => i_pos_pwr=-i_out */
                i_pos_pwr = -i_out;
                pi_pos_pvin = -pi_out_pvin;
                pi_pos_pvneg = -pi_out_pneg_pwr;
                pi_pos_pvpos = -pi_out_ppos_pwr;
                pi_pos_pvout = -pi_out_pvout;
            }
        }
    }
    




    if (ANALYSIS != MIF_AC) {     /* DC & Transient Analyses */


        /* Debug line...REMOVE FOR FINAL VERSION!!! */
        /*OUTPUT(t1) = veq;
        OUTPUT(t2) = r_out;
        OUTPUT(t3) = pveq_pvin;
        OUTPUT(t4) = pveq_pvpos;
        OUTPUT(t5) = pveq_pvneg;*/


        OUTPUT(out) = -i_out;                     /* Remember...current polarity must be    */
        PARTIAL(out,in) = -pi_out_pvin;           /* reversed for SPICE...all previous code */
        PARTIAL(out,out) = -pi_out_pvout;         /* assumes i_out positive when EXITING    */
       											  /* the model and negative when entering.  */
        										  /* SPICE assumes the opposite, so a       */
                                                  /* minus sign is added to all currents    */
                                                  /* and current partials to compensate for */
                                                  /* this fact....                     JPM  */
                                                                                                
        if ( !PORT_NULL(neg_pwr) ) {
            OUTPUT(neg_pwr) = -i_neg_pwr;                                                         
            PARTIAL(neg_pwr,in) = -pi_neg_pvin;                                                   
            PARTIAL(neg_pwr,out) = -pi_neg_pvout;
			if(!PORT_NULL(pos_pwr)){
            	PARTIAL(neg_pwr,pos_pwr) = -pi_neg_pvpos;
			}
            PARTIAL(neg_pwr,neg_pwr) = -pi_neg_pvneg;
        	PARTIAL(out,neg_pwr) = -pi_out_pneg_pwr;  
        }

        if ( !PORT_NULL(pos_pwr) ) {
            OUTPUT(pos_pwr) = -i_pos_pwr;
            PARTIAL(pos_pwr,in) = -pi_pos_pvin;
            PARTIAL(pos_pwr,out) = -pi_pos_pvout;
            PARTIAL(pos_pwr,pos_pwr) = -pi_pos_pvpos;
        	if ( !PORT_NULL(neg_pwr) ) {
            	PARTIAL(pos_pwr,neg_pwr) = -pi_pos_pvneg;
			}
        	PARTIAL(out,pos_pwr) = -pi_out_ppos_pwr;  
        }

    }
    else {                        /* AC Analysis */
        ac_gain.real = -pi_out_pvin;
        ac_gain.imag= 0.0;
        AC_GAIN(out,in) = ac_gain;

        ac_gain.real = -pi_out_pvout;
        ac_gain.imag= 0.0;
        AC_GAIN(out,out) = ac_gain;

        if ( !PORT_NULL(neg_pwr) ) {
            ac_gain.real = -pi_neg_pvin;
            ac_gain.imag= 0.0;
            AC_GAIN(neg_pwr,in) = ac_gain;

            ac_gain.real = -pi_out_pneg_pwr;
            ac_gain.imag= 0.0;
            AC_GAIN(out,neg_pwr) = ac_gain;

            ac_gain.real = -pi_neg_pvout;
            ac_gain.imag= 0.0;
            AC_GAIN(neg_pwr,out) = ac_gain;
    
            ac_gain.real = -pi_neg_pvpos;
            ac_gain.imag= 0.0;
            AC_GAIN(neg_pwr,pos_pwr) = ac_gain;

            ac_gain.real = -pi_neg_pvneg;
            ac_gain.imag= 0.0;
            AC_GAIN(neg_pwr,neg_pwr) = ac_gain;
        }


        if ( !PORT_NULL(pos_pwr) ) {
            ac_gain.real = -pi_pos_pvin;
            ac_gain.imag= 0.0;
            AC_GAIN(pos_pwr,in) = ac_gain;

            ac_gain.real = -pi_out_ppos_pwr;
            ac_gain.imag= 0.0;
            AC_GAIN(out,pos_pwr) = ac_gain;

            ac_gain.real = -pi_pos_pvout;
            ac_gain.imag= 0.0;
            AC_GAIN(pos_pwr,out) = ac_gain;

            ac_gain.real = -pi_pos_pvpos;
            ac_gain.imag= 0.0;
            AC_GAIN(pos_pwr,pos_pwr) = ac_gain;
    
            ac_gain.real = -pi_pos_pvneg;
            ac_gain.imag= 0.0;
            AC_GAIN(pos_pwr,neg_pwr) = ac_gain;
        }
    }
}






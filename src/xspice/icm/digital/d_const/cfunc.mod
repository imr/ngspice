/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================
               
AUTHORS                      

    18 Nov 2019     Florian Ballenegger, Anamosic Ballenegger Design
                                   

SUMMARY

    This file contains the functional description of the d_const
    code model.
    "word" string format: optional prefix 'h'=hex or 'd'=decimal
    Output vector is LSB first (little endian)

===============================================================================*/

#include <string.h>
#include <stdlib.h>

void cm_d_const(ARGS) 

{
    unsigned int i,n,x;
    char	*str;
    Digital_Strength_t	strength;
    Digital_State_t* vec;
    
    n = (unsigned int) PORT_SIZE(out);
    strength = UNDETERMINED;
    
    if(INIT) {
    	str = PARAM(strength);
	if(strcmp(str,"s")==0)
		strength = STRONG;
	else if(strcmp(str,"r")==0)
		strength = RESISTIVE;
	else if(strcmp(str,"h")==0)
		strength = HI_IMPEDANCE;
	else if(strcmp(str,"u")==0)
		strength = UNDETERMINED;
	else {
		cm_message_send("Unknown strength string");
		cm_message_send(PARAM(strength));
	}
	STATIC_VAR(locstrength) = (int) strength;
	
	str = PARAM(word);
    	if(str[0]!='h' && str[0]!='d')
    	if(strlen(str)!=n)
		cm_message_send("Mismatch between word string value length and output vector bus width");

	vec = (Digital_State_t*) calloc(n, sizeof(Digital_State_t));
	STATIC_VAR(locvec) = vec;
	
	if(str[0]=='d') {
		x = (unsigned int) atoi(&str[1]);
		for(i=0;i<n;i++)
			vec[i] = (x & (1<<i)) ? ONE : ZERO;
	} else if(str[0]=='h') {
		x = (unsigned int) strtol(&str[1],NULL,16);
		for(i=0;i<n;i++)
			vec[i] = (x & (1<<i)) ? ONE : ZERO;
	} else for(i=0;i<n;i++) {
		switch(str[i]){
			case '0':
				vec[i] = ZERO;
				break;
			case '1':
				vec[i] = ONE;
				break;
			default:
				vec[i] = UNKNOWN;
		}
	}
	
    }
    if(1) {
    	strength = (Digital_Strength_t) STATIC_VAR(locstrength);
	vec = (Digital_State_t*) STATIC_VAR(locvec);
    	for(i=0;i<n;i++) {
		OUTPUT_STATE(out[i]) = vec[i];
    		OUTPUT_STRENGTH(out[i]) = strength;
	}
    }
}

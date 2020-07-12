/*
** Author: Francesco Lannutti
** Date: 15 March 2020
** Purpose: KLU binding table routines for XSPICE
*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifdefs.h"

int
MIFbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MIFmodel *model = (MIFmodel *)inModel ;
    MIFinstance *here ;
    BindElement i, *matched, *BindStruct ;
    int ii, j, k, l, num_conn, num_port, num_port_k ;
    Mif_Boolean_t is_input, is_output ;
    Mif_Cntl_Src_Type_t cntl_src_type ;
    Mif_Port_Type_t in_type, out_type, type ;
    Mif_Smp_Ptr_t *smp_data_cntl, *smp_data_out ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the MIF models */
    for ( ; model != NULL ; model = MIFnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MIFinstances(model); here != NULL ; here = MIFnextInstance(here))
        {
          /* Skip these expensive allocations if the instance is not analog */
	  if(! here->analog)
	    continue;

	  num_conn = here->num_conn;

            /* loop through all connections on this instance */
            /* and create matrix data needed for outputs and */
            /* V sources associated with I inputs            */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null, skip to next connection */
                if(here->conn[ii]->is_null)
                    continue;

                /* prepare things for convenient access later */
                is_input = here->conn[ii]->is_input;
                is_output = here->conn[ii]->is_output;
                num_port = here->conn[ii]->size;

                /* loop through all ports on this connection */
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this port */
                    type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* if it has a voltage source output, */
                    /* create the matrix data needed      */
                    if( (is_output && (type == MIF_VOLTAGE || type == MIF_DIFF_VOLTAGE)) ||
                                     (type == MIF_RESISTANCE || type == MIF_DIFF_RESISTANCE) ) {

                        /* then make the matrix pointers */
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(pos_branch, pos_branchBinding, pos_node, branch);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(neg_branch, neg_branchBinding, neg_node, branch);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(branch_pos, branch_posBinding, branch,  pos_node);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(branch_neg, branch_negBinding, branch,  neg_node);
                    } /* end if current input */

                    /* if it is a current input */
                    /* create the matrix data needed for the associated zero-valued V source */
                    if(is_input && (type == MIF_CURRENT || type == MIF_DIFF_CURRENT)) {

                        /* then make the matrix pointers */
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(pos_ibranch, pos_ibranchBinding, pos_node, ibranch);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(neg_ibranch, neg_ibranchBinding, neg_node, ibranch);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(ibranch_pos, ibranch_posBinding, ibranch,  pos_node);
                        CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(ibranch_neg, ibranch_negBinding, ibranch,  neg_node);
                    } /* end if current input */
                } /* end for number of ports */
            } /* end for number of connections */

            /* now loop through all connections on the instance and create */
            /* matrix data needed for partial derivatives of outputs       */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if((here->conn[ii]->is_null) || (! here->conn[ii]->is_output))
                    continue;

                /* loop through all ports on this connection */

                num_port = here->conn[ii]->size;
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this output port */
                    out_type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* for this port, loop through all connections */
                    /* and all ports to touch on each possible input */
                    for(k = 0; k < num_conn; k++) {

                        /* if the connection is null or is not an input */
                        /* skip to next connection */
                        if((here->conn[k]->is_null) || (! here->conn[k]->is_input))
                            continue;

                        num_port_k = here->conn[k]->size;
                        /* loop through all the ports of this connection */
                        for(l = 0; l < num_port_k; l++) {

                            /* if port is null, skip to next */
                            if(here->conn[k]->port[l]->is_null)
                                continue;

                            /* determine the type of this input port */
                            in_type = here->conn[k]->port[l]->type;

                            /* create a pointer to the smp data for quick access */
                            smp_data_cntl = &(here->conn[k]->port[l]->smp_data);

                            /* determine type of controlled source according */
                            /* to input and output types */
                            cntl_src_type = MIFget_cntl_src_type(in_type, out_type);

                            switch(cntl_src_type) {
                            case MIF_VCVS:
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_E(e.branch_poscntl, branch_poscntlBinding, branch, pos_node);
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_E(e.branch_negcntl, branch_negcntlBinding, branch, neg_node);
                                break;
                            case MIF_ICIS:
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_F(f.pos_ibranchcntl, pos_ibranchcntlBinding, pos_node, ibranch);
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_F(f.neg_ibranchcntl, neg_ibranchcntlBinding, neg_node, ibranch);
                                break;
                            case MIF_VCIS:
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(g.pos_poscntl, pos_poscntlBinding, pos_node, pos_node);
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(g.pos_negcntl, pos_negcntlBinding, pos_node, neg_node);
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(g.neg_poscntl, neg_poscntlBinding, neg_node, pos_node);
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(g.neg_negcntl, neg_negcntlBinding, neg_node, neg_node);
                                break;
                            case MIF_ICVS:
                                CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_H(h.branch_ibranchcntl, branch_ibranchcntlBinding, branch, ibranch);
                                break;
                            case MIF_minus_one:
                                break;
                            } /* end switch on controlled source type */
                        } /* end for number of input ports */
                    } /* end for number of input connections */
                } /* end for number of output ports */
            } /* end for number of output connections */
        }
    }

    return (OK) ;
}

int
MIFbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MIFmodel *model = (MIFmodel *)inModel ;
    MIFinstance *here ;
    int ii, j, k, l, num_conn, num_port, num_port_k ;
    Mif_Boolean_t is_input, is_output ;
    Mif_Cntl_Src_Type_t cntl_src_type ;
    Mif_Port_Type_t in_type, out_type, type ;
    Mif_Smp_Ptr_t *smp_data_cntl, *smp_data_out ;

    NG_IGNORE (ckt) ;

    /* loop through all the MIF models */
    for ( ; model != NULL ; model = MIFnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MIFinstances(model); here != NULL ; here = MIFnextInstance(here))
        {
          /* Skip these expensive allocations if the instance is not analog */
	  if(! here->analog)
	    continue;

	  num_conn = here->num_conn;

            /* loop through all connections on this instance */
            /* and create matrix data needed for outputs and */
            /* V sources associated with I inputs            */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null, skip to next connection */
                if(here->conn[ii]->is_null)
                    continue;

                /* prepare things for convenient access later */
                is_input = here->conn[ii]->is_input;
                is_output = here->conn[ii]->is_output;
                num_port = here->conn[ii]->size;

                /* loop through all ports on this connection */
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this port */
                    type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* if it has a voltage source output, */
                    /* create the matrix data needed      */
                    if( (is_output && (type == MIF_VOLTAGE || type == MIF_DIFF_VOLTAGE)) ||
                                     (type == MIF_RESISTANCE || type == MIF_DIFF_RESISTANCE) ) {

                        /* then make the matrix pointers */
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(pos_branch, pos_branchBinding, pos_node, branch);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(neg_branch, neg_branchBinding, neg_node, branch);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(branch_pos, branch_posBinding, branch,  pos_node);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(branch_neg, branch_negBinding, branch,  neg_node);
                    } /* end if current input */

                    /* if it is a current input */
                    /* create the matrix data needed for the associated zero-valued V source */
                    if(is_input && (type == MIF_CURRENT || type == MIF_DIFF_CURRENT)) {

                        /* then make the matrix pointers */
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(pos_ibranch, pos_ibranchBinding, pos_node, ibranch);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(neg_ibranch, neg_ibranchBinding, neg_node, ibranch);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(ibranch_pos, ibranch_posBinding, ibranch,  pos_node);
                        CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(ibranch_neg, ibranch_negBinding, ibranch,  neg_node);
                    } /* end if current input */
                } /* end for number of ports */
            } /* end for number of connections */

            /* now loop through all connections on the instance and create */
            /* matrix data needed for partial derivatives of outputs       */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if((here->conn[ii]->is_null) || (! here->conn[ii]->is_output))
                    continue;

                /* loop through all ports on this connection */

                num_port = here->conn[ii]->size;
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this output port */
                    out_type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* for this port, loop through all connections */
                    /* and all ports to touch on each possible input */
                    for(k = 0; k < num_conn; k++) {

                        /* if the connection is null or is not an input */
                        /* skip to next connection */
                        if((here->conn[k]->is_null) || (! here->conn[k]->is_input))
                            continue;

                        num_port_k = here->conn[k]->size;
                        /* loop through all the ports of this connection */
                        for(l = 0; l < num_port_k; l++) {

                            /* if port is null, skip to next */
                            if(here->conn[k]->port[l]->is_null)
                                continue;

                            /* determine the type of this input port */
                            in_type = here->conn[k]->port[l]->type;

                            /* create a pointer to the smp data for quick access */
                            smp_data_cntl = &(here->conn[k]->port[l]->smp_data);

                            /* determine type of controlled source according */
                            /* to input and output types */
                            cntl_src_type = MIFget_cntl_src_type(in_type, out_type);

                            switch(cntl_src_type) {
                            case MIF_VCVS:
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_E(e.branch_poscntl, branch_poscntlBinding, branch, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_E(e.branch_negcntl, branch_negcntlBinding, branch, neg_node);
                                break;
                            case MIF_ICIS:
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_F(f.pos_ibranchcntl, pos_ibranchcntlBinding, pos_node, ibranch);
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_F(f.neg_ibranchcntl, neg_ibranchcntlBinding, neg_node, ibranch);
                                break;
                            case MIF_VCIS:
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(g.pos_poscntl, pos_poscntlBinding, pos_node, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(g.pos_negcntl, pos_negcntlBinding, pos_node, neg_node);
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(g.neg_poscntl, neg_poscntlBinding, neg_node, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(g.neg_negcntl, neg_negcntlBinding, neg_node, neg_node);
                                break;
                            case MIF_ICVS:
                                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_H(h.branch_ibranchcntl, branch_ibranchcntlBinding, branch, ibranch);
                                break;
                            case MIF_minus_one:
                                break;
                            } /* end switch on controlled source type */
                        } /* end for number of input ports */
                    } /* end for number of input connections */
                } /* end for number of output ports */
            } /* end for number of output connections */
        }
    }

    return (OK) ;
}

int
MIFbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MIFmodel *model = (MIFmodel *)inModel ;
    MIFinstance *here ;
    int ii, j, k, l, num_conn, num_port, num_port_k ;
    Mif_Boolean_t is_input, is_output ;
    Mif_Cntl_Src_Type_t cntl_src_type ;
    Mif_Port_Type_t in_type, out_type, type ;
    Mif_Smp_Ptr_t *smp_data_cntl, *smp_data_out ;

    NG_IGNORE (ckt) ;

    /* loop through all the MIF models */
    for ( ; model != NULL ; model = MIFnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MIFinstances(model); here != NULL ; here = MIFnextInstance(here))
        {
          /* Skip these expensive allocations if the instance is not analog */
	  if(! here->analog)
	    continue;

	  num_conn = here->num_conn;

            /* loop through all connections on this instance */
            /* and create matrix data needed for outputs and */
            /* V sources associated with I inputs            */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null, skip to next connection */
                if(here->conn[ii]->is_null)
                    continue;

                /* prepare things for convenient access later */
                is_input = here->conn[ii]->is_input;
                is_output = here->conn[ii]->is_output;
                num_port = here->conn[ii]->size;

                /* loop through all ports on this connection */
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this port */
                    type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* if it has a voltage source output, */
                    /* create the matrix data needed      */
                    if( (is_output && (type == MIF_VOLTAGE || type == MIF_DIFF_VOLTAGE)) ||
                                     (type == MIF_RESISTANCE || type == MIF_DIFF_RESISTANCE) ) {

                        /* then make the matrix pointers */
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(pos_branch, pos_branchBinding, pos_node, branch);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(neg_branch, neg_branchBinding, neg_node, branch);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(branch_pos, branch_posBinding, branch,  pos_node);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(branch_neg, branch_negBinding, branch,  neg_node);
                    } /* end if current input */

                    /* if it is a current input */
                    /* create the matrix data needed for the associated zero-valued V source */
                    if(is_input && (type == MIF_CURRENT || type == MIF_DIFF_CURRENT)) {

                        /* then make the matrix pointers */
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(pos_ibranch, pos_ibranchBinding, pos_node, ibranch);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(neg_ibranch, neg_ibranchBinding, neg_node, ibranch);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(ibranch_pos, ibranch_posBinding, ibranch,  pos_node);
                        CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(ibranch_neg, ibranch_negBinding, ibranch,  neg_node);
                    } /* end if current input */
                } /* end for number of ports */
            } /* end for number of connections */

            /* now loop through all connections on the instance and create */
            /* matrix data needed for partial derivatives of outputs       */
            for(ii = 0; ii < num_conn; ii++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if((here->conn[ii]->is_null) || (! here->conn[ii]->is_output))
                    continue;

                /* loop through all ports on this connection */

                num_port = here->conn[ii]->size;
                for(j = 0; j < num_port; j++) {

                    /* if port is null, skip to next */
                    if(here->conn[ii]->port[j]->is_null)
                        continue;

                    /* determine the type of this output port */
                    out_type = here->conn[ii]->port[j]->type;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(here->conn[ii]->port[j]->smp_data);

                    /* for this port, loop through all connections */
                    /* and all ports to touch on each possible input */
                    for(k = 0; k < num_conn; k++) {

                        /* if the connection is null or is not an input */
                        /* skip to next connection */
                        if((here->conn[k]->is_null) || (! here->conn[k]->is_input))
                            continue;

                        num_port_k = here->conn[k]->size;
                        /* loop through all the ports of this connection */
                        for(l = 0; l < num_port_k; l++) {

                            /* if port is null, skip to next */
                            if(here->conn[k]->port[l]->is_null)
                                continue;

                            /* determine the type of this input port */
                            in_type = here->conn[k]->port[l]->type;

                            /* create a pointer to the smp data for quick access */
                            smp_data_cntl = &(here->conn[k]->port[l]->smp_data);

                            /* determine type of controlled source according */
                            /* to input and output types */
                            cntl_src_type = MIFget_cntl_src_type(in_type, out_type);

                            switch(cntl_src_type) {
                            case MIF_VCVS:
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_E(e.branch_poscntl, branch_poscntlBinding, branch, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_E(e.branch_negcntl, branch_negcntlBinding, branch, neg_node);
                                break;
                            case MIF_ICIS:
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_F(f.pos_ibranchcntl, pos_ibranchcntlBinding, pos_node, ibranch);
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_F(f.neg_ibranchcntl, neg_ibranchcntlBinding, neg_node, ibranch);
                                break;
                            case MIF_VCIS:
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(g.pos_poscntl, pos_poscntlBinding, pos_node, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(g.pos_negcntl, pos_negcntlBinding, pos_node, neg_node);
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(g.neg_poscntl, neg_poscntlBinding, neg_node, pos_node);
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(g.neg_negcntl, neg_negcntlBinding, neg_node, neg_node);
                                break;
                            case MIF_ICVS:
                                CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_H(h.branch_ibranchcntl, branch_ibranchcntlBinding, branch, ibranch);
                                break;
                            case MIF_minus_one:
                                break;
                            } /* end switch on controlled source type */
                        } /* end for number of input ports */
                    } /* end for number of input connections */
                } /* end for number of output ports */
            } /* end for number of output connections */
        }
    }

    return (OK) ;
}

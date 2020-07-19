//this file contains a class definition for models to be integrated into ngspice
//later, it should be moved somewhere else.
//for the time beeing, HICUM is the first model to use this.
#include <unordered_map>
#include <limits>
#include <model_class.hpp>
#include <stdexcept>
extern "C" {
#include "ngspice/cktdefs.h"
#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"
}

Parameter::Parameter(double value_default, double min_value, double max_value) {
    /* Modelcard parameter constructor.
    Input
    -----
    value_default : double
        Default value of the parameter.
    min_value     : double
        Minimum value of the parameter.
    max_value     : double
        Maximum value of the parameter.
    */
    value_default = value_default;
    value_min     = min_value;
    value_max     = max_value;
};

void Parameter::setValue(double new_value){
    /* Set the value of the parameter. TODO: check limits value_min/max
    Input
    -----
    new_value : double
        The new value to be set.
    */
    value = new_value;
};

double Parameter::getValue(){
    /* Return the value of the parameter.
    Output
    -----
    value : double
        The value of the parameter.
    */
    return value;
};


Node::Node(CKTcircuit * ckt, NGSpiceModel * model , char * name){
    /* Node constructor.
    Output
    -----
    ckt : CKTcircuit *
        The circuit in which the node shall be allocated.
    model : NGSpiceModel *
        The model that contains the node.
    name : char *
        The name of the node.
    */
    int      error = 0;

    CKTnode *tmp;
    ckt  = ckt;
    name = name;

    //allocate the node
    error = CKTmkVolt(ckt,&tmp, (char*)model->name, name);
    //if(error) return(error); //why should there be an error?
    id = tmp->number;
    // what is the purpose of this?
    // if (ckt->CKTcopyNodesets) {
    //     if (CKTinst2Node(ckt,here,2,&tmpNode,&tmpName)==OK) {
    //         if (tmpNode->nsGiven) {
    //         tmp->nodeset=tmpNode->nodeset;
    //         tmp->nsGiven=tmpNode->nsGiven;
    //         }
    //     }
    // }
};

double Node::getPotential(){
    /* Return the potential of the node from the RHSold.
    */
    return *(ckt->CKTrhsOld+id);
};


Branch::Branch(Node * node_from,Node * node_to,std::vector<Node*> depends){
    node_from             = node_from;
    node_to               = node_to;
    branch_dependencies   = depends;
};

double inf = std::numeric_limits<double>::infinity();

HICUML2::HICUML2(){
    modelcard = {
        {"c10",Parameter(2.0E-30, 0,1)},
        {"qp0",Parameter(2.0E-14,0,1)},
        {"ich",Parameter(0.0,0,inf)},
        {"hf0",Parameter(1.0,0,inf)},
        {"hfe",Parameter(1.0,0,inf)},
        {"hfc",Parameter(1.0,0,inf)},
        {"hjei",Parameter(1.0,0,100)},
        {"ahjei",Parameter(0.0,0,100)},
        {"rhjei",Parameter(1.0,0,10)},
        {"hjci",Parameter(1.0,0,100)},
        {"ibeis",Parameter(1.0E-18,      0,1)},
        {"mbei",Parameter(1.0,          0,10)},
        {"ireis",Parameter(0.0,          0,1)},
        {"mrei",Parameter(2.0,          0,10)},
        {"ibeps",Parameter(0.0,          0,1)},
        {"mbep",Parameter(1.0,          0,10)},
        {"ireps",Parameter(0.0,          0,1)},
        {"mrep",Parameter(2.0,          0,10)},
        {"mcf",Parameter(1.0,          0,10)},
        {"ibcis",Parameter(1.0E-16   ,    0,1.0)},
        {"mbci",Parameter(1.0           ,0,10)},
        {"ibcxs",Parameter(0.0           ,0,1.0)},
        {"mbcx",Parameter(1.0,0,10)},
        {"ibets",Parameter (0.0     , 0,50)},
        {"abet",Parameter  (40,0,inf)},
        {"tunode",Parameter(1,0,1)},
        {"favl",Parameter(0.0,0,inf)},
        {"qavl",Parameter(0.0,0,inf)},
        {"kavl",Parameter(0.0,0,3)},
        {"alfav",Parameter(0.0 ,-inf,inf)},
        {"alqav",Parameter(0.0 ,-inf,inf)},
        {"alkav",Parameter(0.0,-inf,inf)},
        {"rbi0",Parameter(0.0,0,inf)},
        {"rbx",Parameter(0.0, 0,inf)},
        {"fgeo",Parameter(0.6557, 0,inf)},
        {"fdqr0",Parameter(0.0 , -0.5,100)},
        {"fcrbi",Parameter(0.0, 0,1)},
        {"fqi",Parameter(1.0, 0,1)},
        {"re",Parameter(0.0, 0,inf)},
        {"rcx",Parameter(0.0,0,inf)},
        {"itss",Parameter(0.0 , 0,1.0)},
        {"msf",Parameter(1.0, 0,10)},
        {"iscs",Parameter(0.0, 0,1.0)},
        {"msc",Parameter(1.0, 0,10)},
        {"tsf",Parameter(0.0, 0,inf)},
        {"rsu",Parameter(0.0, 0,inf)},
        {"csu",Parameter(0.0, 0,inf)},
        {"cjei0",Parameter(1.0E-20, 0,inf)},
        {"vdei",Parameter(0.9, 0,10)},
        {"zei",Parameter(0.5,0,1)},
        {"ajei",Parameter(2.5, 1,inf)},
        {"cjep0",Parameter(1.0E-20,0,inf)},
        {"vdep",Parameter(0.9, 0,10)},
        {"zep",Parameter(0.5, 0,1)},
        {"ajep",Parameter(2.5, 1,inf)},
        {"cjci0",Parameter(1.0E-20, 0,inf)},
        {"vdci",Parameter(0.7, 0,10)},
        {"zci",Parameter(0.4, 0,1)},
        {"vptci",Parameter(100, 0,100)},
        {"cjcx0",Parameter(1.0E-20, 0,inf)},
        {"vdcx",Parameter(0.7, 0,10)},
        {"zcx",Parameter(0.4, 0,1)},
        {"vptcx",Parameter(100,0,100)},
        {"fbcpar",Parameter(0.0,0,1)},
        {"fbepar",Parameter(1.0,0,1)},
        {"cjs0",Parameter( 0.0, 0,inf)},
        {"vds",Parameter(0.6,0,10)},
        {"zs",Parameter(0.5, 0,1)},
        {"vpts",Parameter(100,0,100)},
        {"cscp0",Parameter(0.0, 0,inf)},
        {"vdsp",Parameter(0.6, 0,10)},
        {"zsp",Parameter(0.5, 0,1)},
        {"vptsp",Parameter(100, 0,100)},
        {"t0",Parameter(0.0, 0,inf)},
        {"dt0h",Parameter(0.0,-inf,inf)},
        {"tbvl",Parameter(0.0,-inf,inf)},
        {"tef0",Parameter(0.0, 0,inf)},
        {"gtfe",Parameter(1.0, 0,10)},
        {"thcs",Parameter(0.0, 0,inf)},
        {"ahc",Parameter(0.1, 0,50)},
        {"fthc",Parameter(0.0, 0,1)},
        {"rci0",Parameter(150, 0,inf)},
        {"vlim",Parameter(0.5, 0,10)},
        {"vces",Parameter(0.1,0,1)},
        {"vpt",Parameter(100.0,0,inf)},
        {"aick",Parameter(1e-3,0,10)},
        {"delck",Parameter(2.0, 0,10)},
        {"tr",Parameter(0.0, 0,inf)},
        {"vcbar",Parameter(0.0,0,1)},
        {"icbar",Parameter(0.0,0,1)},
        {"acbar",Parameter(0.01,0,10)},
        {"cbepar",Parameter(0.0,0,inf)},
        {"cbcpar",Parameter(0.0,0,inf)},
        {"alqf",Parameter(0.167, 0,1)},
        {"alit",Parameter(0.333, 0,1)},
        {"flnqs",Parameter(0,-10,10)},
        {"kf",Parameter(0.0,0,inf)},
        {"af",Parameter(2.0,0,10)},
        {"cfbe",Parameter(-1,-2,-1)},
        {"flcono",Parameter(0,0,1)},
        {"kfre",Parameter(0.0,0,inf)},
        {"afre",Parameter(2.0,0,10)},
        {"latl",Parameter(0.0,0,inf)},
        {"vgb",Parameter(1.17,0,10)},
        {"alt0",Parameter(0.0,-inf,inf)},
        {"kt0",Parameter(0.0,-inf,inf)},
        {"zetaci",Parameter(0.0,-10,10)},
        {"alvs",Parameter(0.0,-inf,inf)},
        {"alces",Parameter(0.0,-inf,inf)},
        {"zetarbi",Parameter(0.0,-10,10)},
        {"zetarbx",Parameter(0.0,-10,10)},
        {"zetarcx",Parameter(0.0,-10,10)},
        {"zetare",Parameter(0.0,-10,10)},
        {"zetacx",Parameter(1.0,-10,10)},
        {"vge",Parameter(1.17,0,10)},
        {"vgc",Parameter(1.17,0,10)},
        {"vgs",Parameter(1.17,0,10)},
        {"f1vg",Parameter(-1.02377e-4,-inf,inf)},
        {"f2vg",Parameter(4.3215e-4,-inf,inf)},
        {"zetact",Parameter(3.0,-10,10)},
        {"zetabet",Parameter(3.5,-10,10)},
        {"alb",Parameter(0.0,-inf,inf)},
        {"dvgbe",Parameter(0,-inf,inf)},
        {"zetahjei",Parameter(1,-10,10)},
        {"zetavgbe",Parameter(1,-10,10)},
        {"flsh",Parameter(0,0,2)},
        {"rth",Parameter(0.0,0,inf)},
        {"zetarth",Parameter(0.0,-10,10)},
        {"alrth",Parameter(0.0,-10,10)},
        {"cth",Parameter(0.0,0,inf)},
        {"flcomp",Parameter(0.0,0,inf)},
        {"tnom",Parameter(27.0,-inf,inf)},
        {"dt",Parameter(0.0,-inf,inf)},
        {"type",Parameter(1,-1,1)},
    };
};

HICUML2 hicumL2_example = HICUML2();

void setModelcardPara(const char* para_name, double value){
    Parameter para(0,0,0);
    try {
        para = hicumL2_example.modelcard.at(para_name);
    } catch (const std::out_of_range& oor) {
        printf("Experimental HICUM: did not find parameter with name %s.\n", para_name);
    }
    para.setValue(value);
};
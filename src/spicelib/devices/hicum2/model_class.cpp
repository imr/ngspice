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
#include "ngspice/devdefs.h"
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


Node::Node( char * name){
    name = name;
};

double Node::getPotential(){
    /* Return the potential of the node from the RHSold.
    */
    return *(ckt->CKTrhsOld+id);
};


Branch::Branch(Node * node_from,Node * node_to,std::vector<Node*> depends){
    node_from            = node_from;
    node_to              = node_to;
    branch_dependencies  = depends;
};





#ifndef model_class
#define model_class
//this file contains a class definition for models to be integrated into ngspice
//later, it should be moved somewhere else.
//for the time beeing, HICUM is the first model to use this.
#ifdef __cplusplus
#include <unordered_map>
#include <vector>
#include <limits>
extern "C" {
#include "ngspice/cktdefs.h"
#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
}

class NGSpiceModel;

class Parameter
{
    public:
    double value;
    double value_min;
    double value_max;
    double value_default;

    Parameter(double value_default, double min_value, double max_value);
    void setValue(double new_value);
    double getValue(); // -> maybe with operator overloading
};

class Node
{
    public:
    int id;
    char * name;
    CKTcircuit * ckt; //pointer to the circuit in which the node resides
    double getPotential();
    //Node(CKTcircuit * ckt, NGSpiceModel * model, char * name);
    Node(char * name);
};

class Branch
{
    //A branch specifies a current that flows from node_from to node_to 
    public:
    double value;                                    //the current of the branch
    std::unordered_map<int, double> derivatives;     //a map whose keys are node ids and the value is the derivative with that node
    std::vector<Node*> branch_dependencies;          //all nodes for which a derivative is defined
    Node * node_from;                                //the node where the current originates
    Node * node_to;                                  //the node where the current is flowing to
    Branch(Node * node_from, Node * node_to, std::vector<Node*> depends);
};

class NGSpiceModel
{
    public:
    std::unordered_map<const char *, Parameter> modelcard; //the modelcard of the model
    std::vector<Node>   nodes;          //all nodes of the model, including internal and externals
    std::vector<Node>   external_nodes; //the external nodes of the model
    std::vector<Branch> branches;       //the branches of the model
    int n_external_nodes;        //the name of the model
    const char *  name = "None";        //the name of the model
};

class HICUML2 : public NGSpiceModel
{
    public:
    const char *  name = "hl2";
    HICUML2();
    SPICEdev * get_model_info(); // return the SPICEdev
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

//methods that are exposed to C
extern void setModelcardPara(const char* para_name, double value);

#ifdef __cplusplus
}
#endif
#endif
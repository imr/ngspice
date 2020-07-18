#ifndef model_class
#define model_class
//this file contains a class definition for models to be integrated into ngspice
//later, it should be moved somewhere else.
//for the time beeing, HICUM is the first model to use this.
#ifdef __cplusplus
#include <unordered_map>
#include <limits>

class Parameter
{
    public:
    double value;
    double value_min;
    double value_max;
    double value_default;

    Parameter(double value_default, double min_value, double max_value);
    void setValue(double new_value);
    double getValue();
};

class NGSpiceModel
{
    public:
    std::unordered_map<const char *, Parameter> modelcard; //the modelcard of the model
};

class HICUML2 : public NGSpiceModel
{
    public:
    HICUML2();
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
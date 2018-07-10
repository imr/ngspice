/**********
Author: 2018 Thomas P. Dye
**********/

/*
	For translating from a polynomial controlled source to a B source
*/

#include "ngspice/ngspice.h"
#include "ngspice/stringutil.h"
#include "ngspice/stringskip.h"
#include "translatepoly.h"

#define TRANSLATEPOLY_REPLACE TRUE

typedef struct
{
	int Dimensions;
	char **ExpressionList;
	int Coefficients;
	char **CoefficientList;
} Poly_t ;

typedef struct
{
	int *ExpressionIterator;
	int numterms;
} permutation_t ;

int count_occurrences(permutation_t a, int IteratorN)
{
	int result = 0;
	for(int i = 0; i < a.numterms; i++){
		if(a.ExpressionIterator[i] == IteratorN){
			result++;
		}
	}
	return result;
}

int compare(permutation_t a, permutation_t b, int ndimensions)
{
	if(a.numterms != b.numterms){
		return (a.numterms - b.numterms)<<1;
	}
	else {
		int termsfound_a = 0, termsfound_b = 0;
		for(int i = 0; i < ndimensions; i++){
			int ioccursina = 0, ioccursinb = 0;
			if(termsfound_a < a.numterms){
				ioccursina = count_occurrences(a, i);
				termsfound_a += ioccursina;
			}
			if(termsfound_b < a.numterms){
				ioccursinb = count_occurrences(b, i);
				termsfound_b += ioccursinb;
			}
			if(ioccursinb != ioccursina){
				return -1;
			}
			if(termsfound_a == a.numterms && termsfound_b == b.numterms){
				return 0;
			}
		}
	}
}

permutation_t increment(permutation_t input, int ndimensions)
{
	permutation_t next;
	bool nexthasmoreterms = TRUE;
	for(int i = 0; i < input.numterms; i++){
		if(input.ExpressionIterator[i] != (ndimensions - 1)){
			nexthasmoreterms = FALSE;
			break;
		}
	}
	if(nexthasmoreterms){
		next.numterms = input.numterms + 1;
		next.ExpressionIterator = (int * ) tmalloc(next.numterms * sizeof(int));
		if(next.ExpressionIterator == NULL){
			fprintf(stderr, "ERROR: Out of memory");
			controlled_exit(EXIT_BAD);
		}
		for(int i = 0; i < next.numterms; i++){
			next.ExpressionIterator[i] = 0;
		}
	}
	else{
		next.numterms = input.numterms;
		next.ExpressionIterator = (int * ) tmalloc(next.numterms * sizeof(int));
		if(next.ExpressionIterator == NULL){
			fprintf(stderr, "ERROR: Out of memory");
			controlled_exit(EXIT_BAD);
		}
		for(int i = 0; i < next.numterms; i++){
			if(input.ExpressionIterator[i] == (ndimensions - 1)){
				next.ExpressionIterator[i] = 0;
			}
			else{
				next.ExpressionIterator[i] = input.ExpressionIterator[i] + 1;
				break;
			}
		}
	}
	return next;
}

Poly_t interpretpoly(struct line * input, char controlType, char sourceType)
{
	Poly_t Poly;
	Poly.Dimensions = 0;
	Poly.Coefficients = 0;
	char * linestr;
	linestr = input->li_line;
	char * polystr;
	polystr = strstr(linestr, "poly(");
	if(polystr == NULL){
		return Poly;
	}
	int charsAfterPoly = 0;
	sscanf(polystr, "poly( %u )%n", &Poly.Dimensions, &charsAfterPoly);
	if(Poly.Dimensions == 0){
		fprintf(stderr, "ERROR: POLY interpreted as having 0 dimensions, Syntax Error Assumed.");
		controlled_exit(EXIT_BAD);
	}
	Poly.ExpressionList = (char **) tmalloc(Poly.Dimensions * sizeof(char *));
	if(Poly.ExpressionList == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	char * cutstr;
	cutstr = &polystr[charsAfterPoly];
	char ** nodepair;
	nodepair = (char **) tmalloc(2 * sizeof(char *));
	if(nodepair == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	char * formatVC = "v( %s , %s )", * formatCC = "( i( %s ) - i( %s ) )", * formatselected;
	switch(controlType){
		case 'i': formatselected = formatCC;
		case 'v': formatselected = formatVC;
		default: break; //TODO: decide what to do on other characters?
	}
	//Get nodes and populate expression list
	for(int i = 0; i < Poly.Dimensions; i++){
		for(int j = 0; j < 2; j++){
			if(cutstr[0] == '\0'){
				//Syntax Error
				fprintf(stderr, "Error: Too few control node/source pairs for POLY command in line %i", input->li_linenum_orig);
			}
			nodepair[j] = gettok_node(&cutstr);
		}
		const int minExpressionSize = 64;
		Poly.ExpressionList[i] = (char *) tmalloc(minExpressionSize * sizeof(char));
		if(Poly.ExpressionList[i] == NULL){
			fprintf(stderr, "ERROR: Out of memory");
			controlled_exit(EXIT_BAD);
		}
		
		int nreturned = snprintf(Poly.ExpressionList[i],minExpressionSize,formatselected,nodepair[0],nodepair[1]);
		if(nreturned >= minExpressionSize){
			Poly.ExpressionList[i] = (char *) tmalloc((nreturned + 1) * sizeof(char));
			if(Poly.ExpressionList[i] == NULL){
				fprintf(stderr, "ERROR: Out of memory");
				controlled_exit(EXIT_BAD);
			}
			sprintf(Poly.ExpressionList[i],formatselected,nodepair[0],nodepair[1]);
		}
	}
	//Iterate past any whitespace
	while(isspace(cutstr[0])){
		cutstr++;
	}
	char * iterstr;
	iterstr = cutstr;
	int coefficientIter = 0;
	Poly.CoefficientList = (char **) tmalloc(1 * sizeof(char *));
	if(Poly.CoefficientList == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	//Get coefficients until end of line
	//TODO: handle expression based and parametric coefficients
	while(*iterstr != '\0'){
		if(isspace(*iterstr)){
			if(coefficientIter !=0){
				Poly.CoefficientList = (char **) trealloc(Poly.CoefficientList, (coefficientIter + 1) * sizeof(char *));
				if(Poly.CoefficientList == NULL){
					fprintf(stderr, "ERROR: Out of memory");
					controlled_exit(EXIT_BAD);
				}
			}
			Poly.CoefficientList[coefficientIter] = copy_substring(cutstr, iterstr);
			Poly.Coefficients++;
			coefficientIter++;
			while(isspace(*iterstr)){
				iterstr++;
			}
			cutstr = iterstr;
		}
		else{
			iterstr++;
		}
	}
	return Poly;
}

char * expressionfrompoly(Poly_t input)
{
	//start calculating the different permutations of the polynomial control term(s)
	permutation_t thisperm;
	thisperm.numterms = 0;
	permutation_t * permlist;
	permlist = (permutation_t *) tmalloc(input.Coefficients * sizeof(permutation_t));
	if(permlist == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	permlist[0].numterms = thisperm.numterms;
	int permsdone = 1;
	int attempts = 1;
	thisperm.numterms = 1;
	thisperm.ExpressionIterator = (int *) tmalloc(thisperm.numterms * sizeof(int));
	if(thisperm.ExpressionIterator == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	thisperm.ExpressionIterator[0] = 0;
	while(permsdone < input.Coefficients){
		bool validperm = TRUE;
		for(int i = 0; i < permsdone; i++){
			if(compare(permlist[i], thisperm, input.Dimensions)==0){
				validperm = FALSE;
				break;
			}
		}
		if(validperm){
			permlist[permsdone].numterms = thisperm.numterms;
			permlist[permsdone].ExpressionIterator = (int *) tmalloc(permlist[permsdone].numterms * sizeof(int));
			if(permlist[permsdone].ExpressionIterator == NULL){
				fprintf(stderr, "ERROR: Out of memory");
				controlled_exit(EXIT_BAD);
			}
			for(int i = 0; i < thisperm.numterms; i++){
				permlist[permsdone].ExpressionIterator[i] = thisperm.ExpressionIterator[i];
			}
			permsdone++;
		}
		if(permsdone < input.Coefficients){
			thisperm = increment(thisperm, input.Dimensions);
			attempts++;
		}
	}
	tfree(thisperm.ExpressionIterator);
	// Start forming the expression
	int currentsize = 256;
	char * expression;
	expression = (char *) tmalloc(currentsize * sizeof(char));
	if(expression == NULL){
		fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
	}
	strcpy(expression, input.CoefficientList[0]);
	// add permutations and coefficients to expression
	for(int i = 1; i < input.Coefficients; i++){
		int newlength = strlen(expression) + strlen(input.CoefficientList[i]) + (8 * sizeof(char));
		if(newlength >= currentsize){
			expression = (char *) trealloc(expression, newlength);
			if(expression == NULL){
				fprintf(stderr, "ERROR: Out of memory");
				controlled_exit(EXIT_BAD);
			}
			currentsize = newlength;
		}
		strcat(expression, " + ");
		strcat(expression, "((");
		strcat(expression, input.CoefficientList[i]);
		strcat(expression, ")");
		for(int j = 0; j < permlist[i].numterms; j++){
			newlength = ( strlen(expression) + strlen(input.ExpressionList[permlist[i].ExpressionIterator[j]]) + 4 ) * sizeof(char);
			if(newlength >= currentsize){
				expression = (char *) trealloc(expression, newlength);
				if(expression == NULL){
					fprintf(stderr, "ERROR: Out of memory");
					controlled_exit(EXIT_BAD);
				}
				currentsize = newlength;
			}
			strcat(expression, "*");
			strcat(expression, "(");
			strcat(expression, input.ExpressionList[permlist[i].ExpressionIterator[j]]);
			strcat(expression, ")");
		}
		strcat(expression, ")");
	}
	tfree(permlist);
	//TODO: tfree() various pointers
	return expression;
}

struct line * translatepoly(struct line * input_line)
{
	char * linestr;
	linestr = input_line->li_line;
	// check if translation needed
	if(strstr(linestr, "poly") == NULL){
		return input_line;
	}
	// not returned from function so translate
	char controlchar = '\0', sourcechar = '\0';
	switch(linestr[0]){
		case 'e':	controlchar = 'v';	sourcechar = 'v';	break;
		case 'f':	controlchar = 'i';	sourcechar = 'i';	break;
		case 'g':	controlchar = 'v';	sourcechar = 'i';	break;
		case 'h':	controlchar = 'i';	sourcechar = 'v';	break;
		//default:  return input_line;
	}
	Poly_t Poly = interpretpoly(input_line, controlchar, sourcechar);
	struct line * output_line;
	if(TRANSLATEPOLY_REPLACE){
		output_line->li_actual = input_line->li_actual;
		output_line->li_error = input_line->li_error;
		output_line->li_linenum = input_line->li_linenum;
		output_line->li_linenum_orig = input_line->li_linenum_orig;
		output_line->li_next = input_line->li_next;
	}
    char * sourcename;
    int newlinelen;
    for(int i = 0; linestr[i] != '\0'; i++){
        if(isspace(linestr[i])){
            sourcename = copy_substring(linestr, &linestr[i]);
            newlinelen = i + 7;
            break;
        }
    }
    output_line->li_line = (char *) tmalloc( newlinelen * sizeof(char) );
    if(output_line->li_line == NULL){
        fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
    }
    strcpy(output_line->li_line, "b");
    strcat(output_line->li_line, sourcename);
    char * expressionLHS, * expressionRHS;
    expressionLHS = tprintf(" %c = ", sourcechar);
    strcat(output_line->li_line, expressionLHS);
    expressionRHS = expressionfrompoly(Poly);
    newlinelen += strlen(expressionRHS);
    output_line->li_line = (char *) trealloc(output_line->li_line, newlinelen);
    if(output_line->li_line == NULL){
        fprintf(stderr, "ERROR: Out of memory");
		controlled_exit(EXIT_BAD);
    }
    strcat(output_line->li_line, expressionRHS);
    return output_line;
}
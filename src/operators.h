#ifndef OPERATORS_H
#define OPERATORS_H

#include "gpLevel.h"
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>

class laplacianOperation
{
    public:

    laplacianOperation() {}
    void define( const amrex::Vector<level> & levels);


    void apply( amrex::Vector<level> & levelsOld,  amrex::Vector<level> & levelsNew );


    private:

    std::shared_ptr<amrex::MLPoisson> ML;
    
};


#endif
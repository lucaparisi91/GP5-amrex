#ifndef OPERATORS_H
#define OPERATORS_H

#include "gpLevel.h"
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>

namespace gp
{
    class laplacianOperation
    {
        public:

        laplacianOperation() {}
        void define( const amrex::Vector<level> & levels);

        void define( const levels & initLevels);

        void apply( amrex::Vector<level> & levelsOld,  amrex::Vector<level> & levelsNew );

        void apply( levels & levelsOld,  levels &  levelsNew );


        private:

        std::shared_ptr<amrex::MLPoisson> ML;
    };


    class functional
    {
        public:

        virtual void apply( levels & levelsOld , levels & levelsNew)=0;
    };



    class trappedVortex : public functional
    {
        public:

        trappedVortex(real_t g , std::array<real_t,AMREX_SPACEDIM> omega);

        void addVortex( const std::array<real_t,AMREX_SPACEDIM> & x);
        
        void apply( levels & levelsOld , levels & levelsNew);


        void define(levels & initLevels);

        
        private:

        real_t _g;
        std::array<real_t,AMREX_SPACEDIM> _prefactor;
        std::vector<std::array<real_t,AMREX_SPACEDIM> > _vortexCenters;
        laplacianOperation _lap;
    };


}


#endif
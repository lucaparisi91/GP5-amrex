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

        void define( const baseLevels & initLevels);

        void apply( baseLevels & levelsOld,  baseLevels &  levelsNew );
        
        private:

        std::shared_ptr<amrex::MLPoisson> ML;
    };


    class functional
    {
        public:

        virtual void apply( realLevels & levelsOld , realLevels & levelsNew)=0;
        virtual void apply( complexLevels & levelsOld , complexLevels & levelsNew)=0;

    };





    class trappedVortex : public functional
    {
        public:

        trappedVortex(real_t g , std::array<real_t,AMREX_SPACEDIM> omega);

        void addVortex( const std::array<real_t,AMREX_SPACEDIM> & x);


        void apply( realLevels & levelsOld , realLevels & levelsNew);
        void apply( complexLevels & levelsOld , complexLevels & levelsNew);


        void define( baseLevels & initLevels);

        
        private:

        real_t _g;
        std::array<real_t,AMREX_SPACEDIM> _prefactor;
        std::vector<std::array<real_t,AMREX_SPACEDIM> > _vortexCenters;
        laplacianOperation _lap;
    };


}


#endif
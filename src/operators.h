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

        void apply( levels & levelsOld,  levels &  levelsNew );

        private:

        std::vector<std::shared_ptr<amrex::MLPoisson> > ML;
        

    };


    class functional
    {
        public:

        virtual void apply( realWaveFunction & waveOld , realWaveFunction & waveNew){throw std::runtime_error("apply not defined on a real wavefunction for this functional"); }
        virtual void apply( complexWaveFunction & levelsOld , complexWaveFunction & levelsNew){throw std::runtime_error("apply not defined on a complex wavefunction for this functional"); };


        
        auto & getLaplacianOperator() {return _lap;}

        virtual void define(levels & initLevels) { _lap.define(initLevels); }

        private:


        
        laplacianOperation _lap;    

    };


    class trappedVortex : public functional
    {
        public:

        trappedVortex(real_t g , std::array<real_t,AMREX_SPACEDIM> omega);

        void addVortex( const std::array<real_t,AMREX_SPACEDIM> & x);

        void apply( realWaveFunction & levelsOld , realWaveFunction & levelsNew);
        void apply( complexWaveFunction & levelsOld  , complexWaveFunction & levelsNew );

        private:

        real_t _g;
        std::array<real_t,AMREX_SPACEDIM> _prefactor;
        std::vector<std::array<real_t,AMREX_SPACEDIM> > _vortexCenters;
    };
    

    class LHYDroplet : public functional
    {
        public:

        LHYDroplet(){};

       virtual void apply( complexWaveFunction & levelsOld , complexWaveFunction & levelsNew) override;


    };


}


#endif
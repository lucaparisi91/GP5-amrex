#include "operators.h"
#include <complex>


namespace gp
{


    class constraint
    {
        public:

        virtual void apply(complexWaveFunction & wave ){};

    };


    class normalizationConstraint : public constraint
    {
        public:

        normalizationConstraint( std::vector<real_t> N ) : _N(N) {}

        virtual void apply(complexWaveFunction & wave1 ) override
        {
            wave1.getPhi().averageDown();

            assert( wave1.getNSpecies() == _N.size() );

            for( int c=0; c< wave1.getNSpecies() ; c++ )
            {
                wave1.normalize(_N[c], c );
            }


        }


        private:
        std::vector<real_t> _N;

    };


    class timeStepper
    {
        public:

        timeStepper();


        virtual void advanceImaginaryTime( realWaveFunction & oldLevels , realWaveFunction & newLevels , real_t dt )=0;

        virtual void advanceImaginaryTime(complexWaveFunction & oldWave, complexWaveFunction & newWave, real_t dt )=0;

        virtual void advanceRealTime(complexWaveFunction & oldWave, complexWaveFunction & newWave, real_t dt ) = 0;

        virtual void define(complexWaveFunction & waveOld ) {}

        void applyConstraint( complexWaveFunction & wave  )
        {
            for (auto & constraint : _constraints)
            {
                constraint->apply( wave );
            }
        }

        void addConstraint( std::shared_ptr<constraint>  cons2)
        {
            _constraints.push_back(cons2);
        }

        auto &  getConstraints(int c) { return *_constraints[c]; }
        

        private:

        std::vector<std::shared_ptr<constraint> > _constraints;
    };


    class euleroTimeStepper : public timeStepper
    {
        public:    

        using complex_t = std::complex<real_t>;


        euleroTimeStepper(std::shared_ptr<functional> f ) : _func(f) {}


        virtual void advanceImaginaryTime( realWaveFunction & oldLevels , realWaveFunction & newLevels , real_t dt ) override;

        virtual void advanceImaginaryTime(complexWaveFunction & oldWave, complexWaveFunction & newWave, real_t dt ) override;

        virtual void advanceRealTime(complexWaveFunction & oldWave, complexWaveFunction & newWave, real_t dt ) override ;

        private:
        std::shared_ptr<functional> _func;
    };


    class RK4TimeStepper : public timeStepper
    {
        public:

        using complex_t = std::complex<real_t>;

        RK4TimeStepper(std::shared_ptr<functional> f ) : _func(f) {}
        virtual void define( complexWaveFunction & wave   ) override
        {
            k1.define(wave);
            k2.define(wave);
        }

        virtual void advanceRealTime( complexWaveFunction & oldLevels , complexWaveFunction & newLevels, real_t dt ) override;

        void advanceImaginaryTime( complexWaveFunction & oldLevels , complexWaveFunction & newLevels, real_t dt ) override;

        void advanceImaginaryTime( realWaveFunction & oldLevels , realWaveFunction & newLevels, real_t dt ) override { throw std::runtime_error("advance imaginary time on real wavefunction not implemented"); };



        private:

        std::shared_ptr<functional> _func;


        void evaluateRealTime( complexWaveFunction & oldWave , complexWaveFunction & newWave, real_t dt);

        void evaluateImaginaryTime( complexWaveFunction & oldWave , complexWaveFunction & newWave, real_t dt);


        complexWaveFunction k1;
        complexWaveFunction k2;


    };



};
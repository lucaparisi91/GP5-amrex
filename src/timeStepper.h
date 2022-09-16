#include "operators.h"

namespace gp
{
    class euleroTimeStepper
    {
        public:    

        euleroTimeStepper(std::shared_ptr<functional> f ) : _func(f) {}
        void advance( realWaveFunction & oldLevels , realWaveFunction & newLevels, real_t dt );

        private:
        std::shared_ptr<functional> _func;
    };

};
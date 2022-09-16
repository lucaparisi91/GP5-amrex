#include "timeStepper.h"


namespace gp
{

    void euleroTimeStepper::advance( realWaveFunction & oldWave, realWaveFunction & newWave , real_t dt )
    {
        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();

        _func->apply(oldWave,newWave);
        
        for(int lev=0;lev<oldLevels.size();lev++)
        {
            const auto & phiOld = oldLevels[lev].getMultiFab();
            auto & phiNew = newLevels[lev].getMultiFab();

            phiNew.mult( -dt);

            phiNew.plus(phiOld,0 , oldLevels[0].getNComponents() ,0);

            oldLevels[lev].increaseTime(dt);
        }   
    }  

    

};
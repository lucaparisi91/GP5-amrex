#include "timeStepper.h"



namespace gp
{
    void euleroTimeStepper::advance( levels & oldLevels, levels & newLevels, real_t dt )
    {
        _func->apply(oldLevels,newLevels);

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
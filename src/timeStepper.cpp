#include "timeStepper.h"


namespace gp
{

    timeStepper::timeStepper()
    {

    }


    void euleroTimeStepper::advanceImaginaryTime( complexWaveFunction & oldWave, complexWaveFunction & newWave , real_t dt )
    {
        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();

        _func->apply(oldWave,newWave);

        for(int lev=0;lev<oldLevels.size();lev++)
        {
            for(int c=0;c<oldLevels.getNComponents();c++)
            {
                const auto & phiOld = oldLevels[lev].getMultiFab(c);

                auto & phiNew = newLevels[lev].getMultiFab(c);
                
                phiNew.mult( -dt);

                phiNew.plus(phiOld,0 , 1 ,0);
            }
            newLevels[lev].increaseTime(dt);

        }

        applyConstraint(newWave);

    }
    void euleroTimeStepper::advanceImaginaryTime( realWaveFunction & oldWave, realWaveFunction & newWave , real_t dt )
    {
        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();

        _func->apply(oldWave,newWave);

        for(int lev=0;lev<oldLevels.size();lev++)
        {
            for(int c=0;c<oldLevels.getNComponents();c++)
            {
                const auto & phiOld = oldLevels[lev].getMultiFab(c);

                auto & phiNew = newLevels[lev].getMultiFab(c);
                
                phiNew.mult( -dt);

                phiNew.plus(phiOld,0 , 1 ,0);
            }
            newLevels[lev].increaseTime(dt);

        }   
    }

    void euleroTimeStepper::advanceRealTime( complexWaveFunction & oldWave, complexWaveFunction & newWave , real_t dt )
    {

        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();

        _func->apply(oldWave,newWave);

        for(int lev=0;lev<oldLevels.size();lev++)
        {
            for(int c=0;c<oldWave.getNSpecies();c++)
            {
                newLevels[lev].swapComponents(2*c,2*c+1);
                auto & phiRealNew = newLevels[lev].getMultiFab(2*c);
                auto & phiImagNew = newLevels[lev].getMultiFab(2*c+1);
                auto & phiRealOld = oldLevels[lev].getMultiFab(2*c);
                auto & phiImagOld = oldLevels[lev].getMultiFab(2*c+1);

                phiImagNew.mult(-dt);
                phiRealNew.mult(dt);

                phiRealNew.plus(phiRealOld,0 , 1 ,0);
                phiImagNew.plus(phiImagOld,0 , 1 ,0);

            }
            newLevels[lev].increaseTime(dt);

        }   

        applyConstraint(newWave);
        newLevels.averageDown();


        
    }

    void add( waveFunction & wave1, waveFunction & wave2, waveFunction & waveOut,real_t C )
    {
        auto & levels1 = wave1.getPhi();
        auto & levels2 = wave2.getPhi();
        auto & levelsOut = waveOut.getPhi();

        for(int lev=0;lev<levels1.size();lev++)
        {
            for(int c=0;c<wave1.getNComponents();c++)
            {
                auto & phi1 = levels1[lev].getMultiFab(c);
                auto & phi2 = levels2[lev].getMultiFab(c);
                auto & phiOut = levelsOut[lev].getMultiFab(c);

                for ( MFIter mfi(phi1); mfi.isValid(); ++mfi )
                {
                    const Box& vbx = mfi.validbox();
                    auto const& phi1Arr = phi1.array(mfi);
                    auto const& phi2Arr = phi2.array(mfi);
                    auto const& phiOutArr = phiOut.array(mfi);

                    amrex::ParallelFor(vbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        phiOutArr(i,j,k)=phi1Arr(i,j,k) + C*phi2Arr(i,j,k); 

                    });
                }

                

                
            }
        }   

        levelsOut.averageDown();

    }
    

    void RK4TimeStepper::advanceImaginaryTime( complexWaveFunction & oldWave , complexWaveFunction & newWave, real_t dt )
    {

        evaluateImaginaryTime(oldWave,k2,dt);     
        add(oldWave,k2,newWave,1./6);

        add( oldWave,k2,k1,1./2 );
        applyConstraint(k1);
        evaluateImaginaryTime(k1,k2,dt);
        add(newWave,k2,newWave,1./3);

        add( oldWave,k2,k1,1./2 );
        applyConstraint(k1);
        evaluateImaginaryTime(k1,k2,dt);
        add(newWave,k2,newWave,1./3);

        add( oldWave,k2,k1,1 );
        applyConstraint(k1);
        evaluateImaginaryTime(k1,k2,dt);
        add(newWave,k2,newWave,1./6);

        newWave.getPhi().increaseTime(dt);

        applyConstraint(newWave);

    }

    void RK4TimeStepper::advanceRealTime( complexWaveFunction & oldWave , complexWaveFunction & newWave, real_t dt )
    {
        evaluateRealTime(oldWave,k2,dt);     
        add(oldWave,k2,newWave,1./6);

        add( oldWave,k2,k1,1./2 );
        applyConstraint(k1);
        evaluateRealTime(k1,k2,dt);
        add(newWave,k2,newWave,1./3);

        add( oldWave,k2,k1,1./2 );
        applyConstraint(k1);
        evaluateRealTime(k1,k2,dt);
        add(newWave,k2,newWave,1./3);

        add( oldWave,k2,k1,1 );
        applyConstraint(k1);
        evaluateRealTime(k1,k2,dt);
        add(newWave,k2,newWave,1./6);

        newWave.getPhi().increaseTime(dt);

        applyConstraint(newWave);
        
        
    }

    void RK4TimeStepper::advanceRealTime( complexWaveFunction & oldWave , complexWaveFunction & newWave, real_t dt );


    void RK4TimeStepper::evaluateRealTime( complexWaveFunction & oldWave , complexWaveFunction & newWave , real_t dt )
    {

        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();

        _func->apply(oldWave,newWave);

        for(int lev=0;lev<oldLevels.size();lev++)
        {
            for(int c=0;c<oldWave.getNSpecies();c++)
            {
                newLevels[lev].swapComponents(2*c,2*c+1);
                auto & phiRealNew = newLevels[lev].getMultiFab(2*c);
                auto & phiImagNew = newLevels[lev].getMultiFab(2*c+1);

                phiImagNew.mult( -dt );
                phiRealNew.mult( dt );
            }
        }   

    }


    void RK4TimeStepper::evaluateImaginaryTime( complexWaveFunction & oldWave , complexWaveFunction & newWave , real_t dt )
    {

        auto & oldLevels = oldWave.getPhi();
        auto & newLevels = newWave.getPhi();


        _func->apply(oldWave,newWave);

        for(int lev=0;lev<oldLevels.size();lev++)
        {
            for(int c=0;c<oldWave.getNComponents();c++)
            {
                auto & phiNew = newLevels[lev].getMultiFab(c);
                
                phiNew.mult( -dt ,0 , 1);

            }
        }



    }




};
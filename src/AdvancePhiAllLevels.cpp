#include "AmrCoreAdv.H"

#include <AMReX_MultiFabUtil.H>

using namespace amrex;

// advance all levels for a single time step
void
gp::AmrCoreAdv::AdvancePhiAllLevels (Real time, Real dt_lev, int /*iteration*/)
{
    swapOldNewFields();
    _lap.apply( getOldLevels(),getNewLevels() );
    
    for (int lev = 0; lev <= finest_level; lev++)
    {

        auto & phiNew = fieldNew[lev].getMultiFab();
        auto & phiOld = fieldOld[lev].getMultiFab();
        auto & geo= fieldNew[lev].getGeometry();
        const auto right= geo.ProbDomain().hi();
        const auto left= geo.ProbDomain().lo();
        auto  dx = geo.CellSizeArray();



        for ( MFIter mfi(phiNew); mfi.isValid(); ++mfi )
        {
            const Box& vbx = mfi.validbox();
            auto const& phiNewArr = phiNew.array(mfi);
            auto const& phiOldArr = phiOld.array(mfi);

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];

                phiNewArr(i,j,k)=phiOldArr(i,j,k) - dt_lev*( - 0.5* phiNewArr(i,j,k) + ( 0.5*(x*x + y*y) + 100000*phiOldArr(i,j,k)*phiOldArr(i,j,k) +0.5*1/(x*x + y*y) )*phiOldArr(i,j,k) )  ;
            });
        
        }
    }

}

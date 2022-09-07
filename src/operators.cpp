#include "operators.h"
namespace gp
{


void laplacianOperation::define( const amrex::Vector<level> & levels)
{
    auto geoms=getGeometry(levels);
    auto dms=getDistributionMapping(levels);
    auto bas=getBoxArray(levels);
    
    ML=std::make_shared<amrex::MLPoisson>();
    ML->define( geoms,bas, dms  );
    ML->setDomainBC( {AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)} , { AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)}   );
    ML->setMaxOrder(3);
};

void laplacianOperation::define( const levels & initLevels )
{
    auto geoms=initLevels.getGeometry();
    auto dms=initLevels.getDistributionMapping();
    auto bas=initLevels.getBoxArray();
    
    ML=std::make_shared<amrex::MLPoisson>();
    ML->define( geoms,bas, dms  );
    ML->setDomainBC( {AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)} , { AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)}   );
    ML->setMaxOrder(3);
};




void laplacianOperation::apply( amrex::Vector<level> & levelsOld,  amrex::Vector<level> & levelsNew )
{
    assert(levelsOld.size()==levelsNew.size() );
    auto src= getMultiFabsPtr(levelsOld);
    auto dst = getMultiFabsPtr(levelsNew);

    
    for(int lev=0;lev<src.size();lev++)
    {
        ML->setLevelBC(lev,src[lev]);
    }

    amrex::MLMG mlmg(*ML);
    mlmg.apply(dst,src);

}

void laplacianOperation::apply(  levels & levelsOld,  levels &  levelsNew )
{
    assert(levelsOld.size()==levelsNew.size() );
    auto src= levelsOld.getMultiFabsPtr();
    auto dst = levelsNew.getMultiFabsPtr();

    for(int lev=0;lev<src.size();lev++)
    {
        ML->setLevelBC(lev,src[lev]);
    }

    amrex::MLMG mlmg(*ML);
    mlmg.apply(dst,src);

}


trappedVortex::trappedVortex(real_t g, std::array<real_t,AMREX_SPACEDIM>  omega) : _g(g)
{

    for(int d=0;d<AMREX_SPACEDIM;d++)
    {
        _prefactor[d]=0.5*omega[d]*omega[d];
    }
}

void trappedVortex::define(levels & initLevels)
{
    _lap.define(initLevels);
}

void trappedVortex::addVortex( const std::array<real_t,AMREX_SPACEDIM> & x)
{
    _vortexCenters.push_back(x);
}

void trappedVortex::apply( levels & fieldOld , levels & fieldNew)
{
    _lap.apply(fieldOld,fieldNew);

    for (int lev = 0; lev < fieldOld.size(); lev++)
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

                phiNewArr(i,j,k)= - 0.5* phiNewArr(i,j,k) + ( _prefactor[0]*(x*x) + _prefactor[1]*y*y + _g*phiOldArr(i,j,k)*phiOldArr(i,j,k)  )*phiOldArr(i,j,k)   ;

                for(int iV=0;iV<_vortexCenters.size();iV++)
                {
                    const auto & x0 = _vortexCenters[iV];

                    auto r2 = (x - x0[0] )*(x-x0[0] ) + (y-x0[1])*(y-x0[1]) ;
                    phiNewArr(i,j,k)+=0.5/( r2 ) * phiOldArr(i,j,k) ;
                }
            });
        }
    }


}


}

#include "operators.h"

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

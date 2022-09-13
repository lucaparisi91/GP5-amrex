#include "operators.h"
namespace gp
{

    void laplacianOperation::define( const baseLevels & initLevels )
    {
        auto geoms=initLevels.getGeometry();
        auto dms=initLevels.getDistributionMapping();
        auto bas=initLevels.getBoxArray();
        ML.resize( initLevels.getNComponents() ); 

        for(int c=0;c<initLevels.getNComponents();c++)
        {
            ML[c]=std::make_shared<amrex::MLPoisson>();
            ML[c]->define( geoms,bas, dms  );
            ML[c]->setDomainBC( {AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)} , { AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)}   );
            ML[c]->setMaxOrder(3);
        }

    };

    void laplacianOperation::apply(  baseLevels & levelsOld,  baseLevels &  levelsNew )
    {
        for(int i=0;i<levelsOld.getNComponents();i++)
        {
            assert(levelsOld.size()==levelsNew.size() );
            auto src= levelsOld.getMultiFabsPtr(i);
            auto dst = levelsNew.getMultiFabsPtr(i);

            for(int lev=0;lev<src.size();lev++)
            {
                ML[i]->setLevelBC(lev,src[lev]);
            }

            amrex::MLMG mlmg(*ML[i]);
            mlmg.apply(dst,src);
        }

       
    }


    trappedVortex::trappedVortex(real_t g, std::array<real_t,AMREX_SPACEDIM>  omega) : _g(g)
    {

        for(int d=0;d<AMREX_SPACEDIM;d++)
        {
            _prefactor[d]=0.5*omega[d]*omega[d];
        }
    }

    void trappedVortex::define( baseLevels & initLevels)
    {
        _lap.define(initLevels);
    }

    void trappedVortex::addVortex( const std::array<real_t,AMREX_SPACEDIM> & x)
    {
        _vortexCenters.push_back(x);
    }

    void trappedVortex::apply( realLevels & fieldOld , realLevels & fieldNew)
    {
        assert(fieldOld.getNSpecies()==1 );
        assert(fieldNew.getNSpecies()==1 );

        _lap.apply( fieldOld, fieldNew );

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


    void trappedVortex::apply( complexLevels & fieldOld , complexLevels & fieldNew)
    {
        assert(fieldOld.getNSpecies()==1 );
        assert(fieldNew.getNSpecies()==1 );

        _lap.apply( fieldOld, fieldNew );

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

                for (int c=0;c<fieldOld.getNSpecies();c++)
                amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto x = left[0] + (i + 0.5)* dx[0];
                    auto y = left[1] + (j + 0.5 )* dx[1];

                    auto V = ( _prefactor[0]*(x*x) + _prefactor[1]*y*y + _g*(
                        phiOldArr(i,j,k,2*c)*phiOldArr(i,j,k,2*c) +
                        phiOldArr(i,j,k,2*c + 1)*phiOldArr(i,j,k,2*c + 1) 
                    
                        ) ); 
                    
                    phiNewArr(i,j,k,2*c)= - 0.5* phiNewArr(i,j,k,2*c) + 
                    V*phiOldArr(i,j,k,2*c)   ;
                    phiNewArr(i,j,k,2*c + 1)= - 0.5* phiNewArr(i,j,k,2*c + 1) + 
                    V*phiOldArr(i,j,k,2*c + 1) ;


                });
            }
        }


    }

}

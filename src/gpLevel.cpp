#include "gpLevel.h"

geometry_t createGeometry( amrex::Array<real_t,AMREX_SPACEDIM> left,amrex::Array<real_t,AMREX_SPACEDIM> right,amrex::Array<size_t,AMREX_SPACEDIM> shape   )
{
    geometry_t geom;
    Vector<int> is_periodic(AMREX_SPACEDIM,0);
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            is_periodic[idim] = 1;
        }
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL( shape[0]-1, shape[1]-1, shape[2]-1));
    Box domain(dom_lo, dom_hi);

    amrex::RealBox real_box({AMREX_D_DECL(left[0],left[1],left[2])},
                         {AMREX_D_DECL( right[0], right[1], right[2])});

    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic.data() );

    return geom;
};

void initGaussian(level & currentLevel, real_t alpha)
{
    auto & phi = currentLevel.getMultiFab();
    auto & geom = currentLevel.getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            
            const Box& vbx = mfi.validbox();
            auto const& phiArr = phi.array(mfi);
            const auto left= geom.ProbDomain().lo();
            const auto right= geom.ProbDomain().hi();

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];
            #if AMREX_SPACE_DIM > 2
                auto z = left[2] + k* dx[2];
            #endif
                auto r2 = x*x + y*y;
            #if AMREX_SPACE_DIM > 2
                r2 += z*z;
            #endif
                phiArr(i,j,k)=exp( - alpha*r2);
                if ( std::abs(phiArr(i,j,k))<0.01 and (std::abs(x)<dx[0]) and (std::abs(y)<dx[1] )   )
                {
                    std::cout << phiArr(i,j,k) << std::endl;
                    phiArr(i,j,k)=1;
                }

            });
        
        }
}


void level::updateDensity()
{
    auto & phi = getMultiFab();
    auto & geom = getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            const Box& vbx = mfi.validbox();
            auto const& phiArr = phi.array(mfi);
            auto const& densityArr = getDensity().array(mfi);

            const auto left= geom.ProbDomain().lo();
            const auto right= geom.ProbDomain().hi();

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];
            #if AMREX_SPACE_DIM > 2
                auto z = left[2] + k* dx[2];
            #endif
                auto r2 = x*x + y*y;
            #if AMREX_SPACE_DIM > 2
                r2 += z*z;
            #endif
                densityArr(i,j,k)=phiArr(i,j,k) * phiArr(i,j,k);
            });
        
        }
}


real_t level::getNorm()
{
    /* !!!!!   requires the density to have been updated */
    auto & density = getDensity();

    auto norm = density.sum();

    ParallelDescriptor::ReduceRealSum( norm );

    real_t dV=1;
    auto dx = getGeometry().CellSizeArray();

    for(int d=0;d< AMREX_SPACEDIM;d++)
    {
        dV *= dx[d];
    }


    return norm*dV;
};


amrex::Vector<amrex::Geometry> getGeometry(const amrex::Vector<level> & levels)
{
    amrex::Vector<amrex::Geometry> geom;

    for( int lev=0;lev<levels.size();lev++)
    {
        auto & level = levels[lev];
        if (level.isValid() )
        {
            geom.push_back(levels[lev].getGeometry() );
        }
        
    };

    return geom;
}


amrex::Vector<amrex::BoxArray> getBoxArray( const amrex::Vector<level> & levels )
{
    amrex::Vector<amrex::BoxArray> bas;

    for( int lev=0;lev<levels.size();lev++)
    {
        auto & level = levels[lev];
        if (level.isValid() )
        {
            bas.push_back(levels[lev].getBoxArray() );
        }
        
    };

    return bas;
}


amrex::Vector<amrex::DistributionMapping> getDistributionMapping(const amrex::Vector<level> & levels)
{
    amrex::Vector<amrex::DistributionMapping> dms;

    for( int lev=0;lev<levels.size();lev++)
    {
        auto & level = levels[lev];
        if (level.isValid() )
        {
            dms.push_back(levels[lev].getDistributionMap() );
        }
    };

    return dms;
}

amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( amrex::Vector<level> & levels )
{
    amrex::Vector<amrex::MultiFab*> fabs;

    for( int lev=0;lev<levels.size();lev++)
    {
        auto & level = levels[lev];
        if (level.isValid() )
        {
            fabs.push_back( &(levels[lev].getMultiFab()) );
        }
    };

    return fabs;
}

amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( const amrex::Vector<level> & levels )
{
    amrex::Vector<const amrex::MultiFab*> fabs;

    for( int lev=0;lev<levels.size();lev++)
    {
        auto & level = levels[lev];
        if (level.isValid() )
        {
            fabs.push_back( &(levels[lev].getMultiFab()) );
        }
    };

    return fabs;
};





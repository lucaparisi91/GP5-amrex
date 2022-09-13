#include "gpLevel.h"
#include <filesystem>

namespace gp
{

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

void initGaussian(level & currentLevel, real_t alpha,int iComp )
{
    auto & phi = currentLevel.getMultiFab(iComp);
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
                

            });
        
        }
}


void level::saveHDF5( const std::string & filename, size_t i) const
{
    WriteSingleLevelPlotfileHDF5(filename,
                               getMultiFab(i),
                               {varnames()[i]},
                               getGeometry(),
                               getTime(),
                               0);
}

void level::saveHDF5( const std::string & dirname) const
{
    std::filesystem::create_directories(dirname);
    fs::path dirPath(dirname);
    const auto & names = varnames();
    namespace fs = std::filesystem;
    fs::create_directories(dirPath);
    for(int i=0;i<getNComponents();i++)
    {
        fs::path filePath = dirPath / fs::path( names[i] ) ;
        
        saveHDF5( filePath.string(),i);
    }

}


void realLevel::updateDensity(size_t i)
{
    auto & phi = getMultiFab(i);
    auto & geom = getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            const Box& vbx = mfi.validbox();
            auto const& phiArr = phi.array(mfi);
            auto const& densityArr = getDensity(i).array(mfi);

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



void complexLevel::updateDensity(  size_t i )
{
    auto & phiReal = getMultiFab(2*i);
    auto & phiImag = getMultiFab(2*i+1);

    auto & geom = getGeometry();
    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    {
        for ( MFIter mfi(phiReal); mfi.isValid(); ++mfi )
        {
            const Box& vbx = mfi.validbox();
            auto const& phiRealArr = phiReal.array(mfi);
            auto const& phiImagArr = phiImag.array(mfi);

            auto const& densityArr = getDensity(i).array(mfi);

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
                densityArr(i,j,k)=phiRealArr(i,j,k) * phiRealArr(i,j,k) + phiImagArr(i,j,k) * phiImagArr(i,j,k) ;
            });

        }
    }

}

real_t level::getNorm( int iSpecies)
{

    updateDensity(iSpecies);

    /* !!!!!   requires the density to have been updated */
    auto & density = getDensity(iSpecies);

    auto norm = density.sum();



    real_t dV=1;
    auto dx = getGeometry().CellSizeArray();

    for(int d=0;d< AMREX_SPACEDIM;d++)
    {
        dV *= dx[d];
    }

    return norm*dV;
};




template<class level_t >
levels<level_t>::levels( std::vector<std::shared_ptr<level_t> > levels_ ) : _levels(levels_)   {

    _finestLevel=_levels.size() - 1;
    _refRatios.resize(  _levels.size() - 1 );
    for(int lev=0;lev<_finestLevel;lev++)
    {
        const auto & geoCoarse = _levels[lev]->getGeometry();
        const auto & geoFine = _levels[lev+1]->getGeometry();
        const auto & shapeCoarse= geoCoarse.Domain().length();
        const auto & shapeFine= geoFine.Domain().length();

        for(int d=0;d<AMREX_SPACEDIM;d++)
        {
            _refRatios[lev][d] = shapeFine[d]/shapeCoarse[d];

            if (shapeFine[d] % shapeCoarse[d] != 0 )
            {
                throw std::runtime_error("Refinement ratio between levels should be an integer");
            }
        }

    }
}



template<class level_t>
amrex::Vector< amrex::MultiFab* > levels<level_t>::getMultiFabsPtr( size_t i  )
{
    amrex::Vector<amrex::MultiFab*> fabs;

    for( int lev=0;lev<size();lev++)
    {        
        fabs.push_back( &( _levels[lev]->getMultiFab(i)) );
    };

    return fabs;
}


template<class level_t>
amrex::Vector< const amrex::MultiFab* > levels<level_t>::getMultiFabsPtr( size_t i ) const
{
    amrex::Vector< const amrex::MultiFab*> fabs;

    for( int lev=0;lev<size();lev++)
    {        
        fabs.push_back( &( _levels[lev]->getMultiFab(i)) );
    };

    return fabs;
};


template<class level_t>
amrex::Vector<amrex::BoxArray> levels<level_t >::getBoxArray( ) const
{
    amrex::Vector< amrex::BoxArray> bas;

    for( int lev=0;lev<size();lev++)
    {        
        bas.push_back( _levels[lev]->getBoxArray() ) ;
    };

    return bas;
};


template<class level_t>
amrex::Vector<amrex::DistributionMapping> levels< level_t >::getDistributionMapping( ) const
{
    amrex::Vector< amrex::DistributionMapping> dms;
    for( int lev=0;lev<size();lev++)
    {        
        dms.push_back( _levels[lev]->getDistributionMap() ) ;
    };

    return dms;
};

template<class level_t >
amrex::Vector< amrex::Geometry > levels<level_t>::getGeometry( ) const
{
    amrex::Vector< amrex::Geometry> geoms;
    for( int lev=0;lev<size();lev++)
    {        
        geoms.push_back( _levels[lev]->getGeometry() ) ;
    };
    return geoms;
};

template<class level_t>
amrex::Vector< real_t >  levels<level_t>::getTimes( ) const
{
    amrex::Vector< real_t> times;
    for( int lev=0; lev<size(); lev++)
    {        
        times.push_back( _levels[lev]->getTime() ) ;
    };
    return times;
};


template<class level_t>
void levels<level_t>::averageDown ()
{
    for (int lev = size()-2 ; lev >= 0; --lev)
    {

        for(int c=0;c<_levels[lev]->getNComponents();c++)
        {
            auto & phiCoarse = _levels[lev]->getMultiFab(c);
            const auto & phiFine = _levels[lev + 1]->getMultiFab(c);
            auto & geomCoarse = _levels[lev]->getGeometry();
            const auto & geomFine = _levels[lev + 1]->getGeometry();

            const auto & ref =  _refRatios[lev] ;
            amrex::IntVect ratio{ AMREX_D_DECL(ref[0],ref[1],ref[2])} ;        

            amrex::average_down( phiFine, phiCoarse,
                                geomFine, geomCoarse,
                                0, 1 , ratio );
            
            auto & densityCoarse = _levels[lev]->getDensity(c);
            const auto & densityFine = _levels[lev + 1]->getDensity(c);

            amrex::average_down( densityFine, densityCoarse,
                                geomFine, geomCoarse,
                                0, 1 , ratio );
        }
    }

}

template<class level_t>
void levels<level_t>::save(const std::string & dirname) const 
{
    auto geoms= getGeometry();
    amrex::Vector<int> iStep( _levels.size() , 0 );

    const auto &  names = _levels[0]->varnames();
    fs::create_directories(dirname);

    for(int i=0;i<_levels[0]->getNComponents();i++)
    {
        const auto& mf = getMultiFabsPtr( i );

        fs::path filePath = fs::path(dirname) / fs::path( names[i] ) ;

        amrex::WriteMultiLevelPlotfileHDF5(filePath.string(),size(),mf,{names[i] },geoms,_levels[0]->getTime(),iStep, _refRatios );
    }

}


template<>
void levels<gp::realLevel>::normalize( real_t N , int c ) 
{
    for(int lev=0;lev<size();lev++)
    {
        _levels[lev]->updateDensity(c);
    }

    averageDown();

    auto oldN = _levels[0]->getNorm(c);

    auto C = sqrt(N/oldN);

    for(int lev=0;lev<size();lev++)
    {
        
        
        auto & phi = _levels[lev]->getMultiFab(c);
        phi.mult(C,0,1);

    }

}

template<>
void levels<gp::complexLevel>::normalize( real_t N , int c ) 
{
    for(int lev=0;lev<size();lev++)
    {
        _levels[lev]->updateDensity(c);
    }
    
    averageDown();

    auto oldN = _levels[0]->getNorm(c);

    auto C = sqrt(N/oldN);

    for(int lev=0;lev<size();lev++)
    {
        auto & phiR = _levels[lev]->getMultiFab(c);
        phiR.mult(C,0,1);
        auto & phiI = _levels[lev]->getMultiFab(c+1);
        phiI.mult(C,0,1);
        
    }

}







template<class level_t>
void levels<level_t>::normalize( const std::vector<real_t>  & N) 
{
    assert( N.size() == getNSpecies() );

    for(int c=0;c<getNSpecies();c++)
    {
        normalize(N[c],c);
    }
}

std::vector<real_t> level::getNorm()
{
    std::vector<real_t> norms( getNSpecies());
    for( int c=0; c<getNSpecies(); c++ )
    {
        norms[c]=getNorm(c);
    }
    return norms;

}


template class levels<realLevel>;
template class levels<complexLevel>;


}
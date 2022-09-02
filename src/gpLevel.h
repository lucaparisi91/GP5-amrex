
#ifndef GPLEVEL_H
#define GPLEVEL_H


#include <AMReX.H>
#include <filesystem>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>

namespace fs = std::filesystem;
using namespace amrex;
using geometry_t = Geometry;
using real_t = Real;



class level
{
    public:

    level( geometry_t geom, BoxArray ba, DistributionMapping dm ) : level()
    {
        define(geom,ba,dm);
    }


    level() : _time(0) , _invalid(true) {};

    void define( geometry_t geom, BoxArray ba, DistributionMapping dm )
    {
        if (not _invalid)
        {
            throw std::runtime_error("Level should be invalidated first.");
        }

        _geom=geom;_ba=ba;_dm=dm;
        _phi.define(_ba,_dm,_nComponents,_nGrow);
        _density.define(_ba,_dm,_nComponents,_nGrow);
        
        _invalid=false;
    }

    auto &     getMultiFab() {return _phi;};
    const auto &     getMultiFab() const  {return _phi;};

    auto &  getGeometry() {return _geom; };
    const auto &  getGeometry() const {return _geom; };


    auto & getDistributionMap() {return _dm; }    
    const auto & getDistributionMap() const {return _dm; }    


    auto & getBoxArray() {return _ba; }
    const auto & getBoxArray()  const {return _ba; }

    void saveToFile(const std::string filename)
    {
        WriteSingleLevelPlotfileHDF5(filename, _phi, {"phi"}, _geom, _time, 0);
    }

    void setTime( real_t tNew) {_time=tNew;}
    auto getTime() const {return _time; };
    
    auto getNComponents() const {return _nComponents;}

    void clear() { _phi.clear() ; _density.clear();  _invalid=true; }

    auto isValid() const {return !_invalid;}

    auto & getDensity() { return _density;}
    const auto & getDensity() const  { return _density;}

    void updateDensity();

    real_t  getNorm();


    private:
    geometry_t  _geom;
    MultiFab _phi;
    MultiFab _density;
    BoxArray _ba;
    DistributionMapping _dm;
    amrex::Array<int,AMREX_SPACEDIM> _refinement;
    int _nGrow=2;
    int _nComponents=1;
    std::string name = "phi";
    real_t _time;
    bool _invalid;
};



geometry_t createGeometry( amrex::Array<real_t,AMREX_SPACEDIM> left,amrex::Array<real_t,AMREX_SPACEDIM> right,amrex::Array<size_t,AMREX_SPACEDIM> shape   );

void initGaussian(level & currentLevel, real_t alpha);

amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( const amrex::Vector<level> & levels );

amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( amrex::Vector<level> & levels );



amrex::Vector<amrex::BoxArray> getBoxArray( const amrex::Vector<level> & levels );
amrex::Vector<amrex::DistributionMapping> getDistributionMapping(const amrex::Vector<level> & levels);

amrex::Vector<amrex::Geometry> getGeometry(const amrex::Vector<level> & levels);

amrex::Vector< real_t > getTimes( const amrex::Vector<level> & levels );


#endif

#ifndef GPLEVEL_H
#define GPLEVEL_H


#include <AMReX.H>
#include <filesystem>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>

namespace fs = std::filesystem;
using namespace amrex;
using geometry_t = Geometry;
using real_t = Real;



namespace gp{


class level
{
    public:

    level( geometry_t geom, BoxArray ba, DistributionMapping dm, int nComponents=1 ) : level() 
    {
        define(geom,ba,dm,nComponents);
    }

    level() : _time(0) , _invalid(true) , _nComponents(1)  {};


     virtual int getNSpecies() const = 0;
    auto getNComponents() const {return _nComponents;}


    virtual void define( geometry_t geom, BoxArray ba, DistributionMapping dm , int nComponents = 1 )
    {
        _nComponents=nComponents;

        if (not _invalid)
        {
            throw std::runtime_error("Level should be invalidated first.");
        }

    
        _geom=geom;_ba=ba;_dm=dm;
        _phi.define(_ba,_dm,_nComponents,_nGrow);
        
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

    virtual amrex::Vector<std::string> varnames() const = 0 ;


    virtual void updateDensity()=0;

    virtual amrex::MultiFab & getDensity()=0;
    virtual const amrex::MultiFab & getDensity() const =0;

    void setTime( real_t tNew) {_time=tNew;}

    void increaseTime(real_t dt ) {_time+=dt;}

    auto getTime() const {return _time; };
    

    virtual void clear() { _phi.clear() ;   _invalid=true; }

    auto isValid() const {return !_invalid;}

    void saveHDF5(const std::string & filename) const;


    real_t  getNorm( int iSpecies);
    



    std::vector<real_t> getNorm();



    private:
    geometry_t  _geom;
    MultiFab _phi;

    BoxArray _ba;
    DistributionMapping _dm;
    amrex::Array<int,AMREX_SPACEDIM> _refinement;
    int _nComponents;
    real_t _time;
    bool _invalid;
    protected:
    int _nGrow=2;

};


class realLevel : public level
{
    public:

    realLevel( geometry_t geom, BoxArray ba, DistributionMapping dm, int nSpecies=1 ) :  realLevel() { 
        define(geom,ba,dm,nSpecies);
     }
    
    realLevel() : level( ) {_nSpecies=0;} ;


    virtual int getNSpecies() const override {return _nSpecies; }

    virtual void define( geometry_t geom, BoxArray ba, DistributionMapping dm , int nSpecies=1) override
    {
        _nSpecies=nSpecies;
        level::define(geom,ba,dm,nSpecies);
        _density.define(ba,dm,nSpecies,_nGrow);
    }


    virtual amrex::Vector<std::string> varnames() const override
    {
        amrex::Vector<std::string> names;
        names.resize( getNComponents() );

        for(int iComp=0;iComp<getNComponents();iComp++)
        {
            names[iComp]=std::string("phi") + std::to_string(iComp) ;
        }

        return names;

    }

    virtual void clear() override
    {
        level::clear();
        _density.clear();
    }

    virtual void updateDensity() override;

     virtual amrex::MultiFab & getDensity() override { return _density;}
    const amrex::MultiFab & getDensity() const override  { return _density;}


    private:

    MultiFab _density;
    int _nSpecies;
};

class complexLevel : public level
{
    public:

    complexLevel( geometry_t geom, BoxArray ba, DistributionMapping dm, int nSpecies=1 ) : _nSpecies(_nSpecies), level::level(geom,ba,dm,nSpecies*2) {_nSpecies=nSpecies;}

    complexLevel() : level( ) {_nSpecies=0;} ;


    virtual int getNSpecies() const override {return _nSpecies; }


    virtual void define( geometry_t geom, BoxArray ba, DistributionMapping dm , int nSpecies=1) override
    {
        _nSpecies=nSpecies;
        level::define(geom,ba,dm,nSpecies*2);
        _density.define(ba,dm,nSpecies,_nGrow);
        _phase.define(ba,dm,nSpecies,_nGrow);
    }

    virtual amrex::Vector<std::string> varnames() const override
    {
        amrex::Vector<std::string> names;
        names.resize( getNComponents() );

        for(int iComp=0;iComp<getNSpecies();iComp++)
        {

            names[2*iComp]=std::string("phi") + std::to_string(iComp) + std::string("Real");
            names[2*iComp + 1 ]=std::string("phi") + std::to_string(iComp) + std::string("Img");
        }
        assert( names.size() == getNComponents() );

        return names;

    }

    virtual amrex::MultiFab & getDensity() override { return _density;}
    const amrex::MultiFab & getDensity() const override  { return _density;}


    virtual void updateDensity() override;


    private:

    MultiFab _density;
    MultiFab _phase;
    int _nSpecies;
};





geometry_t createGeometry( amrex::Array<real_t,AMREX_SPACEDIM> left,amrex::Array<real_t,AMREX_SPACEDIM> right,amrex::Array<size_t,AMREX_SPACEDIM> shape   );


void initGaussian( level & currentLevel, real_t alpha, int iComp=0);


class baseLevels
{
    public :


    virtual amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( ) const = 0 ;
    

    virtual amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( ) = 0;

    virtual amrex::Vector<amrex::BoxArray> getBoxArray( ) const = 0;
    
    virtual amrex::Vector<amrex::DistributionMapping> getDistributionMapping() const = 0;


    virtual amrex::Vector<amrex::Geometry> getGeometry() const = 0 ;

    virtual amrex::Vector< real_t > getTimes() const = 0 ;


    virtual int size() const = 0;

};



template<class level_t>
class levels : public  baseLevels
{

    public:

    levels() : _finestLevel(-1) {}


    
    levels( std::vector<std::shared_ptr<level_t> > levels_);


    const auto & operator[](size_t i) const {return *_levels[i] ; }
    auto & operator[](size_t i) {return *_levels[i] ; }

    int size() const {return _finestLevel+1 ; }

    amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( ) const ;

    amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( );

    amrex::Vector<amrex::BoxArray> getBoxArray( ) const ;
    
    amrex::Vector<amrex::DistributionMapping> getDistributionMapping() const;


    amrex::Vector<amrex::Geometry> getGeometry() const ;

    amrex::Vector< real_t > getTimes() const ;



    void averageDown() ;

    const auto &  getRefRatio(int lev ) const  {return _refRatios[lev];}

    void save(const std::string & filename) const ;

    void normalize(real_t N, int c);
    void normalize( const std::vector<real_t> & v);    


    void updateDensity()
    {
        for(int lev=0;lev<size();lev++)
        {
            _levels[lev]->updateDensity();
        }
    }

    auto  getNorm(int c) {return _levels[0]->getNorm(c); }

    auto  getNorm() {return _levels[0]->getNorm(); }




    auto getTime(){return _levels[0]->getTime();}

    int getNSpecies() const  {return _levels[0]->getNSpecies(); }
    private:


    std::vector<std::shared_ptr<level_t> > _levels;
    amrex::Vector< IntVect > _refRatios ;
    int _finestLevel;

};


using realLevels = levels<realLevel> ;
using complexLevels = levels<complexLevel> ;



}


#endif

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
        _phi.resize( nComponents);
        for(int i=0;i<nComponents;i++)
        {
            _phi[i].define(_ba,_dm,1,_nGrow);
        }

        _invalid=false;
    }

    auto &     getMultiFabs() {return _phi;};
    const auto &     getMultiFabs() const  {return _phi;};


    auto &     getMultiFab( size_t i = 0 ) {return _phi[i];};
    const auto &     getMultiFab( size_t i=0 ) const  {return _phi[i];};

    auto & operator[](size_t i ) {return getMultiFab(i);}
    const auto & operator[](size_t i ) const {return getMultiFab(i);}



    auto &  getGeometry() {return _geom; };
    const auto &  getGeometry() const {return _geom; };


    auto & getDistributionMap() {return _dm; }    
    const auto & getDistributionMap() const {return _dm; }    


    auto & getBoxArray() {return _ba; }
    const auto & getBoxArray()  const {return _ba; }

    virtual amrex::Vector<std::string> varnames() const = 0 ;


    virtual void updateDensity(size_t i = 0  )=0;


    virtual void updateDensity()
    {
        for(int i=0;i<getNSpecies();i++)
        {
            updateDensity(i);
        }
    }

    virtual amrex::MultiFab & getDensity(size_t i = 0)=0;
    virtual const amrex::MultiFab & getDensity(size_t i = 0 ) const =0;

    void setTime( real_t tNew) {_time=tNew;}

    void increaseTime(real_t dt ) {_time+=dt;}

    auto getTime() const {return _time; };

    virtual void clear() {
        for(int i=0;i<getNComponents();i++)
        {
            _phi[i].clear() ;
        }
        _invalid=true; 

    }

    auto isValid() const {return !_invalid;}

    void saveHDF5(const std::string & filename,size_t i ) const;

    void saveHDF5(const std::string & dirname) const  ;

    real_t  getNorm( int iSpecies);
    



    std::vector<real_t> getNorm();


    private:
    geometry_t  _geom;
    std::vector<MultiFab> _phi;


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
        _density.resize(_nSpecies);
        for(int i=0;i<nSpecies;i++)
        {
            _density[i].define(ba,dm,1,_nGrow);
        }

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
        for(int i=0;i<_nSpecies;i++)
        {
            _density[i].clear();
        }
    }

    virtual void updateDensity(size_t i = 0 ) override;


     virtual amrex::MultiFab & getDensity(size_t i=0 ) override { return _density[i];}
    const amrex::MultiFab & getDensity( size_t i = 0 ) const override  { return _density[i];}


    private:

    std::vector<MultiFab> _density;
    int _nSpecies;
};



class complexLevel : public level
{
    public:

    complexLevel( geometry_t geom, BoxArray ba, DistributionMapping dm, int nSpecies=1 ) : level() {define(geom,ba,dm,nSpecies); }

    complexLevel() : level( ) {_nSpecies=0;} ;


    virtual int getNSpecies() const override {return _nSpecies; }


    virtual void define( geometry_t geom, BoxArray ba, DistributionMapping dm , int nSpecies=1) override
    {
        _nSpecies=nSpecies;
        level::define(geom,ba,dm,nSpecies*2);
        _density.resize(nSpecies);
        _phase.resize(nSpecies);

        for(int i=0;i<nSpecies;i++)
        {
            _density[i].define(ba,dm,1,_nGrow);
            _phase[i].define(ba,dm,1,_nGrow);
        }


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

    auto & getRealMultiFab(int iSpec) { return getMultiFab(2*iSpec) ;}
    const auto & getRealMultiFab(int iSpec) const { return getMultiFab(2*iSpec) ;}

    auto & getImagMultiFab(int iSpec) { return getMultiFab(2*iSpec + 1) ; }
    const auto & getImagMultiFab(int iSpec) const { return getMultiFab(2*iSpec + 1) ; }


    
   
    

    virtual amrex::MultiFab & getDensity(size_t i = 0 ) override { return _density[i];}
    const amrex::MultiFab & getDensity(size_t i = 0 ) const override  { return _density[i];}





    virtual void updateDensity(size_t i = 0 ) override;



    private:

    std::vector<MultiFab> _density;
    std::vector<MultiFab> _phase;
    int _nSpecies;
};





geometry_t createGeometry( amrex::Array<real_t,AMREX_SPACEDIM> left,amrex::Array<real_t,AMREX_SPACEDIM> right,amrex::Array<size_t,AMREX_SPACEDIM> shape   );


void initGaussian( level & currentLevel, real_t alpha, int iComp=0);


class baseLevels
{
    public :


    virtual amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( size_t i = 0 ) const = 0 ;
    

    virtual amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( size_t i = 0 ) = 0;

    virtual amrex::Vector<amrex::BoxArray> getBoxArray( ) const = 0;
    
    virtual amrex::Vector<amrex::DistributionMapping> getDistributionMapping() const = 0;


    virtual amrex::Vector<amrex::Geometry> getGeometry() const = 0 ;

    virtual amrex::Vector< real_t > getTimes() const = 0 ;


    virtual int size() const = 0;

    virtual int getNComponents() const = 0;


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

    amrex::Vector< const amrex::MultiFab* > getMultiFabsPtr( size_t i = 0 ) const ;

    amrex::Vector< amrex::MultiFab* > getMultiFabsPtr( size_t i = 0 );

    amrex::Vector<amrex::BoxArray> getBoxArray( ) const ;
    
    amrex::Vector<amrex::DistributionMapping> getDistributionMapping() const;


    amrex::Vector<amrex::Geometry> getGeometry() const ;

    amrex::Vector< real_t > getTimes() const ;



    void averageDown() ;

    const auto &  getRefRatio(int lev ) const  {return _refRatios[lev];}

    void save(const std::string & filename) const ;

    void normalize(real_t N, int c);
    void normalize( const std::vector<real_t> & v);


    int getNComponents() const override {return _levels[0]->getNComponents();}

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


template<>
void levels<gp::realLevel>::normalize( real_t N , int c ) ;




}


#endif
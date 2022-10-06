
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

        level() : _time(0) , _nComponents(1)  {};

        
        auto getNComponents() const {return _nComponents;}

        void define(const level & level2);


        virtual void define( geometry_t geom, BoxArray ba, DistributionMapping  dm , int nComponents = 1 )
        {
            _geom=geom;_ba=ba;_dm=dm;

            resizeNComponents(nComponents);

            

        }

        void resizeNComponents(int nComponents);

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

        


        void setTime( real_t tNew) {_time=tNew;}

        void increaseTime(real_t dt ) {_time+=dt;}

        auto getTime() const {return _time; };

        virtual void clear() {
            for(int i=0;i<getNComponents();i++)
            {
                _phi[i].clear() ;
            }

        }

        
        void saveHDF5(const std::string & filename,size_t i ) const;

        void saveHDF5(const std::string & dirname) const  ;

        real_t  getNorm( int iSpecies);
        
        
        void setNames( const amrex::Vector<std::string> & names) { _names=names;
        }


        const auto & getNames() const {return _names; }


         void swapComponents(int i1, int i2)
        {
            std::swap(_phi[i1] , _phi[i2] );
        }


        private:
        geometry_t  _geom;
        std::vector<MultiFab> _phi;

        BoxArray _ba;
        DistributionMapping _dm;
        amrex::Array<int,AMREX_SPACEDIM> _refinement;
        int _nComponents;
        real_t _time;
        

        protected:
        int _nGrow=2;
        amrex::Vector<std::string> _names;
        
    };


geometry_t createGeometry( amrex::Array<real_t,AMREX_SPACEDIM> left,amrex::Array<real_t,AMREX_SPACEDIM> right,amrex::Array<size_t,AMREX_SPACEDIM> shape   );


void initGaussian( level & currentLevel, real_t alpha, int iComp=0);

class levels
{   
    
    public:
    using level_t = level;

    levels() : _finestLevel(-1) {}

    levels( std::vector<std::shared_ptr<level_t> > levels_)
    {
        define(levels_);
    }



    void resizeNComponents(int nComponents);

    void define(const levels & levels2);
    void define( std::vector<std::shared_ptr<level_t> > levels_);


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


    int getNComponents() const {return _levels[0]->getNComponents();}

    auto getTime(){return _levels[0]->getTime();}

    void setTime( real_t t0);


    void increaseTime(real_t dt );



   
    private:


    std::vector<std::shared_ptr<level_t> > _levels;
    amrex::Vector< IntVect > _refRatios ;
    int _finestLevel;
    
};



void updateDensity(level & phiLevel, level & densityLevel , int c);
void updateDensity(level & phiLevel, level & densityLevel , int c1 , int c2, int c3);



class waveFunction
{
    public:

    auto getNSpecies() {return _density->getNComponents(); }

    auto getNComponents() {return _phi->getNComponents(); }


    real_t getNorm( int c);

    const auto & getPhi() const {return *_phi ; }
    const auto & getDensity() const  {return *_density ; }

    auto & getPhi() {return *_phi ; }
    auto & getDensity() {return *_density ; }


    void updateDensity();
    virtual void normalize(real_t N , int c)=0;
    virtual void updateDensity( int c )=0;

    std::vector<real_t> getNorm();

    auto getTime() const {return (*_phi)[0].getTime()  ; }

    void setTime(real_t t) {_phi->setTime(t); }


    protected:

    std::shared_ptr<levels> _phi;
    std::shared_ptr<levels> _density;
};



class realWaveFunction : public waveFunction
{
    public:

    realWaveFunction(){}


    realWaveFunction(  std::shared_ptr<levels> waveLevels );


    void define( const realWaveFunction & wave );


    virtual void updateDensity( int ) override;

    virtual void normalize(real_t N,int c) override;


    private:
   
};


class complexWaveFunction : public waveFunction
{
    public:

    complexWaveFunction(){}

    complexWaveFunction(  std::shared_ptr<levels> waveLevels );

    virtual void updateDensity( int ) override;

    virtual void normalize(real_t N,int c) override;

    void define( complexWaveFunction & wave );


    private:

    levels _phase;
};


}


#endif
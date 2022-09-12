


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mpi.h>
#include <pybind11/stl.h>
#include <AMReX.H>
#include <AMReX.H>
#include "../src/gpLevel.h"
#include "../src/operators.h"
#include "../src/timeStepper.h"

namespace py = pybind11;
using real_t = double;

namespace pyInterfaceAmreX
{


    void initialize()
    {
        int argc = 0;
        char ** argv = nullptr;
        amrex::Initialize( argc,argv  );   
    };

    void finalize()
    {
        amrex::Finalize(  );
    }

    class geometry
    {
        public:

        geometry( std::array<real_t,AMREX_SPACEDIM> left,std::array<real_t,AMREX_SPACEDIM> right , std::array<size_t,AMREX_SPACEDIM> shape  ) : 
        _left(left),_right(right),_shape(shape)
        {
            geom=gp::createGeometry(left,right,shape);
        }

        const auto & getLeft() const {return _left;}
        const auto & getRight() const {return _right;}
        const auto & getShape() const {return _shape;}

        auto & getGeometry() {return geom; }
        const auto & getGeometry() const {return geom; }
        


        private:

        std::array<real_t,AMREX_SPACEDIM> _left;
        std::array<real_t,AMREX_SPACEDIM> _right;
        std::array<size_t,AMREX_SPACEDIM> _shape;

        geometry_t geom;
    };

    class box
    {
        public:

        box() {}

        box( const std::array<size_t,AMREX_SPACEDIM> & left , const std::array<size_t,AMREX_SPACEDIM> & right)
        {
          define(left,right);
        }

        void define( const std::array<size_t,AMREX_SPACEDIM> & left , const std::array<size_t,AMREX_SPACEDIM> & right)
        {
            IntVect low( AMREX_D_DECL(left[0],left[1],left[2]));
            IntVect hi( AMREX_D_DECL(right[0],right[1],right[2]));
            _box=amrex::Box( low, hi      );
        }



        std::array<int,AMREX_SPACEDIM>  getLow() const 
        {
            const auto & low = _box.smallEnd();
            std::array<int,AMREX_SPACEDIM> lowArr {AMREX_D_DECL( low[0],low[1],low[2]) };
            return lowArr;
        };

        std::array<int,AMREX_SPACEDIM>  getHigh() const 
        {
            const auto & hi = _box.bigEnd();
            std::array<int,AMREX_SPACEDIM> highArr {AMREX_D_DECL( hi[0],hi[1],hi[2]) };
            return highArr;
        };

        auto & getBox() {return _box;}


        private:

        amrex::Box _box;
    };

    template<class level_t>
    class level
    {
        public:

        level( const geometry & geo, std::vector<box> & boxes, int nComponents )
        {

            auto & geoCpp = geo.getGeometry();

            amrex::BoxList bl;

            boxes.resize(boxes.size());
            for(int i=0;i<boxes.size();i++)
            {
                bl.push_back( boxes[i].getBox() );
            }

            amrex::BoxArray ba(bl);
            amrex::DistributionMapping dm(ba);

            _level=std::make_shared<level_t>(geoCpp,ba,dm,nComponents);

        }

        void save( std::string filename) 
        {
            _level->saveHDF5(filename);
        }

        void setData(const py::array_t<double> & phiPy, box & pyBox)
        {
            // set up python array
            py::buffer_info info = phiPy.request();
            if (info.shape.size() != AMREX_SPACEDIM + 1 )
            {
                throw std::runtime_error(std::string("Numpy array should have rank ") + std::to_string(AMREX_SPACEDIM + 1));
            };
            auto phiArrPy = phiPy.unchecked<AMREX_SPACEDIM + 1>();

            const auto & box = pyBox.getBox();


            auto & phi = _level->getMultiFab();
            int nc= _level->getNComponents();
                

            for ( amrex::MFIter mfi(phi); mfi.isValid(); ++mfi )
            {
                       
                const Box& vbx = mfi.validbox();
                auto const& phiArr = phi.array(mfi);
                auto & lo = vbx.smallEnd();
                
                if (vbx == box) {

                      amrex::ParallelFor(vbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for(int c=0 ; c< nc ; c++)
                        {
                            phiArr(i,j,k,c)=phiArrPy( AMREX_D_DECL(i-lo[0],j-lo[1],k - lo[2]) , c );
                        }
                        
                    });

                    return;

                }

            }

        }


        auto getNorm(  ) const  { return _level->getNorm(); }


        auto getData(box & pyBox)
        {
            const auto & box = pyBox.getBox();

            auto & phi = _level->getMultiFab();

            int nc = _level->getNComponents();


            for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
            {
                const Box& vbx = mfi.validbox();
                auto const& phiArr = phi.array(mfi);
                auto & lo = vbx.smallEnd();
                const auto & shape = vbx.length();


                if (vbx == box) {
                     py::array_t<double> phiPy( { AMREX_D_DECL( shape[0],shape[1],shape[2] ) , nc } ) ;
                     auto phiArrPy = phiPy.mutable_unchecked<AMREX_SPACEDIM + 1 >();

                      amrex::ParallelFor(vbx,
                        [&] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for(int c=0;c<nc;c++)
                        {
                            phiArrPy( AMREX_D_DECL(i-lo[0],j-lo[1],k - lo[2]) ,c )=phiArr(i,j,k,c);
                        }
                        
                    });

                    return phiPy;
                }

            }
            // if box not found returns an empty array
             py::array_t<double> phiPy( { AMREX_D_DECL((int)0, (int)0, (int)0) } ) ;
             return phiPy;
        }

        auto getLevelPtr() {return _level; }

        private:

        std::shared_ptr<level_t> _level;


    };


    using realLevel=level<gp::realLevel>;
    using complexLevel=level<gp::complexLevel>;


    template<class level_t>
    class field
    {
        public:

        field(std::vector<level<level_t > > & levels)
        {
            
            std::vector<std::shared_ptr<level_t> > levelsPtr;

            for ( auto & level : levels)
            {
                levelsPtr.push_back( level.getLevelPtr() );
            }

            _levels=std::make_shared<gp::levels<level_t> >(levelsPtr);

        }

        void averageDown()
        {
            _levels->averageDown();
        }

        void save( const std::string & filename)
        {
            _levels->save(filename);
        }

        auto getLevelsPtr() {return _levels; }

        auto & getLevels() {return *_levels;}

        void normalize(const std::vector<real_t>  & N ) { _levels->normalize(N) ; }

        std::vector<real_t> getNorm() { return _levels->getNorm() ;   }

        
        private:

        std::shared_ptr<gp::levels<level_t>  > _levels;
    };

    using realField=field<gp::realLevel> ;
    using complexField=field<gp::complexLevel> ;


    class functional
    {
        public:

        virtual void apply( realField & levelsOld , realField & levelsNew)=0;

        virtual std::shared_ptr<gp::functional> getFunctional()=0;


    };


    class trappedVortex : public functional
    {
        public:

        trappedVortex(real_t g , std::array<real_t,AMREX_SPACEDIM> omega)  {
            _trappedVortex=std::make_shared<gp::trappedVortex>(g,omega); 
        }


        void addVortex( const std::array<real_t,AMREX_SPACEDIM> & x)
        {
            _trappedVortex->addVortex(x);
        }

        void apply( realField & levelsOld , realField & levelsNew)
        {
            _trappedVortex->apply( levelsOld.getLevels(),levelsNew.getLevels() );
        }

        void define( realField & initLevels )
        {
            _trappedVortex->define( initLevels.getLevels()   );
        }



        std::shared_ptr<gp::functional> getFunctional() {return _trappedVortex; }

        private:

        std::shared_ptr<gp::trappedVortex> _trappedVortex;

    };

    class stepper
    {
        public:

        using stepper_t = gp::euleroTimeStepper;

        stepper( trappedVortex & func )
        {
            _stepper=std::make_shared<stepper_t>( func.getFunctional() );
        };


        void advance( realField & oldLevels, realField & newLevels,real_t dt)
        {
            _stepper->advance(oldLevels.getLevels(),newLevels.getLevels(),dt);
        }

        private:

        std::shared_ptr<stepper_t> _stepper;
    };

};

PYBIND11_MODULE(gpAmreX, m) {
     py::class_<pyInterfaceAmreX::geometry>(m, "geometry")
     .def(py::init< std::array<real_t,AMREX_SPACEDIM> , std::array<real_t,AMREX_SPACEDIM> , std::array<size_t,AMREX_SPACEDIM> >() )
     .def("getLeft",&pyInterfaceAmreX::geometry::getLeft)
      .def("getRight",&pyInterfaceAmreX::geometry::getRight)
     .def("getShape",&pyInterfaceAmreX::geometry::getShape)
     ;
      m.def("initialize", &pyInterfaceAmreX::initialize, "Initialize amrex");
    m.def("finalize", &pyInterfaceAmreX::finalize, "Finalize amrex");

    py::class_<pyInterfaceAmreX::box>(m, "box")
     .def(py::init<  const std::array<size_t,AMREX_SPACEDIM> & ,  const std::array<size_t,AMREX_SPACEDIM> & >() )
     .def("getLow",&pyInterfaceAmreX::box::getLow)
      .def("getHigh",&pyInterfaceAmreX::box::getHigh);
    

    py::class_<pyInterfaceAmreX::level<gp::realLevel> >(m, "realLevel")
     .def(py::init<  const pyInterfaceAmreX::geometry & , std::vector<pyInterfaceAmreX::box> &  , int >() )  
     .def("setData",&pyInterfaceAmreX::level<gp::realLevel>::setData)
     .def("getData",&pyInterfaceAmreX::level<gp::realLevel>::getData) 
     .def("save",&pyInterfaceAmreX::level<gp::realLevel>::save) 
     .def("getNorm",&pyInterfaceAmreX::level<gp::realLevel>::getNorm) 
     ;



    py::class_<pyInterfaceAmreX::field<gp::realLevel> >(m, "field")
     .def(py::init<  std::vector<pyInterfaceAmreX::realLevel > &  >() )
     .def("save",&pyInterfaceAmreX::realField::save)
     .def("averageDown",&pyInterfaceAmreX::realField::averageDown)
     .def("getNorm",&pyInterfaceAmreX::realField::getNorm)
     .def("normalize",&pyInterfaceAmreX::realField::normalize);

    

    py::class_<pyInterfaceAmreX::functional>(m, "functional" )
    .def("apply",& pyInterfaceAmreX::functional::apply)
    
    ;


    py::class_<pyInterfaceAmreX::trappedVortex,pyInterfaceAmreX::functional>(m, "trappedVortex")
     .def(py::init<  real_t ,  std::array<real_t,AMREX_SPACEDIM>  >() )
     .def("addVortex",&pyInterfaceAmreX::trappedVortex::addVortex)
     .def("apply",& pyInterfaceAmreX::trappedVortex::apply)
     .def("define",& pyInterfaceAmreX::trappedVortex::define )        
     ;

    py::class_<pyInterfaceAmreX::stepper>(m, "stepper")
     .def(py::init<  pyInterfaceAmreX::trappedVortex &  >() )
     .def("advance",& pyInterfaceAmreX::stepper::advance )
    ;

     
 


  ;           
}
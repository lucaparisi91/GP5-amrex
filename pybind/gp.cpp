


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


    class level
    {
        public:

        level( const geometry & geo, std::vector<box> & boxes)
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

            _level=std::make_shared<gp::level>(geoCpp,ba,dm);

        }

        void save( std::string filename) 
        {
            _level->saveHDF5(filename);
        }

        void setData(const py::array_t<double> & phiPy, box & pyBox)
        {
            // set up python array
            py::buffer_info info = phiPy.request();
            if (info.shape.size() != AMREX_SPACEDIM )
            {
                throw std::runtime_error(std::string("Numpy array should have rank ") + std::to_string(AMREX_SPACEDIM));
            };
            auto phiArrPy = phiPy.unchecked<AMREX_SPACEDIM>();

            const auto & box = pyBox.getBox();


            auto & phi = _level->getMultiFab();


            for ( amrex::MFIter mfi(phi); mfi.isValid(); ++mfi )
            {
                       
                const Box& vbx = mfi.validbox();
                auto const& phiArr = phi.array(mfi);
                auto & lo = vbx.smallEnd();

                if (vbx == box) {

                      amrex::ParallelFor(vbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                
                        phiArr(i,j,k)=phiArrPy( AMREX_D_DECL(i-lo[0],j-lo[1],k - lo[2]) );
                    });

                    return;

                }

            }


        }


        auto getNorm() const  { return _level->getNorm();}

        auto getData(box & pyBox)
        {
            const auto & box = pyBox.getBox();

            auto & phi = _level->getMultiFab();

            for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
            {
                const Box& vbx = mfi.validbox();
                auto const& phiArr = phi.array(mfi);
                auto & lo = vbx.smallEnd();
                const auto & shape = vbx.length();

                
                if (vbx == box) {
                     py::array_t<double> phiPy( { AMREX_D_DECL( shape[0],shape[1],shape[2]) } ) ;
                     auto phiArrPy = phiPy.mutable_unchecked<AMREX_SPACEDIM>();

                      amrex::ParallelFor(vbx,
                        [&] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                
                        phiArrPy( AMREX_D_DECL(i-lo[0],j-lo[1],k - lo[2]) )=phiArr(i,j,k);
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

        std::shared_ptr<gp::level> _level;


    };


    class field
    {
        public:

        field(std::vector<level> & levels)
        {

            std::vector<std::shared_ptr<gp::level> > levelsPtr;

            for ( auto & level : levels)
            {
                levelsPtr.push_back( level.getLevelPtr() );
            }

            _levels=std::make_shared<gp::levels>(levelsPtr);

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


        void normalize(real_t N ) { _levels->normalize(N) ; }


        real_t getNorm() { return _levels->getNorm() ;   }


        private:

        std::shared_ptr<gp::levels> _levels;
    };

    class functional
    {
        public:

        virtual void apply( field & levelsOld , field & levelsNew)=0;
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

        void apply( field & levelsOld , field & levelsNew)
        {
            _trappedVortex->apply( levelsOld.getLevels(),levelsNew.getLevels() );
        }

        void define( field & initLevels )
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


        void advance( field & oldLevels, field & newLevels,real_t dt)
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
    
    py::class_<pyInterfaceAmreX::level>(m, "level")
     .def(py::init<  const pyInterfaceAmreX::geometry & , std::vector<pyInterfaceAmreX::box> &  >() )  
     .def("setData",&pyInterfaceAmreX::level::setData)
     .def("getData",&pyInterfaceAmreX::level::getData) 
     .def("save",&pyInterfaceAmreX::level::save) 
     .def("getNorm",&pyInterfaceAmreX::level::getNorm) 

     ;


    py::class_<pyInterfaceAmreX::field>(m, "field")
     .def(py::init<  std::vector<pyInterfaceAmreX::level> &  >() )
     .def("save",&pyInterfaceAmreX::field::save)
     .def("averageDown",&pyInterfaceAmreX::field::averageDown)
     .def("getNorm",&pyInterfaceAmreX::field::getNorm)
     .def("normalize",&pyInterfaceAmreX::field::normalize);


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
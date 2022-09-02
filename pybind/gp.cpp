


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mpi.h>
#include <pybind11/stl.h>
#include <AMReX.H>
#include <AMReX.H>
#include "../src/gpLevel.h"

namespace py = pybind11;
using real_t = double;

namespace pyInterfaceAmreX
{

    class geometry
    {
        public:

        geometry( std::array<real_t,AMREX_SPACEDIM> left,std::array<real_t,AMREX_SPACEDIM> right , std::array<size_t,AMREX_SPACEDIM> shape  ) : 
        _left(left),_right(right),_shape(shape)
        {
            geom=createGeometry(left,right,shape);
        }
        

        const auto & getLeft() const {return _left;}
        const auto & getRight() const {return _right;}
        const auto & getShape() const {return _shape;}

        private:

        std::array<real_t,AMREX_SPACEDIM> _left;
        std::array<real_t,AMREX_SPACEDIM> _right;
        std::array<size_t,AMREX_SPACEDIM> _shape;

        geometry_t geom;


    };

};


PYBIND11_MODULE(gpAmreX, m) {
     py::class_<pyInterfaceAmreX::geometry>(m, "geometry")
     .def(py::init< std::array<real_t,AMREX_SPACEDIM> , std::array<real_t,AMREX_SPACEDIM> , std::array<size_t,AMREX_SPACEDIM> >() )
     .def("getLeft",&pyInterfaceAmreX::geometry::getLeft)
      .def("getRight",&pyInterfaceAmreX::geometry::getRight)
     .def("getShape",&pyInterfaceAmreX::geometry::getShape)
     ;
        
  ;           
}
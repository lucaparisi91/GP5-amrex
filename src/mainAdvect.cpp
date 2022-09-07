#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include "AmrCoreAdv.H"

using namespace amrex;
using real_t = Real;
using namespace gp;


int main(int argc, char* argv[])
{
    argc=0;
    argv = nullptr;
    amrex::Initialize( argc,argv  );

    {
        // timer for profiling
        BL_PROFILE("main()");


        // wallclock time
        const auto strt_total = amrex::second();

        amrex::Array<real_t,AMREX_SPACEDIM> left { AMREX_D_DECL(-25,-25,-25)};
        amrex::Array<real_t,AMREX_SPACEDIM> right { AMREX_D_DECL( 25, 25,25) };

        amrex::RealBox real_box({AMREX_D_DECL(left[0],left[1],left[2])},
                         {AMREX_D_DECL( right[0], right[1], right[2])});
    
        
        AmrCoreAdv amr_core_adv(real_box,{AMREX_D_DECL(64,64,64)}, { { AMREX_D_DECL(2,2,2)},{AMREX_D_DECL(2,2,2)},{AMREX_D_DECL(2,2,2)},{ AMREX_D_DECL(4,4,4)  }  }      );


        // initialize AMR data
        amr_core_adv.InitData();

        //amr_core_adv.WritePlotFile();

        // advance solution to final time
        //amr_core_adv.Evolve();

        // wallclock time
        auto end_total = amrex::second() - strt_total;        

        if (amr_core_adv.Verbose()) {
            // print wallclock time
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
        }

    }

    amrex::Finalize();
}

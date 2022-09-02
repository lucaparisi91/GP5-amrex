
#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include "AmrCoreAdv.H"
#include <filesystem>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>

#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>
#include "gpLevel.h"
#include "operators.h"

using namespace amrex;
int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    int nComp=1;

    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions


    // make BoxArray and geometry_t
    int nLevels=2;
    auto baseGeom = createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });

    Box dom( baseGeom.Domain());

    Geometry geom0(baseGeom);
    BoxArray ba0(dom);
    DistributionMapping dm0(ba0);

    amrex::Vector<level> levelsNew(nLevels);
    amrex::Vector<level> levelsOld(nLevels);

    levelsNew[0].define( geom0,ba0,dm0);
    levelsOld[0].define( geom0,ba0,dm0);

    int offset=(64*2 / (4*2))*2;
    int lCore= (128/8) * 2;
    //int lCore=128;
    //int offset=0;
    

    Geometry geom1(baseGeom);
    geom1.refine({AMREX_D_DECL(2,2,2)});

    Box core( IntVect(AMREX_D_DECL(offset,offset,offset) )  , IntVect(AMREX_D_DECL(offset + lCore -1 ,offset+ lCore - 1,offset+lCore - 1) )   );
    BoxArray ba1(core);
    DistributionMapping dm1(ba1);


    levelsNew[1].define(  geom1,  ba1 , dm1 );
    levelsOld[1].define(  geom1,  ba1 , dm1 );

    auto & level0 = levelsOld[0];
    auto & level1 = levelsOld[1];

    auto & level0_new = levelsNew[0];
    auto & level1_new = levelsNew[1];


  


    real_t alpha=1./(2*0.1*0.1);
    initGaussian(level0,alpha);
    initGaussian(level1,alpha);


    level0.getMultiFab().FillBoundary(   level0.getGeometry().periodicity());
    level1.getMultiFab().FillBoundary(level1.getGeometry().periodicity());

    //amrex::average_down(level1.getMultiFab(), level0.getMultiFab(),
    //                        level0.getGeometry(), level1.getGeometry(),
    //                        0, nComp, 2);

    level0.saveToFile("level0");
    level1.saveToFile("level1");

    auto lap=new laplacianOperation();
    lap->define(levelsNew);
    lap->apply(levelsOld,levelsNew);
    
    level1_new.saveToFile("level1_2");
    level0_new.saveToFile("level0_2");

    delete lap;
    amrex::Finalize();

};
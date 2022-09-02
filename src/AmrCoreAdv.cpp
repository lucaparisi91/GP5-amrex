
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include "AmrCoreAdv.H"

using namespace amrex;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
AmrCoreAdv::AmrCoreAdv(const RealBox& rb,
             const Vector<int>& n_cell_in,
             Vector<IntVect> const& ref_ratios ) :
             AmrCore::AmrCore(rb, ref_ratios.size() , n_cell_in,0,ref_ratios,{AMREX_D_DECL(1,1,1)} )
{

    ReadParameters();

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    if (do_subcycle) {
        for (int lev = 1; lev <= max_level; ++lev) {
            nsubsteps[lev] = MaxRefRatio(lev-1);
        }
    }


    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    // periodic boundaries
    int bc_lo[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};
    int bc_hi[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};
    int nComponents=1;

    bcs.resize( nComponents);     // Setup 1-component

    for(int iC=0;iC< nComponents;iC++)
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            {  // external Drichlet
                bcs[iC].setLo(idim, bc_lo[idim]);
                bcs[iC].setHi(idim, bc_hi[idim]);
            }
        }
    
    fieldNew.resize(nlevs_max);
    fieldOld.resize(nlevs_max);

    verbose=0;

}

AmrCoreAdv::~AmrCoreAdv ()
{

}


// advance solution to final time
void
AmrCoreAdv::Evolve ()
{
    Real cur_time = getNewLevels()[0].getTime();
    int last_plot_file_step = 0;

    auto & levels = getNewLevels();

    int nSubSteps=1000;
    max_step=100;

    ComputeDt();

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << std::endl;
        amrex::Print() << "levels: " << finest_level + 1 << std::endl;

        for(int iiStep=0;iiStep<nSubSteps;iiStep++)
        {
            int lev = 0;
            int iteration = 1;
            if (do_subcycle)
                timeStepWithSubcycling(lev, cur_time, iteration);
            else
                timeStepNoSubcycling(cur_time, iteration);

            cur_time += dt[0];
            
            if (max_level > 0 && regrid_int > 0)  // We may need to regrid
            {
                if (istep[0] % regrid_int == 0)
                {
                    regrid(0,cur_time);
                    _lap.define(getNewLevels());
                    AverageDown();


                }
            }

        }
        // sum phi to check conservation
        Real sum_phi = getNewLevels()[0].getMultiFab().sum();


        amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt[0] << " Sum(Phi) = " << sum_phi << std::endl;

        // sync up time
        for (int lev = 0; lev <= finest_level; ++lev) {
             levels[lev].setTime( cur_time);
        }


        if (plot_int > 0 && (step+1) % plot_int == 0) {
            last_plot_file_step = step+1;
            WritePlotFile(step+1);
        }


        //if (chk_int > 0 && (step+1) % chk_int == 0) {
        //    WriteCheckpointFile();
        //}

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step+1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6*dt[0]) break;
    }

}

// initializes multilevel data
void
AmrCoreAdv::InitData ()
{
    // start simulation from the beginning
    const Real time = 0.0;
    InitFromScratch(time);


    normalize(norm);
    _lap.define(getNewLevels()) ;


}


// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::MakeNewLevelFromCoarse (int lev, Real time, const BoxArray& ba,
                                    const DistributionMapping& dm)
{

    exit(0);
    fieldNew[lev].define( Geom()[lev] , ba , dm );
    fieldOld[lev].define( Geom()[lev] , ba , dm );


    auto n = ba.size();


    
    t_new[lev] = time;
    t_old[lev] = time - 1.e200;


    FillCoarsePatch(lev, time, fieldNew[lev].getMultiFab(), 0, fieldNew[lev].getMultiFab().nComp() );

}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::RemakeLevel (int lev, Real time, const BoxArray& ba,
                         const DistributionMapping& dm)
{

    auto & prevLevel = getNewLevels()[lev];

    level newLevel;
    newLevel.define( geom[lev], ba ,dm );
    newLevel.setTime(time);

    FillPatch(lev, time, newLevel.getMultiFab() , 0, prevLevel.getNComponents());



    std::swap(prevLevel,newLevel);


    getOldLevels()[lev].clear();

    getOldLevels()[lev].define(geom[lev],ba,dm);


}

// Delete level data
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ClearLevel (int lev)
{
    getNewLevels()[lev].clear();
    getOldLevels()[lev].clear();

}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
void AmrCoreAdv::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba,
                                          const DistributionMapping& dm)
{

    fieldNew[lev].clear();
    fieldOld[lev].clear();
    
    fieldNew[lev].define( Geom()[lev] , ba , dm );
    fieldOld[lev].define( Geom()[lev] , ba , dm );

    fieldNew[lev].setTime(time);
    fieldOld[lev].setTime(-200);


    auto n = ba.size();

    real_t alpha=1./(2 * (6*6) );

    initGaussian(fieldNew[lev],alpha);

}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ErrorEst (int lev, TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
//    const int clearval = TagBox::CLEAR;
    const int   tagval = TagBox::SET;
    Vector<real_t> cutOff { 1 , 0.5 , 0.25 , 0.15 };
    const MultiFab& state = fieldNew[lev].getMultiFab();
    int nTags=0;

    const auto & geo = getNewLevels()[lev].getGeometry();
    const auto dx = geo.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {

        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx  = mfi.tilebox();
            
            const auto statefab = state.array(mfi);
            const auto tagfab  = tags.array(mfi);

            const auto left= geo.ProbDomain().lo();
            const auto right= geo.ProbDomain().hi();


            amrex::ParallelFor(bx,
            [&] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];

                bool toTag = (x*x + y*y) < cutOff[lev]*cutOff[lev] ? true : false;
                if (toTag)
                {
                    tagfab(i,j,k)=TagBox::SET;
                    nTags++;
                }

            });
        }
    }
}

// read in some parameters from inputs file
void
AmrCoreAdv::ReadParameters ()
{

    stop_time=100000;

    chk_int=0;
    plot_int=1;
    plot_file="plt";
    regrid_int=0;
    norm=1;


    cfl=0.001;

    do_reflux=0;
    do_subcycle=0;

    SetMaxGridSize(64);

}


// set covered coarse cells to be the average of overlying fine cells
void
AmrCoreAdv::AverageDown ()
{
    auto & levels = getNewLevels();

    for (int lev = finest_level-1; lev >= 0; --lev)
    {
        auto & phiCoarse = levels[lev].getMultiFab();
        const auto & phiFine = levels[lev + 1].getMultiFab();


        amrex::average_down( phiFine, phiCoarse,
                            geom[lev+1], geom[lev],
                            0, levels[0].getNComponents() , refRatio(lev) );
        
        auto & densityCoarse = levels[lev].getDensity();
        const auto & densityFine = levels[lev + 1].getDensity();



        amrex::average_down( densityFine, densityCoarse,
                            geom[lev+1], geom[lev],
                            0, levels[0].getNComponents() , refRatio(lev) );
        
    }
}

void AmrCoreAdv::normalize(real_t N)
{
    for(int lev=0;lev<=finest_level;lev++)
    {
        auto & level = getNewLevels()[lev];
        level.updateDensity();
    }

    AverageDown();

    auto oldN = getNewLevels()[0].getNorm();

    auto C = sqrt(N/oldN);
    for(int lev=0;lev<=finest_level;lev++)
    {
        auto & level = getNewLevels()[lev];
        auto & phi = level.getMultiFab();
        phi.mult( C, 0, level.getNComponents() );
    }


}

// more flexible version of AverageDown() that lets you average down across multiple levels
void
AmrCoreAdv::AverageDownTo (int crse_lev)
{
    amrex::average_down(phi_new[crse_lev+1], phi_new[crse_lev],
                        geom[crse_lev+1], geom[crse_lev],
                        0, phi_new[crse_lev].nComp(), refRatio(crse_lev));
}

// compute a new multifab by coping in phi from valid region and filling ghost cells
// works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)

void
AmrCoreAdv::FillPatch(int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    if (lev == 0)
    {
        auto & oldLevel = getNewLevels()[lev];
        Vector<MultiFab*> smf { &(oldLevel.getMultiFab() ) };
        Vector<Real> stime {  oldLevel.getTime()   };

        if(Gpu::inLaunchRegion())
        {
            /* GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbc(geom[lev],bcs,gpu_bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0) ; */
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0);
        }
    }
    else
    {
        auto & oldLevelFine = getNewLevels()[lev];
        auto & oldLevelCoarse = getNewLevels()[lev-1];

        Vector<MultiFab*> cmf { &(oldLevelCoarse.getMultiFab() ) } , fmf {&(oldLevelFine.getMultiFab() ) } ;
        Vector<Real> ctime {oldLevelCoarse.getTime()}, ftime { oldLevelFine.getTime()};


        Interpolater* mapper = &cell_cons_interp;

        if(Gpu::inLaunchRegion())
        {
            //GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            //PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            //PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            //amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
            //                          0, icomp, ncomp, geom[lev-1], geom[lev],
            //                          cphysbc, 0, fphysbc, 0, refRatio(lev-1),
            //                          mapper, bcs, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

            amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                      0, icomp, ncomp, geom[lev-1], geom[lev],
                                      cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                      mapper, bcs, 0);
        }
    }
}



// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void
AmrCoreAdv::FillCoarsePatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    if(Gpu::inLaunchRegion())
    {
        //GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        //PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        //PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        //amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
        //                             cphysbc, 0, fphysbc, 0, refRatio(lev-1),
        //                             mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void
AmrCoreAdv::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}

// Advance a level by dt
// (includes a recursive call for finer levels)
void
AmrCoreAdv::timeStepWithSubcycling (int lev, Real time, int iteration)
{
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time);
                

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
            }
        }
    }

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE with time = " << t_new[lev]
                       << " dt = " << dt[lev] << std::endl;
    }

    // Advance a single level for a single time step, and update flux registers

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    Real t_nph = t_old[lev] + 0.5*dt[lev];

    //DefineVelocityAtLevel(lev, t_nph);
    //AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsubsteps[lev]);

    ++istep[lev];

    if (Verbose())
    {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            timeStepWithSubcycling(lev+1, time+(i-1)*dt[lev+1], i);
        }

        if (do_reflux)
        {
            // update lev based on coarse-fine flux mismatch
            flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
        }

        AverageDownTo(lev); // average lev+1 down to lev
    }

}

// Advance all the levels with the same dt
void
AmrCoreAdv::timeStepNoSubcycling (Real time, int iteration)
{

   

    {
    

    if ( Verbose() ) {
        for (int lev = 0; lev <= finest_level; lev++)
        {
           amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
           amrex::Print() << "ADVANCE with time = " << t_new[lev]
                          << " dt = " << dt[0] << std::endl;
        }
    }


    AdvancePhiAllLevels (time, dt[0], iteration);

    // Make sure the coarser levels are consistent with the finer levels
    normalize(norm);

    }

    for (int lev = 0; lev <= finest_level; lev++)
        ++istep[lev];

    if (Verbose())
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
            amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
        }
    }


}

// a wrapper for EstTimeStep
void
AmrCoreAdv::ComputeDt ()
{
    for(int lev=0;lev<=finest_level;lev++)
    {
        const auto & dx = Geom()[finest_level].CellSize();
        dt[lev]=cfl * dx[0] * dx[0];
    }
}

// compute dt from CFL considerations
Real
AmrCoreAdv::EstTimeStep (int lev, Real time)
{
    BL_PROFILE("AmrCoreAdv::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const Real* dx  =  geom[lev].CellSize();

    if (time == 0.0) {
       //DefineVelocityAtLevel(lev,time);
    } else {
       Real t_nph_predicted = time + 0.5 * dt[lev];
       //DefineVelocityAtLevel(lev,t_nph_predicted);
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        Real est = facevel[lev][idim].norminf(0,0,true);
        dt_est = amrex::min(dt_est, dx[idim]/est);
    }

    dt_est *= cfl;

    return dt_est;
}

// get plotfile name
std::string
AmrCoreAdv::PlotFileName (int lev) const
{
    return amrex::Concatenate(plot_file, lev, 5);
}

// put together an array of multifabs for writing
Vector<const MultiFab*>
AmrCoreAdv::PlotFileMF () const
{
    Vector<const MultiFab*> r;
    for (int i = 0; i <= finest_level; ++i) {
        r.push_back(&phi_new[i]);
    }
    return r;
}

// set plotfile variable names
Vector<std::string>
AmrCoreAdv::PlotFileVarNames () const
{
    return {"phi"};
}

// write plotfile to disk
void
AmrCoreAdv::WritePlotFile (int step) const
{
    const std::string plotfilename = std::string("phi") + std::to_string(step);
    const auto& mf = getMultiFabsPtr( getNewLevels() );

    amrex::Print() << "Writing plotfile " << plotfilename << "\n";
    amrex::WriteMultiLevelPlotfileHDF5(plotfilename,finest_level+1,mf,{"phi"},Geom(),getNewLevels()[0].getTime(),istep,refRatio() );

    //amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, mf, varnames,
    //                               Geom(), t_new[0], istep, refRatio());


}

void
AmrCoreAdv::WriteCheckpointFile () const
{

    // chk00010            write a checkpoint file with this root directory
    // chk00010/Header     this contains information you need to save (e.g., finest_level, t_new, etc.) and also
    //                     the BoxArrays at each level
    // chk00010/Level_0/
    // chk00010/Level_1/
    // etc.                these subdirectories will hold the MultiFab data at each level of refinement

    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = amrex::Concatenate(chk_file,istep[0]);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    const int nlevels = finest_level+1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
   if (ParallelDescriptor::IOProcessor()) {

       std::string HeaderFileName(checkpointname + "/Header");
       VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
       std::ofstream HeaderFile;
       HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
       HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                               std::ofstream::trunc |
                                               std::ofstream::binary);
       if( ! HeaderFile.good()) {
           amrex::FileOpenFailed(HeaderFileName);
       }

       HeaderFile.precision(17);

       // write out title line
       HeaderFile << "Checkpoint file for AmrCoreAdv\n";

       // write out finest_level
       HeaderFile << finest_level << "\n";

       // write out array of istep
       for (int i = 0; i < istep.size(); ++i) {
           HeaderFile << istep[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of dt
       for (int i = 0; i < dt.size(); ++i) {
           HeaderFile << dt[i] << " ";
       }
       HeaderFile << "\n";

       // write out array of t_new
       for (int i = 0; i < t_new.size(); ++i) {
           HeaderFile << t_new[i] << " ";
       }
       HeaderFile << "\n";

       // write the BoxArray at each level
       for (int lev = 0; lev <= finest_level; ++lev) {
           boxArray(lev).writeOn(HeaderFile);
           HeaderFile << '\n';
       }
   }

   // write the MultiFab data to, e.g., chk00010/Level_0/
   for (int lev = 0; lev <= finest_level; ++lev) {
       VisMF::Write(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "phi"));
   }

}


namespace {
// utility to skip to next line in Header
void GotoNextLine (std::istream& is)
{
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}
}

void
AmrCoreAdv::ReadCheckpointFile ()
{

    amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

    // Header
    std::string File(restart_chkfile + "/Header");

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    is >> finest_level;
    GotoNextLine(is);

    // read in array of istep
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            istep[i++] = std::stoi(word);
        }
    }

    // read in array of dt
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            dt[i++] = std::stod(word);
        }
    }

    // read in array of t_new
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            t_new[i++] = std::stod(word);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {

        // read in level 'lev' BoxArray from Header
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

        // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);

        // build MultiFab and FluxRegister data
        int ncomp = 1;
        int nghost = 0;
        phi_old[lev].define(grids[lev], dmap[lev], ncomp, nghost);
        phi_new[lev].define(grids[lev], dmap[lev], ncomp, nghost);

        if (lev > 0 && do_reflux) {
            flux_reg[lev].reset(new FluxRegister(grids[lev], dmap[lev], refRatio(lev-1), lev, ncomp));
        }

        // build face velocity MultiFabs
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, 1);
        }
    }

    // read in the MultiFab data
    for (int lev = 0; lev <= finest_level; ++lev) {
        VisMF::Read(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi"));
    }

    

}

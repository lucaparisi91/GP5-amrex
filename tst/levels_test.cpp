#include "gtest/gtest.h"
#include "mpi.h"
#include "../src/gpLevel.h"
#include "../src/operators.h"
#define TOL 1e-7

void testGaussian(gp::level & currentLevel, real_t alpha)
{
    auto & phi = currentLevel.getMultiFab();
    auto & geom = currentLevel.getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            
            const Box& vbx = mfi.validbox();
            auto const& phiArr = phi.array(mfi);
            const auto left= geom.ProbDomain().lo();
            const auto right= geom.ProbDomain().hi();

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];
            #if AMREX_SPACE_DIM > 2
                auto z = left[2] + k* dx[2];
            #endif
                auto r2 = x*x + y*y;
            #if AMREX_SPACE_DIM > 2
                r2 += z*z;
            #endif
                ASSERT_NEAR( exp( - alpha*r2) , phiArr(i,j,k) , TOL);


            });
        
        }
}

void testGaussianLaplacian(gp::level & currentLevel, real_t alpha, real_t tol)
{
    auto & phi = currentLevel.getMultiFab();
    auto & geom = currentLevel.getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            
            const Box& vbx = mfi.validbox();
            auto const& phiArr = phi.array(mfi);
            const auto left= geom.ProbDomain().lo();
            const auto right= geom.ProbDomain().hi();

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto x = left[0] + (i + 0.5)* dx[0];
                auto y = left[1] + (j + 0.5 )* dx[1];
            #if AMREX_SPACE_DIM > 2
                auto z = left[2] + k* dx[2];
            #endif
                auto r2 = x*x + y*y;
            #if AMREX_SPACE_DIM > 2
                r2 += z*z;
            #endif
                ASSERT_NEAR( exp( - alpha*r2)*(2 -2*alpha*r2)*(-2*alpha) , phiArr(i,j,k) , tol);


            });
        
        }
}



TEST( levels, initGaussian )
{
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain());
    BoxArray ba(dom);
    DistributionMapping dm(ba);
    
    gp::level level0(geo,ba,dm);
    real_t alpha=1./2;

    initGaussian(level0,alpha);
    testGaussian(level0,alpha);

}

TEST( level, saveGaussian )
{
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain());
    BoxArray ba(dom);
    DistributionMapping dm(ba);
    
    gp::level level0(geo,ba,dm);
    real_t alpha=1./2;

    initGaussian(level0,alpha);

    level0.saveHDF5("gauss");
    

}


auto createBiLayer()
{
    auto geo0 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom0(geo0.Domain());
    BoxArray ba0(dom0);
    DistributionMapping dm0(ba0);

    auto geo1 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(128,128,128 ) });
    Box dom1( {44,44},{75,75});
    BoxArray ba1(dom1);
    DistributionMapping dm1(ba1);

    
    auto level0=std::make_shared<gp::level> (geo0,ba0,dm0);
    auto level1=std::make_shared<gp::level> (geo1,ba1,dm1);
    
    gp::levels levels( { level0,level1});
    return levels;

}



TEST( levels, saveGaussian )
{

    auto geo0 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom0(geo0.Domain());
    BoxArray ba0(dom0);
    DistributionMapping dm0(ba0);

    auto geo1 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(128,128,128 ) });
    Box dom1( {44,44},{75,75});
    BoxArray ba1(dom1);
    DistributionMapping dm1(ba1);

    
    auto level0=std::make_shared<gp::level> (geo0,ba0,dm0);
    auto level1=std::make_shared<gp::level> (geo1,ba1,dm1);


    real_t alpha=1./(2*0.1*0.1);

    gp::levels levels( { level0,level1});


    ASSERT_EQ(levels.size(),2);
    ASSERT_EQ(levels.getTimes()[0],0);
    ASSERT_EQ(levels.getRefRatio(0)[0],2);
    ASSERT_EQ(levels.getRefRatio(0)[1],2);

    initGaussian(*level0,alpha);
    initGaussian(*level1,alpha);


    levels.averageDown();

    levels.save("gauss-2level");

}

TEST( levels, laplacian )
{
    auto levelsLeft = createBiLayer();
    auto levelsRight = createBiLayer();

    real_t alpha = 1/(2*0.1*0.1);

    initGaussian( levelsLeft[0],alpha);
    initGaussian( levelsLeft[1],alpha);

    levelsLeft.averageDown();

    auto lap = std::make_shared<gp::laplacianOperation>();
    lap->define(levelsLeft);
    lap->apply(levelsLeft,levelsRight);

    testGaussianLaplacian(levelsRight[0],alpha,10);
    testGaussianLaplacian(levelsRight[1],alpha,2);

    levelsRight.save("gauss-lap");

}

TEST(levels,gp)
{
    auto levelsLeft = createBiLayer();
    auto levelsRight = createBiLayer();

    real_t alpha = 1/(2*0.1*0.1);

    initGaussian( levelsLeft[0],alpha);
    initGaussian( levelsLeft[1],alpha);
    levelsLeft.averageDown();


    gp::trappedVortex func(1 , {AMREX_D_DECL(1,1,1)} );
    func.addVortex({AMREX_D_DECL(0,0,0)} );
    func.define( levelsLeft);
    func.apply( levelsLeft, levelsRight );
    
    levelsRight.averageDown();
    levelsRight.save("gpFunc");

}









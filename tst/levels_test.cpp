#include "gtest/gtest.h"
#include "mpi.h"
#include "../src/gpLevel.h"
#include "../src/operators.h"
#define TOL 1e-7

void testGaussian(gp::level & currentLevel, real_t alpha, int iComponent=0)
{
    auto & phi = currentLevel.getMultiFab(iComponent);
    auto & geom = currentLevel.getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    bool empty=true;
    
    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            empty=false;
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
    ASSERT_EQ(empty, false);
}

void testBiGaussian(gp::level & currentLevel, real_t alpha1 , real_t alpha2 , int iComponent=0 )
{
    auto & phi = currentLevel.getMultiFab(iComponent);
    auto & geom = currentLevel.getGeometry();

    amrex::GpuArray<real_t,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    bool empty=true;
    
    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
        {
            empty=false;
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
                ASSERT_NEAR( exp( - alpha1*r2) + exp( - alpha2*r2) , phiArr(i,j,k) , TOL );

            });
        
        }
    ASSERT_EQ(empty, false);

}




void testGaussianLaplacian(gp::level & currentLevel, real_t alpha, real_t tol,int iComponent=0)
{
    auto & phi = currentLevel.getMultiFab(iComponent);
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

    gp::level level0(geo,ba,dm,2);
    real_t alpha=1./2;


    initGaussian(level0,alpha,0);
    testGaussian(level0,alpha,0);

    initGaussian(level0,alpha,1);
    testGaussian(level0,alpha,1);

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
    level0.saveHDF5("out/gauss");
}


TEST( complexLevel, laplacian )
{
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain());
    BoxArray ba(dom);
    DistributionMapping dm(ba);

    auto levLeft= std::make_shared<gp::level>(geo,ba,dm,2);
    auto levRight= std::make_shared<gp::level>(geo,ba,dm,2);


    levLeft->getMultiFab().mult(0);
    levRight->getMultiFab().mult(0);


    

    real_t alphaReal=1./(2*0.1*0.1);
    real_t alphaImag=1./(2*0.2*0.2);

    initGaussian(*levLeft,alphaReal,0);
    initGaussian(*levLeft,alphaImag,1);

    gp::levels levelsLeft ({levLeft});
    gp::levels levelsRight ({levRight});

    auto lap = std::make_shared<gp::laplacianOperation>();
    lap->define(levelsLeft);
    lap->apply(levelsLeft,levelsRight);


    levelsLeft.save("out/gaussComplex");
    levelsRight.save("out/lap-gaussComplex");


    testGaussianLaplacian(levelsRight[0],alphaReal,8,0);
    testGaussianLaplacian(levelsRight[0],alphaImag,8,1);

}

TEST(levels,define)
{
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain() );
    BoxArray ba( dom );
    DistributionMapping dm( ba );

    auto level0=std::make_shared<gp::level>(geo,ba,dm);
    real_t alpha=1./(2*0.1*0.1);


    auto levels = std::make_shared<gp::levels>();
    levels->define({level0});
    (*levels)[0].getMultiFab().mult(0);

    auto level0New=std::make_shared<gp::level>();

    level0New->define(*level0);

    initGaussian( *level0New,alpha,0);
    testGaussian( *level0New,alpha,0);

    auto levelsNew = std::make_shared<gp::levels>();
    levelsNew->define(*levels);

    initGaussian( (*levelsNew)[0],alpha,0);
    testGaussian( (*levelsNew)[0],alpha,0);

}

TEST( level, normalizationSingleLayer )
{
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain() );
    BoxArray ba( dom );
    DistributionMapping dm( ba );

    auto level0=std::make_shared<gp::level>(geo,ba,dm);
    real_t alpha=1./(2*0.1*0.1);
    initGaussian(*level0,alpha);

    auto levels = std::make_shared<gp::levels>();
    levels->define({level0});

    gp::realWaveFunction wave( levels  );
    
    wave.updateDensity(0);

    auto & density = wave.getDensity();


    testGaussian(density[0],2*alpha);
    
    ASSERT_NEAR( wave.getNorm(0), 2*M_PI*0.1*0.1/2,1e-9);

    auto norm = wave.getNorm();

    ASSERT_EQ(norm.size(),1);
    ASSERT_EQ( norm[0], wave.getNorm(0) );

    wave.normalize(1,0);

    ASSERT_NEAR(  wave.getNorm(0) , 1 , 1e-9  );

}


TEST( complexLevel, normalizationSingleLayer )
{
    
    auto geo = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom( geo.Domain() );
    BoxArray ba( dom );
    DistributionMapping dm( ba );

    auto level0=std::make_shared<gp::level>(geo,ba,dm,2);
    real_t alphaR=1./(2*0.1*0.1);
    real_t alphaI=1./(2*0.2*0.2);

    initGaussian(*level0,alphaR,0);
    initGaussian(*level0,alphaI,1);

    auto levels = std::make_shared<gp::levels>();
    levels->define({level0});

    gp::complexWaveFunction wave( levels );

    wave.updateDensity(0);

    auto & density = wave.getDensity();
    
    testBiGaussian(density[0],2*alphaR,2*alphaI);
    
    real_t expectedNorm=2*M_PI*0.1*0.1/2 + 2*M_PI*0.2*0.2/2 ;
    ASSERT_NEAR( wave.getNorm(0), expectedNorm ,1e-9);


    auto norm = wave.getNorm();

    ASSERT_EQ(norm.size(),1);
    ASSERT_NEAR( norm[0], expectedNorm ,1e-9);

    wave.normalize(1,0);
    ASSERT_NEAR( wave.getNorm(0), 1 ,1e-9);

    wave.normalize(1,0);

    ASSERT_NEAR(  wave.getNorm(0) , 1 , 1e-9  );

}

auto createBiLayerLevels( bool real=true)
{
    auto geo0 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(64,64,64 ) });
    Box dom0(geo0.Domain());
    BoxArray ba0(dom0);
    DistributionMapping dm0(ba0);

    auto geo1 = gp::createGeometry( { AMREX_D_DECL(-1,-1,-1)},{AMREX_D_DECL(1,1, 1)},{AMREX_D_DECL(128,128,128 ) });
    Box dom1( {44,44},{75,75});
    BoxArray ba1(dom1);
    DistributionMapping dm1(ba1);

    int nComponents=1;
    if (not real) nComponents*=2;

    auto level0=std::make_shared<gp::level> (geo0,ba0,dm0,nComponents);
    auto level1=std::make_shared<gp::level> (geo1,ba1,dm1,nComponents);

    std::vector<std::shared_ptr<gp::level> > _levels {level0,level1};
    auto levels=std::make_shared<gp::levels>( _levels  );

    return levels;
}





TEST( level, normalizationBiLayer )
{

    auto levels = createBiLayerLevels() ;
    

    real_t alpha=1/(2*0.1*0.1);

    initGaussian((*levels)[0],alpha);
    initGaussian((*levels)[1],alpha);
    levels->averageDown();

    gp::realWaveFunction wave( levels );

    ASSERT_NEAR( wave.getNorm(0), 2*M_PI*0.1*0.1/2, 2e-4 ) ;


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

    levels.save("out/gauss-2level");

}

TEST( levels, laplacian )
{
    auto levelsLeft = *createBiLayerLevels();
    auto levelsRight = *createBiLayerLevels();

    real_t alpha = 1/(2*0.1*0.1);

    initGaussian( levelsLeft[0],alpha);
    initGaussian( levelsLeft[1],alpha);

    levelsLeft.averageDown();

    auto lap = std::make_shared<gp::laplacianOperation>();
    lap->define(levelsLeft);
    lap->apply(levelsLeft,levelsRight);

    testGaussianLaplacian(levelsRight[0],alpha,10);
    testGaussianLaplacian(levelsRight[1],alpha,2);

    levelsRight.save("out/gauss-lap");

}


TEST(levels,gp)
{
    auto levelsLeft = createBiLayerLevels();
    auto levelsRight = createBiLayerLevels();

    real_t alpha = 1/(2*0.1*0.1);

    initGaussian( (*levelsLeft)[0],alpha);
    initGaussian( (*levelsLeft)[1],alpha);
    levelsLeft->averageDown();


    gp::realWaveFunction waveLeft(levelsLeft);
    gp::realWaveFunction waveRight(levelsRight);
    
    gp::trappedVortex func(1 , {AMREX_D_DECL(1,1,1)} );
    func.addVortex( {AMREX_D_DECL(0,0,0)} );
    func.define( *levelsLeft );
    func.apply( waveLeft, waveRight );


    //waveRight->averageDown();

    waveRight.getPhi().save("out/gpFunc");

}










from locale import normalize
import unittest
from gp import gp
import numpy as np
import gpAmreX
import matplotlib.pylab as plt
import tqdm
from pathlib import Path

def createGaussianBiLayer(domain=[[-1,1],[-1,1]],shape=(64,64) ,alpha=1/(2*0.1*0.1) ):
    geo=gp.geometry(  domain, shape    )
    level0=gp.level( geo,[geo.domainBox()] )
    level1=gp.level( geo.refine([2,2]), [ gp.box([[22,44],[22,44]  ]).refine([2,2]) ] )
    levels=[level0,level1]

    for level in levels:
        X,Y = gp.grid(level.boxes[0],level.geo)
        r=np.sqrt(X**2 + Y**2)
        phi=np.exp(-alpha*r**2)
        level[0]=[phi]
    phi = gp.field(levels)
    phi.averageDown()
    return(phi)


def createGaussianLayer(domain=[[-1,1],[-1,1]],shape=(64,64) ,alpha=1/(2*0.1*0.1) ):
    geo=gp.geometry(  domain, shape    )
    dtype="None"

    if hasattr(alpha,"__iter__"):
        dtype="complex"
    else:
        dtype="real"
    

    level=gp.level( geo,[geo.domainBox()] ,dtype=dtype)
    X,Y = gp.grid(level.boxes[0],level.geo)
    r=np.sqrt(X**2 + Y**2)

    if dtype=="complex":
        phi=np.exp(-alpha[0]*r**2) + 1j*np.exp(-alpha[1]*r**2)
    else:
        if dtype=="real":
            phi=np.exp(-alpha*r**2)

    level[0]=[phi]
    return gp.field([level])


class gpTest(unittest.TestCase):
    def assertNear( self,x1, x2, tol):
        self.assertLess( np.abs(x1-x2) , tol  )




class levels(unittest.TestCase):

    def test_geometry(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )

        self.assertEqual( len(geo.left), 2  )
        self.assertAlmostEqual( geo.left[0], -20  )
        self.assertAlmostEqual( geo.left[1], -20  )

        self.assertEqual( len(geo.right), 2  )
        self.assertAlmostEqual( geo.right[0], 20  )
        self.assertAlmostEqual( geo.right[1], 20  )

        self.assertEqual( len(geo.shape), 2  )
        self.assertEqual( geo.shape[0], 64  )
        self.assertEqual( geo.shape[1], 64  )

        self.assertEqual( len(geo.cellSize), 2  )
        self.assertAlmostEqual( geo.cellSize[0], 40./64  )
        self.assertAlmostEqual( geo.cellSize[1], 40./64  )

        geo2=gp.geometry.fromGeometry(geo)
        self.assertAlmostEqual( geo.left[0], geo2.left[1]  )
        self.assertAlmostEqual( geo.left[1], geo2.left[1]  )
        self.assertAlmostEqual( geo.right[0], geo2.right[1]  )
        self.assertAlmostEqual( geo.right[1], geo2.right[1]  )
        self.assertEqual( geo.shape[0], geo2.shape[0]  )
        self.assertEqual( geo.shape[1], geo2.shape[1]  )




    def test_geometrySave( self ):
            geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
            ob=geo.toYAML()
            geo2=gp.geometry.fromYAML(ob)
            self.assertAlmostEqual( geo.left[0],geo2.left[0] )
            self.assertAlmostEqual( geo.left[1],geo2.left[1] )
            self.assertAlmostEqual( geo.shape[0],geo2.shape[0] )
            self.assertAlmostEqual( geo.shape[1],geo2.shape[1] )
    
    

    def test_geometryRefine(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        geo2=geo.refine([2,2])
        self.assertEqual( geo2.shape, [128,128]  )
        self.assertAlmostEqual( geo2.left[0], -20  )
        self.assertAlmostEqual( geo2.right[0], 20  )
        self.assertAlmostEqual( geo2.left[1], -20  )
        self.assertAlmostEqual( geo2.right[1], 20  )
    
    
    def test_geometryBox(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=geo.domainBox()
        self.assertEqual( len(b.shape), 2  )
        self.assertEqual( b.shape[0], 64  )
        self.assertEqual( b.shape[1], 64  )

        self.assertEqual( len(b.left), 2  )
        self.assertEqual( b.left[0], 0  )
        self.assertEqual( b.left[1], 0  )

        self.assertEqual( len(b.right), 2  )
        self.assertEqual( b.right[0], 63  )
        self.assertEqual( b.right[1], 63  )


    
    def test_geometrySelect(self):
        geo=gp.geometry(  [ (-5,5) , (-5,5) ] , (20,20)    )

        idx=geo.index( [-3.8,-3.2 ]  )
        self.assertEqual(idx[0],2)
        self.assertEqual(idx[1],3)

        selectionBox=geo.selectBox( [(-2,2),(-3,3)] )
        self.assertEqual( selectionBox.left[0], 6 )
        self.assertEqual( selectionBox.left[1], 4 )
        
        self.assertEqual( selectionBox.right[0], 14 )
        self.assertEqual( selectionBox.right[1], 16 )
               
        
  



    def test_boxRefine(self):
        b=gp.box( ((11,18),(12,21)  )   )
        b2=b.refine((2,3) )
        
        self.assertEqual( len(b2.shape), 2  )
        self.assertEqual( b2.shape[0], 8*2  )
        self.assertEqual( b2.shape[1], 10*3  )


    

    def test_boxSaveLoad(self):
        b=gp.box( ((11,18),(12,21)  )   )
        ob = b.toYAML()
        b2=gp.box.fromYAML(ob)

        self.assertEqual( b.left[0], b2.left[0]  )
        self.assertEqual( b.left[1], b2.left[1]  )


    def test_level(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )


        l=gp.level( geo, [b] )
        self.assertEqual( l.data[0].shape, (64,64) )





    def test_levelSaveLoad(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        l=gp.level( geo, [b] )
        ob=l.toYAML()
        l2=gp.level.fromYAML(ob)

        self.assertEqual( l2.data[0].shape, (64,64) )
        self.assertEqual( l.dtype, l2.dtype )
        self.assertEqual( l.nComponents, l2.nComponents )
        self.assertEqual( l.geo.left[0], l2.geo.left[0] )
        self.assertEqual( l.geo.left[1], l2.geo.left[1] )
        self.assertEqual( l.geo.right[0], l2.geo.right[0] )
        self.assertEqual( l.geo.right[1], l2.geo.right[1] )
    

    
    def test_grid(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )

        X,Y = gp.grid(b,geo)
        self.assertAlmostEqual( X[1,0] - X[0,0] , geo.cellSize[0] )
        self.assertAlmostEqual( Y[0,1] - Y[0,0] , geo.cellSize[1] )
    

    def test_box(self):
        b=gp.box([ [0,10],[0,10] ]  )
        np.testing.assert_array_equal(b.left,[0,0])
        np.testing.assert_array_equal(b.right,[10,10])
        np.testing.assert_array_equal(b.shape,[11,11] )

        b2=gp.box.fromBox( b  )

        np.testing.assert_array_equal(b.left,b2.left)
        np.testing.assert_array_equal(b.right,b2.right)




class initialization(gpTest):

    
    def test_initGaussianSingleLevelReal(self):        
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b])
        alpha=1/(2*4**2)
        phi=np.exp(-alpha*r**2)
        l[0]=[phi]
        psi=l[0][0]
        self.assertNear( np.max(np.abs(psi - phi) ) , 0 ,1e-6)
    


    def test_fieldSaveLoadReal( self ):        
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b])
        alpha=1/(2*4**2)
        phi=np.exp(-alpha*r**2)
        l[0]=[phi]
        psi=gp.field( [l] )

        psi.save("gaussTest")
        psi2=gp.field.load("gaussTest")


        self.assertEqual(psi.dtype,psi2.dtype)
        self.assertEqual(psi.nComponents,psi2.nComponents)
        self.assertEqual(psi.time,psi2.time)

        
        self.assertEqual(psi[0].geo.left[0],psi2[0].geo.left[0])
        self.assertEqual(psi[0].geo.left[1],psi2[0].geo.left[1])

        self.assertAlmostEqual( np.max(np.abs(psi2[0][0][0] - psi[0][0][0]) ) , 0 )
        
        
        plt.show()
    


    def test_fieldSaveLoadComplex( self ):        
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b],dtype="complex")
        alpha=1/(2*4**2)
        phi=np.exp( 1j * r**2) * np.exp(-alpha*r**2) 

        l[0]=[phi]
        psi=gp.field( [l] )

        psi.save("gaussTest")
        psi2=gp.field.load("gaussTest")

        self.assertEqual(psi.dtype,psi2.dtype)
        self.assertEqual(psi.nComponents,psi2.nComponents)

        self.assertEqual(psi[0].geo.left[0],psi2[0].geo.left[0])
        self.assertEqual(psi[0].geo.left[1],psi2[0].geo.left[1])

        self.assertAlmostEqual( np.max(np.real(psi2[0][0][0] - psi[0][0][0]) ) , 0 )
        self.assertAlmostEqual( np.max(np.imag(psi2[0][0][0] - psi[0][0][0]) ) , 0 )


    
    def test_initGaussianSingleLevelComplex(self):        
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b],dtype="complex")
        alphaR=1/(2*4**2)
        alphaI=1/(2*3**2)

        phi=np.exp(-alphaR*r**2) + 1j * np.exp(-alphaI*r**2)
        l[0]=[phi]
        psi=l[0][0]
        self.assertNear( np.max(np.abs(psi - phi) ) , 0 ,1e-6)

        l2=gp.level.fromLevel(l)
        self.assertEqual(l.nComponents,l2.nComponents)

        
        psi=l2[0][0]
        np.testing.assert_array_equal(psi.shape,phi.shape)
        self.assertNear( np.max(np.abs(psi - phi) ) , 0 ,1e-6)





    

    def test_initGaussianMultipleLevel(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        level0=gp.level( geo,[geo.domainBox()] )
        level1=gp.level( geo.refine([2,2]), [ gp.box([[22,44],[22,44]  ]).refine([2,2]) ] )
        levels=[level0,level1]

        for level in levels:
            X,Y = gp.grid(level.boxes[0],level.geo)
            r=np.sqrt(X**2 + Y**2)
            alpha=1/(2*4**2)
            phi=np.exp(-alpha*r**2)
            level[0]=[phi] 

        
        phi = gp.field(levels)

        phi.averageDown()

        
        phi.save("gauss")


    def test_normalizationSingleLevelReal(self):
        alpha=1/(2*0.1**2)
        phi = createGaussianLayer(alpha=alpha )

        self.assertNear(phi.norm[0] ,  0.031415926535897934,1e-5)
        
        phi.norm=[1]
        self.assertAlmostEqual(phi.norm[0],1 )
        
        for i in [0]:
            X,Y = gp.grid( phi[i].boxes[0], phi[i].geo)
            r=np.sqrt(X**2 + Y**2)
            var=1/(2.*alpha)
            expected=np.exp(-alpha*r**2)/np.sqrt((2*np.pi*var/2)) *np.sqrt(1)

            self.assertNear( np.max(np.abs(expected - phi[i][0] ) ), 0,1e-5)
    

    def test_normalizationSingleLevelComplex(self):
        alphaR=1/(2*0.1**2)
        alphaI=1/(2*0.2**2)

        phi = createGaussianLayer(alpha=(alphaR,alphaI) )

        norm=phi.norm
        self.assertEqual(len(norm),1)
        self.assertNear(norm[0] ,  0.15707963267948968,1e-5)

        phi.norm=[1]
        self.assertNear( phi.norm[0] ,  1,1e-5)



    def test_normalizationTwoLevel(self):
        alpha=1/(2*0.1**2)
        phi = createGaussianBiLayer(alpha=alpha )
        norm=phi.norm
        self.assertEqual( len(norm), 1)
        self.assertNear(norm[0] ,  0.031415926535897934,2e-4 )
        phi.norm=[1]
        self.assertAlmostEqual(phi.norm[0],1 )


        for i in [0,1]:
            X,Y = gp.grid( phi[i].boxes[0], phi[i].geo)
            r=np.sqrt(X**2 + Y**2)
            var=1/(2.*alpha)
            expected=np.exp(-alpha*r**2)/np.sqrt((2*np.pi*var/2)) *np.sqrt(1)

            self.assertNear( np.max(np.abs(expected - phi[i][0] ) ), 0,4e-2)
        

class functional(unittest.TestCase):

    def test_vortexTrapped(self):
        phi1=createGaussianBiLayer()
        phi2=createGaussianBiLayer()
        func=gp.trappedVortex()
        func.addVortex([0,0])

        func.apply(phi1,phi2)
        phi2.save("trappedVortex")









def plotRadial(phi,ob="density"):
    for lev in range(len(phi) ):
            level=phi[lev]
            X,Y =gp.grid(level.boxes[0],level.geo)
            r=np.sqrt(X**2 + Y**2)
            y=None
            if ob == "density":
                y=np.abs(phi[lev][0][0].flatten())**2
            else:
                if ob == "phase":
                    y=np.angle(phi[lev][0][0].flatten()   )

            plt.plot(r.flatten(), y , "o" , label=str(lev) )


class timeStepping(gpTest):

    def timeSteppingTest( self,stepperName ):
        print("Time stepping - {} ".format(stepperName) )
        
        alpha=1/(2*2**2)
        shape=(64,64)
        domain=[ [-10,10],[-10,10] ]
        selections=[]
        #selections=[ [ [-2,2] , [-2,2]     ]  ,[  [-1,1] , [-1,1]   ] , [ [-0.5,0.5], [-0.5,0.5]  ] , [[-0.1,0.1],[-0.1,0.1]] ]

        phi0=gp.createField( domain,shape,selections=selections,dtype="complex" )
        #phiNew=createField( domain,shape,selections=selections,dtype="complex" )

        gp.initGaussian(phi0,alphaReal=alpha)


        phi0.norm=[1]

        #plotRadial(phiOld)
        #plt.show()

        N=1

        

        dt=0.01* phi0[ len(phi0) - 1  ].geo.cellSize[0] **2

        func=gp.trappedVortex( g=100,omega=[1,1] )
        stepper=gp.stepper( func, kind=stepperName,realTime=False )
        stepper.enableNormalization([N])


        maxNBlocks=10
        sim=gp.simulation(phi0,stepper=stepper,maxStepsPerBlock=1000,maxNBlocks=maxNBlocks,timeStep=dt,outputDir="imagTime")

        sim.run()


        phiOld=gp.field.load("imagTime/fields/phi_{:d}".format(maxNBlocks-1))

        
        densityMax=np.max(np.abs(phiOld[0][0][0])**2)
        self.assertNear(densityMax , 0.056432 , 1e-6 )

        #plotRadial(phiOld)

        #plt.legend()
        #plt.show()

        

    ########### real time evolution


        func=gp.trappedVortex(g=100,omega=[1,1] )
        stepper=gp.stepper( func , realTime= True,kind="RK4" )
        #stepper.enableNormalization([N])

        dt=0.01* phi0[ len(phi0) - 1  ].geo.cellSize[0] **2
        phiOld.time=0
        simReal = gp.simulation(phiOld,stepper=stepper,maxStepsPerBlock=1000,maxNBlocks=10,timeStep=dt,outputDir="realTime")
                
        simReal.run()
        
        densityMax=np.max(np.abs(simReal.phiOld[0][0][0])**2)
        self.assertNear(densityMax , 0.056432 , 1e-6 )


        #plotRadial(phiOld,ob="phase")



        #plt.legend()
        #plt.show()


    def test_eulero(self):
        self.timeSteppingTest("eulero")
    def test_RK4(self):
        self.timeSteppingTest("RK4")
    

    @unittest.skip("time stepping on two levels")
    def test_euleroTwoLevels(self):
        alpha=1/(2*2**2)
        shape=(64,64)


        phiOld=createGaussianBiLayer(domain=[[-20,20],[-20,20]] , alpha=alpha,shape=shape)
        phiNew=createGaussianBiLayer(domain=[[-20,20],[-20,20]] , alpha=alpha,shape=shape)

        phiOld.norm=[1]

        dt=0.1* phiOld[1].geo.cellSize[0] **2

        func=gp.trappedVortex(g=10000,omega=[1,1] )
        func.cpp().define(phiOld.cpp() )
        stepper=gpAmreX.stepper( func.cpp() , "eulero", realTime=False )

        func.addVortex( [0,0 ] )

        phiOld.save("init")

        for iStep in tqdm.tqdm(range(10000) ):
            stepper.advance( phiOld.cpp(),phiNew.cpp(), dt )
            phiNew.averageDown()
            phiOld, phiNew = phiNew, phiOld
        
        phiOld.save("result")
    




if __name__ == '__main__':
    gpAmreX.initialize()
    unittest.main()
    gpAmreX.finalize()
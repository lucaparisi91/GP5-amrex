
from locale import normalize
import unittest
from gp import gp
import numpy as np
import gpAmreX
import matplotlib.pylab as plt
import tqdm





def createGaussianBiLayer(domain=[[-1,1],[-1,1]],shape=(64,64) ,alpha=1/(2*0.1*0.1) ):
    geo=gp.geometry(  domain, shape    )
    level0=gp.level( geo,[geo.domainBox()] )
    level1=gp.level( geo.refine([2,2]), [ gp.box([[22,44],[22,44]  ]).refine([2,2]) ] )
    levels=[level0,level1]

    for level in levels:
        X,Y = gp.grid(level.boxes[0],level.geo)
        r=np.sqrt(X**2 + Y**2)
        phi=np.exp(-alpha*r**2)
        level[0]=phi
    phi = gp.field(levels)
    phi.averageDown()
    return(phi)


def createGaussianLayer(domain=[[-1,1],[-1,1]],shape=(64,64) ,alpha=1/(2*0.1*0.1) ):
    geo=gp.geometry(  domain, shape    )
    level=gp.level( geo,[geo.domainBox()] )
    X,Y = gp.grid(level.boxes[0],level.geo)
    r=np.sqrt(X**2 + Y**2)
    phi=np.exp(-alpha*r**2)
    level[0]=phi
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



               
        
    def test_box(self):
        b=gp.box( ((0,64),(0,64)  )   )
        self.assertEqual( len(b.shape), 2  )
        self.assertEqual( b.shape[0], 65  )
        self.assertEqual( b.shape[1], 65  )

        self.assertEqual( len(b.left), 2  )
        self.assertEqual( b.left[0], 0  )
        self.assertEqual( b.left[1], 0  )

        self.assertEqual( len(b.right), 2  )
        self.assertEqual( b.right[0], 64  )
        self.assertEqual( b.right[1], 64  )
    def test_boxRefine(self):
        b=gp.box( ((11,18),(12,21)  )   )
        b2=b.refine((2,3) )
        
        self.assertEqual( len(b2.shape), 2  )
        self.assertEqual( b2.shape[0], 8*2  )
        self.assertEqual( b2.shape[1], 10*3  )




    def test_level(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )


        l=gp.level( geo, [b] )
        self.assertEqual( l.data[0].shape, (64,64) )

    
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



class initialization(gpTest):

    def test_initGaussianSingleLevel(self):        
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b])
        alpha=1/(2*4**2)
        phi=np.exp(-alpha*r**2)
        l[0]=phi
        psi=l[0]
        self.assertAlmostEqual( np.max(np.abs(psi - phi) ) , 0)
    

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
            level[0]=phi
        
        phi = gp.field(levels)

        phi.averageDown()

        
        phi.save("gauss")


    def test_normalizationSingleLevel(self):
        alpha=1/(2*0.1**2)
        phi = createGaussianLayer(alpha=alpha )

        self.assertAlmostEqual(phi[0].norm ,  0.031415926535897934)
        phi.norm=1
        self.assertAlmostEqual(phi[0].norm,1 )
        self.assertAlmostEqual(phi.norm,1 )

        
        for i in [0]:
            X,Y = gp.grid( phi[i].boxes[0], phi[i].geo)
            r=np.sqrt(X**2 + Y**2)
            var=1/(2.*alpha)
            expected=np.exp(-alpha*r**2)/np.sqrt((2*np.pi*var/2)) *np.sqrt(1)

            self.assertAlmostEqual( np.max(np.abs(expected - phi[i][0] ) ), 0)




    def test_normalizationTwoLevel(self):
        alpha=1/(2*0.1**2)
        phi = createGaussianBiLayer(alpha=alpha )
        self.assertNear(phi[0].norm ,  0.031415926535897934,2e-4 )
        phi.norm=1
        self.assertAlmostEqual(phi[0].norm,1 )
        self.assertAlmostEqual(phi.norm,1 )
        phi.save("gp")


        for i in [0,1]:
            X,Y = gp.grid( phi[i].boxes[0], phi[i].geo)
            r=np.sqrt(X**2 + Y**2)
            var=1/(2.*alpha)
            expected=np.exp(-alpha*r**2)/np.sqrt((2*np.pi*var/2)) *np.sqrt(1)

            self.assertNear( np.max(np.abs(expected - phi[i][0] ) ), 0,2e-2)


class functional(unittest.TestCase):

    def test_vortexTrapped(self):
        phi1=createGaussianBiLayer()
        phi2=createGaussianBiLayer()
        func=gp.trappedVortex()
        func.addVortex([0,0])

        func.apply(phi1,phi2)


        phi2.save("gp")


class timeStepping(gpTest):

    def test_eulero(self):
        alpha=1/(2*2**2)
        shape=(128,128)

        phiOld=createGaussianLayer(domain=[[-10,10],[-10,10]] , alpha=alpha,shape=shape)
        phiNew=createGaussianLayer(domain=[[-10,10],[-10,10]] , alpha=alpha,shape=shape)



        phiOld.norm=1

        dt=0.1* phiOld[0].geo.cellSize[0] **2


        func=gp.trappedVortex(g=1,omega=[1,1] )
        func.cpp().define(phiOld.cpp())
        stepper=gpAmreX.stepper( func.cpp() )

        phiOld.save("init")


        for iStep in tqdm.tqdm(range(100000) ):
            stepper.advance( phiOld.cpp(),phiNew.cpp(), dt )
            phiNew.norm=1
            phiOld, phiNew = phiNew, phiOld
        
        phiOld.save("result")





if __name__ == '__main__':
    gpAmreX.initialize()
    unittest.main()
    gpAmreX.finalize()
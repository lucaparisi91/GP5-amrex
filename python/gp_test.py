
import unittest
import gp
import numpy as np
import matplotlib.pylab as plt

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

        
        #self.assertAlmostEqual( geo.right, (20,20)   )
        #self.assertEqual( geo.shape, (64,64,64)   )        
        
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
    

class initialization(unittest.TestCase):

    def test_initGaussian(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        l=gp.level(geo,[b])
        alpha=1/(2*4**2)
        l.data[0]=np.exp(-r**2*alpha)
    
    def test_evaluate_gaussian(self):
        geo=gp.geometry(  [ (-20,20) , (-20,20) ] , (64,64)    )
        b=gp.box( ((0,63),(0,63)  )   )
        X,Y = gp.grid(b,geo)
        r=np.sqrt(X**2 + Y**2)
        alpha=1/(2*4**2)
        l.data[0]=np.exp(-r**2*alpha)

        l0=gp.level(geo,[b])
        l1=gp.level(geo,[b])
        


        alpha=1/(2*4**2)







if __name__ == '__main__':
    unittest.main()
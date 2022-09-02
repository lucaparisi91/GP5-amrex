import numpy as np


class geometry:
    def __init__(self,domain,shape):
        self.domain=domain
        self.shape=shape
    

    @property
    def left(self):
        return np.array( [ extent[0] for extent in self.domain] )
    @property
    def right(self):
        return np.array([ extent[1] for extent in self.domain  ] )
    @property
    def cellSize(self):
        return (self.right - self.left)/self.shape

        

class box:
    def __init__(self,extent):
        self.extent=extent

        self.left=np.array([  dExtent[0]    for dExtent in extent] ).astype(int)
        self.right=np.array([  dExtent[1]    for dExtent in extent] ).astype(int)
        self.shape=self.right - self.left + 1


class level:
    def __init__(self,geo,boxes):
        self.geo=geo
        self.boxes=boxes
        self.data=[]
        for currentBox in boxes:
            self.data.append(np.zeros(currentBox.shape) )



class field:
    def __init__(self, levels):
        self.levels=levels



def grid( box,geo):
    dims=len(box.shape)
    x=[]
    for d in range(dims):
        x.append( (  np.arange(box.left[d],box.right[d] + 1 ,1 ) + 0.5)*geo.cellSize[d] + geo.left[d] )
    
    return np.meshgrid(*x,indexing="ij")



from plant import Plant
import numpy as np     
import rospy 
from std_msgs.msg import String


class Material_Handling(Plant):       
       
    def __init__(self, method):
        self.pub = rospy.Publisher('plant_simulation_topic', String, queue_size=10)    
        Plant.__init__(self, method)       

    def get_file_name_plant(self):                                                             
        return "MaterialHandling.spp"            

    def update(self, data):
         
        self.connect.setValue(".Models.Modelo.espera", data[0])
        self.connect.setValue(".Models.Modelo.stock", data[1])
        self.connect.setValue(".Models.Modelo.numviajes", data[2])
        self.connect.startSimulation(".Models.Modelo")
    
        a = np.zeros(9)
        b = np.zeros(20)
        c = np.zeros(20)
        for g in range(1,10):
            a[g-1] = plant_sim.getValue(".Models.Modelo.transportes[2,%s]" % (g))
        for h in range(1,21):
            b[h-1] = plant_sim.getValue(".Models.Modelo.buffers[3,%s]" % (h))
            c[h-1] = plant_sim.getValue(".Models.Modelo.salidas[2,%s]" % (h))
        d = np.sum(a)
        e = np.sum(b)
        f = np.sum(c)
        r = d*0.25+e*0.25+f*0.5
    
        self.connect.resetSimulation(".Models.Modelo")
        
        # topic publish
        resultado_str = "YES" 
        rospy.loginfo(resultado_str)
        self.pub.publish(resultado_str)    
        return r
    
    def process_simulation(self):
        if (self.connection()):
            self.connect.setVisible(True)
            self.method.process()
            self.method.plot()

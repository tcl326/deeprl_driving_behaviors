from .traffic_env import TrafficEnv
from .ego_vehicle import EgoVehicle
import os


class TrafficEnvCross2(TrafficEnv):
    def __init__(self, mode="gui"):
        lanes = ["n_m_0", "s_m_0", "e_m_0", "w_m_0", "m_n_0", "m_s_0", "m_e_0", "m_w_0"]

        # We don't use lights or loops
        lights=[]
        loops = []
        exitloops = []
        
        basepath = os.path.join(os.path.dirname(__file__), "config", "cross2")
        netfile = os.path.join(basepath, "cross2.net.xml")
        routefile = os.path.join(basepath, "cross2.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "cross2.add.xml")

        ego_vehicles = [EgoVehicle('ego_car', 'route_sn', 'EgoCar', 245., 261., 0.), ]
                        #EgoVehicle('ego_car', 'route_se', 'EgoCar', 245., 261., 0.),
                       # EgoVehicle('ego_car', 'route_sw', 'EgoCar', 245., 241., 0.)]

        super(TrafficEnvCross2, self).__init__(ego_vehicles=ego_vehicles, mode=mode, lights=lights, netfile=netfile,
                                               routefile=routefile, guifile=guifile, loops=loops, addfile=addfile,
                                               step_length="0.1", simulation_end=3000, lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        # if self.np_random.uniform(0, 1) > 0.5:
        ew = 0.15
        we = 0.15
        ns = 0.12
        sn = 0.01
        return {"ns": ns,
                "sn": sn,
                "ew": ew,
                "we": we,
                }

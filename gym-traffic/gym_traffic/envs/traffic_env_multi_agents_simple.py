from .traffic_env_multi_agents import TrafficEnvMultiAgents
from .ego_vehicle import EgoVehicle
import os
import numpy as np


class TrafficEnvMultiAgentsSimple(TrafficEnvMultiAgents):
    def __init__(self, mode="gui"):
        loops = ["loop{}".format(i) for i in range(12)]
        lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0", "0_n_0", "0_s_0", "0_e_0", "0_w_0"]

        lights=[]
        basepath = os.path.join(os.path.dirname(__file__), "config", "multiagents")
        netfile = os.path.join(basepath, "cross.net.xml")
        routefile = os.path.join(basepath, "cross.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "cross.add.xml")
        exitloops = ["loop4", "loop5", "loop6", "loop7"]

        ego_vehicles_s = [EgoVehicle('ego_car_s', 'route_sn', 'EgoCarS', 245., 261., 0.),
                        EgoVehicle('ego_car_s', 'route_se', 'EgoCarS', 245., 261., 0.),
                        EgoVehicle('ego_car_s', 'route_sw', 'EgoCarS', 245., 241., 0.)]

        ego_vehicles_w = [EgoVehicle('ego_car_w', 'route_we', 'EgoCarW', 245., 261., 0.),
                        EgoVehicle('ego_car_w', 'route_ws', 'EgoCarW', 245., 241., 0.),
                        EgoVehicle('ego_car_w', 'route_wn', 'EgoCarW', 245., 261., 0.)]

        ego_vehicles_n = [EgoVehicle('ego_car_n', 'route_ns', 'EgoCarN', 245., 241., 0.),
                        EgoVehicle('ego_car_n', 'route_nw', 'EgoCarN', 245., 241., 0.),
                        EgoVehicle('ego_car_n', 'route_ne', 'EgoCarN', 245., 261., 0.)]

        ego_vehicles_e = [EgoVehicle('ego_car_e', 'route_ew', 'EgoCarE', 245., 241., 0.),
                        EgoVehicle('ego_car_e', 'route_en', 'EgoCarE', 245., 261., 0.),
                        EgoVehicle('ego_car_e', 'route_es', 'EgoCarE', 245., 241., 0.)]

        ego_vehicles_dict_master = {'s': ego_vehicles_s, 'w':ego_vehicles_w, 'n':ego_vehicles_n, 'e':ego_vehicles_e}

        super(TrafficEnvMultiAgentsSimple, self).__init__(ego_vehicles_dict_master=ego_vehicles_dict_master, mode=mode, lights=lights, netfile=netfile,
                                               routefile=routefile, guifile=guifile, loops=loops, addfile=addfile,
                                               step_length="0.1", simulation_end=3000, lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        # if self.np_random.uniform(0, 1) > 0.5:
        ew = np.random.normal(0.15, 0.02)
        we = np.random.normal(0.12, 0.02)
        ns = np.random.normal(0.08, 0.02)
        sn = 0.01

        routes = {"ns": ns, "sn": sn, "ew": ew, "we": we}

        return routes

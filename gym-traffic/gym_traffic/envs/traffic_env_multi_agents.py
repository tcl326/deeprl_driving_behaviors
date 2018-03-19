from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from scipy.ndimage import rotate
# from scipy.misc import imsave
import matplotlib.pyplot as plt
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import random
import time

from skimage.draw import polygon

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficEnvMultiAgents(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lights, netfile, routefile, guifile, addfile, ego_vehicles_dict_master, loops=[], lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml", pngfile="tmp.png", mode="gui", detector="detector0",
                 step_length = "0.1",  simulation_end=3600, sleep_between_restart=0.1):
        # "--end", str(simulation_end),
        self.simulation_end = simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.mode = mode
        self._seed()
        self.loops = loops
        self.exitloops = exitloops
        self.loop_variables = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION, tc.LAST_STEP_VEHICLE_NUMBER]
        self.lanes = lanes
        self.detector = detector
        args = ["--net-file", netfile, "--route-files", routefile, "--additional-files", addfile, "--step-length", step_length]
                # "--collision.check-junctions", "true", "--collision.action", "remove", "--no-warnings"]

        if mode == "gui":
            binary = "sumo-gui"
            args += ["-S", "-Q", "--gui-settings-file", guifile]
        else:
            binary = "sumo"
            args += ["--no-step-log"]

        # with open(routefile) as f:
        #     self.route = f.read()
        self.tmpfile = tmpfile
        self.pngfile = pngfile
        self.sumo_cmd = [binary] + args
        self.sumo_step = 0
        self.lights = lights

        self.action_space = spaces.Discrete(3)
        self.throttle_actions = {0: 0., 1: 1., 2:-1.}

        self.ego_vehicles_dict_master = ego_vehicles_dict_master

        self.orientation_orders = ['s', 'w', 'n', 'e']
        # print(ego_vehicles_dict)
        self.ego_vehicles_dict = {orientation: self.random_vehicle(self.ego_vehicles_dict_master[orientation]) for orientation in self.orientation_orders}

        self.ego_veh_collision_dict = {orientation: False for orientation in self.orientation_orders}

        self.ego_veh_removed_dict = {orientation: False for orientation in self.orientation_orders}

        self.braking_time = 0.

        self.sumo_running = False
        self.viewer = None

    def random_vehicle(self, ego_vehicles_list):
        # print(ego_vehicles_list)
        ego_vehicle = ego_vehicles_list[np.random.choice(len(ego_vehicles_list))]
        return ego_vehicle

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    # def write_routes(self):
    #     self.route_info = self.route_sample()
    #     with open(self.tmpfile, 'w') as f:
    #         f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_sumo(self):
        self.route_info = self.route_sample()
        if not self.sumo_running:
            # self.write_routes()
            traci.start(self.sumo_cmd)
            for loopid in self.loops:
                traci.inductionloop.subscribe(loopid, self.loop_variables)
            self.sumo_running = True
        else: # Reset vehicles in simulation
            for orientation in self.orientation_orders:
                ego_veh = self.ego_vehicles_dict[orientation]
                if ego_veh.vehID in traci.vehicle.getIDList():
                    traci.vehicle.remove(vehID=ego_veh.vehID, reason=2)
            traci.simulation.clearPending()

        self.sumo_step = 0
        self.sumo_deltaT = traci.simulation.getDeltaT()/1000. # Simulation timestep in seconds

        self.ego_vehicles_dict = {orientation: self.random_vehicle(self.ego_vehicles_dict_master[orientation]) for orientation in self.orientation_orders}
        self.ego_veh_collision_dict = {orientation: False for orientation in self.orientation_orders}

        self.ego_veh_removed_dict = {orientation: False for orientation in self.orientation_orders}

        self.braking_time = 0.

        for orientation in self.orientation_orders:
            ego_veh = self.ego_vehicles_dict[orientation]
            traci.vehicle.add(vehID=ego_veh.vehID, routeID=ego_veh.routeID,
                              pos=ego_veh.start_pos, speed=ego_veh.start_speed, typeID=ego_veh.typeID)
            traci.vehicle.setSpeedMode(vehID=ego_veh.vehID, sm=0) # All speed checks are off
        # self.screenshot()

    def _stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def check_collision(self, orientation, obstacle_image):
        if (obstacle_image[:,:,0] * obstacle_image[:,:,1]).any():
            self.ego_veh_collision_dict[orientation] = True
        else:
            self.ego_veh_collision_dict[orientation] = False

        ego_veh = self.ego_vehicles_dict[orientation]
        if ego_veh.vehID in traci.vehicle.getIDList():
            min_dist = 100.00
            ego_pos = np.array(traci.vehicle.getPosition(ego_veh.vehID))
            for i in traci.vehicle.getIDList():
                # excluding ego vehicle AND any vehicle from the opposite direction (NS) for comparison
                if i != ego_veh.vehID:
                    pos = np.array(traci.vehicle.getPosition(i))
                    new_dist = np.linalg.norm(ego_pos - pos)
                    if new_dist < min_dist:
                        min_dist = new_dist
            if min_dist < 1.25: self.ego_veh_collision_dict[orientation] = True
            else: self.ego_veh_collision_dict[orientation] = False

        else:
            min_dist = 0.
            self.ego_veh_collision_dict[orientation] = True
        # if self.ego_veh_collision_dict[orientation] == True:
            # traci.vehicle.remove(vehID=ego_veh.vehID, reason=2)
            # traci.vehicle.remove(vehID=ego_veh.vehID, reason=2)
            # print("COLLISSION")
        return min_dist

    # choose action based on the Time-To-Collision metric
    # def action_from_ttc(self):
    #     ego_id = self.ego_veh.vehID
    #     ego_ang = traci.vehicle.getAngle(ego_id) * math.pi / 180.0 # in radian (clockwise)
    #     ego_vel = traci.vehicle.getSpeed(ego_id)  # in m/s
    #     # NOTE different coordinate system for angle and pose!
    #     ego_vel_x = ego_vel * math.sin(ego_ang)
    #     ego_vel_y = ego_vel * math.cos(ego_ang)
    #     # calculate position of vehicle center instead of front bumper
    #     ego_pos = np.array(traci.vehicle.getPosition(ego_id))
    #     # ego_pos[0] = ego_pos[0] - 2.5*math.cos(ego_ang)
    #     # ego_pos[1] = ego_pos[1] - 2.5*math.sin(ego_ang)
    #
    #     # initialize
    #     min_ttc = 10
    #     action  = -1
    #     front_warning = False
    #     pre_collision = False
    #
    #     routeID = self.ego_veh.routeID
    #     # straight cross
    #     if routeID == 'route_sn':
    #         for i in traci.vehicle.getIDList():
    #             # excluding ego vehicle AND NS traffic for comparison
    #             if i != ego_id and i.find('flow_n_s') == -1:
    #                 new_ang = traci.vehicle.getAngle(i) * math.pi / 180.0 # in radian (clockwise)
    #                 new_vel = traci.vehicle.getSpeed(i) # in m/s
    #                 new_vel_x = new_vel * math.sin(new_ang)
    #                 new_vel_y = new_vel * math.cos(new_ang)
    #                 new_pos = np.array(traci.vehicle.getPosition(i)) # in m
    #
    #                 rel_vel = np.linalg.norm([ego_vel_x - new_vel_x, ego_vel_y - new_vel_y])
    #                 rel_dist = np.linalg.norm(ego_pos - new_pos)
    #                 new_ttc = rel_dist / rel_vel
    #
    #                 rel_ang = math.atan2(new_pos[0] - ego_pos[0], new_pos[1] - ego_pos[1]) # in radian
    #                 if rel_ang < 0.0:
    #                     rel_ang = 2*math.pi + rel_ang
    #
    #                 if abs(rel_ang - ego_ang) < 0.5 and rel_dist < 10.0:
    #                     front_warning = True
    #                     if rel_dist < 2.0:
    #                         pre_collision = True
    #
    #                 if new_ttc < min_ttc:
    #                     min_ttc = new_ttc
    #     # left turn
    #     elif routeID == 'route_sw':
    #         for i in traci.vehicle.getIDList():
    #             # excluding ego vehicle for comparison only
    #             if i != ego_id:
    #                 new_ang = traci.vehicle.getAngle(i) * math.pi / 180.0 # in radian (clockwise)
    #                 new_vel = traci.vehicle.getSpeed(i) # in m/s
    #                 new_vel_x = new_vel * math.sin(new_ang)
    #                 new_vel_y = new_vel * math.cos(new_ang)
    #                 new_pos = np.array(traci.vehicle.getPosition(i)) # in m
    #
    #                 rel_vel = np.linalg.norm([ego_vel_x - new_vel_x, ego_vel_y - new_vel_y])
    #                 rel_dist = np.linalg.norm(ego_pos - new_pos)
    #                 new_ttc = rel_dist / rel_vel
    #
    #                 rel_ang = math.atan2(new_pos[0] - ego_pos[0], new_pos[1] - ego_pos[1]) # in radian
    #                 if rel_ang < 0.0:
    #                     rel_ang = 2*math.pi + rel_ang
    #
    #                 if abs(rel_ang - ego_ang) < 0.5 and rel_dist < 10.0:
    #                     front_warning = True
    #                     if rel_dist < 2.0:
    #                         pre_collision = True
    #
    #                 if new_ttc < min_ttc:
    #                     min_ttc = new_ttc
    #
    #     # right turn
    #     elif routeID == 'route_se':
    #         for i in traci.vehicle.getIDList():
    #             # ONLY take WE traffic for comparison
    #             if i.find('flow_w_e') != -1:
    #                 new_ang = traci.vehicle.getAngle(i) * math.pi / 180.0 # in radian (clockwise)
    #                 new_vel = traci.vehicle.getSpeed(i) # in m/s
    #                 new_vel_x = new_vel * math.sin(new_ang)
    #                 new_vel_y = new_vel * math.cos(new_ang)
    #                 new_pos = np.array(traci.vehicle.getPosition(i)) # in m
    #
    #                 rel_vel = np.linalg.norm([ego_vel_x - new_vel_x, ego_vel_y - new_vel_y])
    #                 rel_dist = np.linalg.norm(ego_pos - new_pos)
    #                 new_ttc = rel_dist / rel_vel
    #
    #                 rel_ang = math.atan2(new_pos[0] - ego_pos[0], new_pos[1] - ego_pos[1]) # in radian
    #                 if rel_ang < 0.0:
    #                     rel_ang = 2*math.pi + rel_ang
    #
    #                 if abs(rel_ang - ego_ang) < 0.5 and rel_dist < 10.0:
    #                     front_warning = True
    #                     if rel_dist < 2.0:
    #                         pre_collision = True
    #
    #                 if new_ttc < min_ttc:
    #                     min_ttc = new_ttc
    #
    #     else:
    #         print('Invalid route for ego-vehicle. No action will be taken.')
    #
    #     # decision making based on min time-to-collision
    #     if (front_warning and min_ttc < 3.0) or pre_collision:
    #         action = 2
    #     else:
    #         if min_ttc > 2.0 or abs(ego_pos[0] - 250) > 3 or ego_pos[1] - 250 > 2:
    #             action = 1
    #         elif min_ttc < 1.5:
    #             action = 2
    #         else:
    #             action = 0
    #
    #     # print 'min ttc: ', min_ttc, ' front warning: ', front_warning
    #
    #     return action

    # TODO: Refine reward function!!
    def _reward(self, orientation, min_dist, ego_vehicle):
        if self.ego_veh_collision_dict[orientation]:
            reward = -5000
        elif ego_vehicle.reached_goal(traci.vehicle.getPosition(ego_vehicle.vehID)):
            reward = 1000
        elif min_dist < 2.5:
            reward = -100
        else:
            reward = -1
        return reward

    def _step(self, actions):
        # actions_dict = {'n': actionOfSouthVeh, 'w': actionOfWestVeh, 's': actionOfNorthVeh, 'e': actionOfEastVeh}
        if not self.sumo_running:
            self.start_sumo()
        self.sumo_step += 1

        for i, orientation in enumerate(self.orientation_orders):
            if self.ego_veh_collision_dict[orientation]:
                continue
            action = actions[i]
            if action == 2:
                self.braking_time += 1

            new_speed = traci.vehicle.getSpeed(self.ego_vehicles_dict[orientation].vehID) + self.sumo_deltaT * self.throttle_actions[action]
            traci.vehicle.setSpeed(self.ego_vehicles_dict[orientation].vehID, new_speed)

        # print("Step = ", self.sumo_step, "   | action = ", action)
        # print("car speed = ", traci.vehicle.getSpeed(self.ego_veh.vehID), "   | new speed = ",new_speed)

        traci.simulationStep()
        observation_list = []
        reward_list = []
        done_list = []
        for i, orientation in enumerate(self.orientation_orders):
            observation_list.append(self._observation(orientation))
            min_dist = self.check_collision(orientation, observation_list[i])
            reward_list.append(self._reward(orientation, min_dist, self.ego_vehicles_dict[orientation]))
            done = self.ego_veh_collision_dict[orientation] \
                   or self.ego_vehicles_dict[orientation].reached_goal(traci.vehicle.getPosition(self.ego_vehicles_dict[orientation].vehID)) \
                   or (self.sumo_step > self.simulation_end)
            done_list.append(done)

        self.remove_collided_cars()

        # if self.sumo_step%5 == 0:
        # # plt.imshow(observation_list[0][:,:,0])ex
        #     plt.imshow(observation_list[2][:,:,1])
        #     plt.colorbar()
        #     plt.show()

        return np.array(observation_list), np.array(reward_list), np.array(done_list), self.route_info

    def remove_collided_cars(self):
        for orientation in self.orientation_orders:
            if self.ego_veh_collision_dict[orientation] and not self.ego_veh_removed_dict[orientation]:
                self.ego_veh_removed_dict[orientation] = True
                ego_veh = self.ego_vehicles_dict[orientation]
                if ego_veh.vehID not in traci.vehicle.getIDList():
                    continue
                traci.vehicle.remove(vehID=ego_veh.vehID, reason=3)

    def screenshot(self):
        if self.mode == "gui":
            # print('Screenshotting at', self.pngfile)
            traci.gui.screenshot("View #0", self.pngfile)

    def _observation(self, orientation):
        if self.ego_veh_collision_dict[orientation]:
            return np.zeros((84, 84, 2))
        state = []
        visible = []
        ego_car_in_scene=False

        ego_veh = self.ego_vehicles_dict[orientation]

        if ego_veh.vehID in traci.vehicle.getIDList():
            ego_car_pos = traci.vehicle.getPosition(ego_veh.vehID)
            ego_car_ang = traci.vehicle.getAngle(ego_veh.vehID)
            # print(orientation, ego_car_pos, ego_car_ang)
            ego_car_in_scene = True

        for i in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(i)
            pos = traci.vehicle.getPosition(i)
            angle = traci.vehicle.getAngle(i)
            laneid = traci.vehicle.getRouteID(i)
            state_tuple = (i,pos[0], pos[1], angle, speed, laneid)
            state.append(state_tuple)
            if ego_car_in_scene:
                if(np.linalg.norm(np.asarray(pos)-np.asarray(ego_car_pos))<42) and i not in ego_veh.vehID: #42 is 42 meters
                    visible.append(state_tuple)
        # print(visible)
        if not ego_car_in_scene:
            bound = 84
            obstacle_image = np.zeros((bound,bound,2))
        else:
            # print('Rendering')
            obstacle_image = self.render_scene(visible, ego_car_pos, ego_car_ang, orientation)

            # plt.imsave('test.jpg', obstacle_image)
            # plt.ion()
            # plt.imshow(obstacle_image)
            # plt.imshow(obstacle_image[:,:,0])
            # plt.imshow(obstacle_image[:,:,1])
            # plt.imshow(obstacle_image[:,:,2])
            # plt.draw(plt.imshow(obstacle_image))
            # plt.draw()
            # time.sleep(1.0)
            # time.sleep(5.0)
            # import IPython
            # IPython.embed()
            # plt.show(block=False)
            # plt.show()

        index = self.orientation_orders.index(orientation)
        obstacle_image = np.rot90(obstacle_image, k=index)
        return obstacle_image

    def render_scene(self, visible, ego_car_pos, ego_car_ang, orientation):

        def get_car_shape(car_length, car_width):
            r = [-1, -1, car_length/unit_dist, car_length/unit_dist]
            c = [-1, car_width/unit_dist, car_width/unit_dist,-1]
            rr, cc = polygon(r, c)
            return rr-car_length/1.2/unit_dist, cc-car_width/2.0/unit_dist

        def get_car_coords(x, y, angle, car_template_y, car_template_x):
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            transform = np.array([[c, -s, 0],[s, c, 0],[x, y, 1]])
            car_homogenous = np.ones((car_template_y.shape[0], 3))
            car_homogenous[:,0] = car_template_x
            car_homogenous[:,1] = car_template_y
            return np.dot(car_homogenous, transform)

        def draw_scene(X, Y, angles, car_template_y, car_template_x, bound, ids):
            cars = []
            for i, angle in enumerate(angles):
                car = get_car_coords(X[i], Y[i], angle, car_template_y, car_template_x)
                cars.append(car)
            scene = np.zeros((int(bound/unit_dist),int(bound/unit_dist),2))
            for id, car in zip(ids, cars):
                car_x = car[:,0]
                car_y = car[:,1]
                car_x = np.clip(car_x, 0, int(bound/unit_dist)-1)
                car_y = np.clip(car_y, 0, int(bound/unit_dist)-1)
                # print id
                val = 1
                # if id[-1] == 'e':
                #     val = 1
                # elif id[-1] == 'n':
                #     val = 2
                # elif id[-1] == 'w':
                #     val = 3
                # elif id[-1] == 's':
                #     val = 4
                scene[np.rint(car_y).astype(int), np.rint(car_x).astype(int), 1] = val

            ego_car_coord = get_car_coords(bound/2.0/unit_dist, bound/2.0/unit_dist, 0, car_template_y, car_template_x)
            car_x = ego_car_coord[:,0]
            car_y = ego_car_coord[:,1]
            scene[np.rint(car_y).astype(int), np.rint(car_x).astype(int), 0] = 1
            return scene

        def transform_visible(visible, ego_car_pos, ego_car_ang, bound):
            X, Y, angles, ids = [], [], [], []
            ego_x, ego_y = ego_car_pos[0], ego_car_pos[1]
            for car_id,pos_x, pos_y, angle, speed, laneid in visible:
                X.append((pos_x - ego_x)/unit_dist + bound/2.0/unit_dist)
                Y.append((pos_y - ego_y)/unit_dist + bound/2.0/unit_dist)
                ids.append(car_id)
                index = self.orientation_orders.index(orientation)
                angles.append(angle-ego_car_ang-90*index)

            return np.array(X), np.array(Y), np.array(angles), ids

        car_length = 4.3
        car_width = 1.8
        unit_dist = .1
        bound = 42
        car_template_y, car_template_x = get_car_shape(car_length, car_width)

        X, Y, angles, ids = transform_visible(visible, ego_car_pos, ego_car_ang, bound)

        # X = np.clip(X, 0, int(bound/unit_dist)-1)
        # Y = np.clip(Y, 0, int(bound/unit_dist)-1)

        obstacle_image = draw_scene(X, Y, angles, car_template_y, car_template_x, bound, ids)

        return np.flipud(obstacle_image)

    def _reset(self):
        # if self.sumo_running:
        #     traci.vehicle.remove(vehID=self.ego_veh.vehID, reason=2)
        #     traci.simulation.clearPending()
        #     self.sumo_step = 0
        #     for i in range(800):
        #         traci.simulationStep()
        #     self.ego_veh = random.choice(self.ego_vehicles)
        #     self.ego_veh_collision = False
        #     traci.vehicle.add(vehID=self.ego_veh.vehID, routeID=self.ego_veh.routeID,
        #                       pos=self.ego_veh.start_pos, speed=self.ego_veh.start_speed, typeID=self.ego_veh.typeID)
        #     traci.vehicle.setSpeedMode(vehID=self.ego_veh.vehID, sm=0) # All speed checks are off
        # else:
        #     # self.stop_sumo()
        #     self.ego_veh = random.choice(self.ego_vehicles)
        #     # sleep required on some systems
        #     if self.sleep_between_restart > 0:
        #         time.sleep(self.sleep_between_restart)
        self.start_sumo()
        observations = []
        for orientation in self.orientation_orders:
            observations.append(self._observation(orientation))
        return np.array(observations)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.mode == "gui":
            img = imread(self.pngfile, mode="RGB")
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            raise NotImplementedError("Only rendering in GUI mode is supported. Please use Traffic-...-gui-v0.")

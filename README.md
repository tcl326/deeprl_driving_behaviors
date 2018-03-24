A Deep Reinforcement Learning Approach to Behavioral Planning for Autonomous Vehicles
=====================================================================================

Behavioral planning for autonomous vehicles in scenarios such as intersections and lane-changing/merging is a challenging task. The code in this repository is for a couple of the experiments performed for [this paper](https://drive.google.com/file/d/1SCEF6Fn0J4lhWHa8SsjbmhmZMXFfp1ba/view?usp=sharing).

Installation
------------

The code of this repo has been tested on Ubuntu 14.04 and with Python 2.7

1. [Install SUMO 0.30](http://sumo.dlr.de/wiki/Installing)

   Execute the following:

   `sudo apt-get install sumo sumo-tools sumo-doc`

   `sudo add-apt-repository ppa:sumo/stable`

   `sudo apt-get update`

   `sudo apt-get install sumo sumo-tools sumo-doc`

   Please make sure these instructions are followed exactly.

2. Additional Python Packages install via pip:

   `pip install moviepy imageio tqdm tensorflow==1.4 requests scipy matplotlib ipython`

3. Include `export SUMO_HOME=/usr/share/sumo` in your `~/.bashrc` and `source ~/.bashrc`

Training
--------

To train the DRQN:
Add a folder called Center in the gym-traffic folder. In the Center folder, create a subdirectory called frames. This is where the learning result frames are stored.
Create a folder in the examples folder called drqn. This is where the weights checkpoint files are stored. The checkpoint files are created every 100 episodes, so you must train at least 100 in order to run the DRQN.

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_drqn.py train`

Experiments
-----------
For each of these experiments, you will probably want to set the delay to 0ms once the GUI opens. This will decrease the amount of time you spend waiting for the ego-vehicle to reach the intersection. The simulation should slow down when the ego-vehicle arrives, but you may still need to manage the speed manually via the delay setting.

To test a random routine:

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_random.py`


To test a TTC (Time to Collision) rule based approach:

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_ttc.py`


To test the DRQN (Deep Recurrent Q-Network) based approach (you must have trained at least 100 episodes before running this):

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_drqn.py test`

To run with the GUI enabled:

`python example_gym_traffic_drqn.py test --gui`





# Installation of Gazebo with ROS Indigo

1. Install gazebo from source

$sudo apt-get remove '.*gazebo.*' '.*sdformat.*' '.*ignition-math.*' '.*ignition-msgs.*' '.*ignition-transport.*'

$sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable lsb_release -cs main" > /etc/apt/sources.list.d/gazebo-stable.list'

$wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

$sudo apt-get update

$sudo apt-get install ros-$ROS_DISTRO-gazebo7-ros-pkgs

2. Make sure that you succeed with installing gazebo

In terminal:

$ gazebo

You should see a GUI with a empty world.

$ which gzserver

$ which gzclient

You should see:

/usr/local/bin/gzserver

/usr/local/bin/gzclient

3. Install gazebo_ros_pkgs

$mkdir -p ~/autonomous_car/src

$cd ~/autonomous_car/src

$catkin_init_workspace

$cd ~/autonomous_car

$catkin_make

$echo "source ~/autonomous_car/devel/setup.bash" >> ~/.bashrc



$sudo apt-get install -y gazebo2


$cd ~/autonomous_car/src

$git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b indigo-devel

(you might need to delete some previous installed gazebo7, choose yes)

$rosdep update

$rosdep check --from-paths . --ignore-src --rosdistro indigo

$rosdep install --from-paths . --ignore-src --rosdistro indigo -y


$cd ~/autonomous_cat/

$catkin_make

$sudo gedit /usr/share/gazebo-2.2/setup.sh

change the "GAZEBO_MODEL_DATABASE_URI" to http://models.gazebosim.org


4. Test

source, roscore

source, rosrun gazebo_ros gazebo

(You should see a GUI)

source, rostopic list

(You should see
/gazebo/link_states
/gazebo/model_states
/gazebo/parameter_descriptions
/gazebo/parameter_updates
/gazebo/set_link_state
/gazebo/set_model_state
)

source, rosservice list

(You should see
/gazebo/apply_body_wrench
/gazebo/apply_joint_effort
/gazebo/clear_body_wrenches
/gazebo/clear_joint_forces
/gazebo/delete_model
/gazebo/get_joint_properties
/gazebo/get_link_properties
/gazebo/get_link_state
/gazebo/get_loggers
/gazebo/get_model_properties
/gazebo/get_model_state
/gazebo/get_physics_properties
/gazebo/get_world_properties
/gazebo/pause_physics
/gazebo/reset_simulation
/gazebo/reset_world
/gazebo/set_joint_properties
/gazebo/set_link_properties
/gazebo/set_link_state
/gazebo/set_logger_level
/gazebo/set_model_configuration
/gazebo/set_model_state
/gazebo/set_parameters
/gazebo/set_physics_properties
/gazebo/spawn_gazebo_model
/gazebo/spawn_sdf_model
/gazebo/spawn_urdf_model
/gazebo/unpause_physics
/rosout/get_loggers
/rosout/set_logger_level
)

Reference:
http://gazebosim.org/tutorials

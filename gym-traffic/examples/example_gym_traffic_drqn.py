# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_path, 'gym-bstriner'))
sys.path.insert(0, os.path.join(repo_path, 'gym-traffic'))



# from gym_traffic.agents import DQN, EpsilonExplorer
from gym_traffic.agents import DRQN
# from gym_traffic.runners import SimpleRunner
from gym_traffic.runners import DRQNRunner
from gym_traffic.runners import DRQNTester
from gym_traffic.runners import DRQNTesterGUI
import gym
# from gym_traffic.runners.agent_runner import run_agent
# from gym_traffic.runners import DRQNRunner
import sys
import argparse

# def build_agent(env):
#     return DRQN(env)
#
def example(operation, gui):
    # train_env = gym.make('Traffic-Simple-cli-v0')
    # main_agent = build_agent(train_env)  ## just initialize a DRQN object here - this builds the network as well
    # target_agent = build_agent(train_env)
    # path = "output/traffic/simple/dqn"
    # explorer = EpsilonExplorer(agent, epsilon=0.5, decay=5e-7)  ## we do need this ? helps us trade off between
    ## exploration and exploitation? need this? (only difference in act function)

    # if gui:
    #     def test_env_func():
    #         return gym.make('Traffic-Simple-gui-v0')
    # else:
    #     def test_env_func():
    #         return train_env

    ## TRAINING
    if (operation == 'train'):
        train_env = gym.make('Traffic-Multi-cli-v0')
        runner = DRQNRunner(max_steps_per_episode=1000) ## make a new class DRQNRunner which will have all the
    ## functionality of "Training the network" - in the run function
    ## Don't worry about testing here's
    ## you will need to retain newepisode, observe, learn, etc. in DRQN class
    ## DRQNRunner will retain basic things like agent.new_episode(), etc. but everything else will be from DRQN:
    ## with all double DQN, etc., targetQN, etc.
    ## add info everywhere, see if it will cause any problems

    # def run(self, env, nb_epoch, nb_episodes = 100, render=True, verbose=True, train=True)
        runner.run_training(train_env)
    elif (operation == 'test'):
        if gui:
            train_env = gym.make('Traffic-Multi-gui-v0')
            tester = DRQNTesterGUI()
        else:
            train_env = gym.make('Traffic-Multi-cli-v0')
            tester = DRQNTester()
        tester.run_testing(train_env)



    # video_callable = None if gui else False
    # run_agent(runner=runner, agent=explorer, test_agent=explorer, train_env=train_env, test_env_func=test_env_func,
    #           nb_episodes=500, test_nb_episodes=10, nb_epoch=100, path=path, video_callable=video_callable)


def main():
    # operation = sys.argv[1]
    # if sys.argv[2] == 'gui':
    #     gui = True

    # example(operation)
    parser = argparse.ArgumentParser(description='Example DRQN implementation ofr intersection navigation.')
    # parser.add_argument('--train', action='store_const', const='train', help='Train the DRQN Model')
    # parser.add_argument('--test', action='store_const', const='test', help='Test the DRQN Model')
    parser.add_argument("operation", help="Example Modes: (1) train (2) test")
    parser.add_argument('-G', '--gui', action="store_true", help='run GUI mode during testing to render videos')

    args = parser.parse_args()
    example(args.operation, args.gui)

    # example(operation, gui)


if __name__ == "__main__":
    main()

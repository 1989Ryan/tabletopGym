from os.path import abspath, dirname, join
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), 'ompl/py-bindings/'))
# import OMPL lib
from ompl import base as ob
from ompl import geometric as og
from itertools import product
import numpy as np
from utils import pairwise_collision, pairwise_link_collision, get_self_link_pairs, get_moving_links,\
    DEFAULT_PLANNING_TIME, INTERPOLATE_NUM


'''
toolbox for ompl planner connecting with simulation engine
this lib should not use the pybullet lib directly
'''

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.
    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, client, id):
        """
        init function
        args:
            id: robot id in pybullet engine
        """
        # Public attributes
        self.client = client
        self.id = id
        all_joint_num = self.client.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        self.joint_bounds = []
        self.ll = []
        self.ul = []
        self.jr = []
        self.state = [self.client.getJointState(self.id, i)[0] for i in self.joint_idx]

    def _is_not_fixed(self, joint_idx):
        joint_info = self.client.getJointInfo(self.id, joint_idx)
        return joint_info[2] != self.client.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = self.client.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            self.ll.append(low + 0.01)
            self.ul.append(high - 0.01)
            self.jr.append(high-low - 0.01)
            if low < high:
                self.joint_bounds.append([low + 0.01, high - 0.01])
        
        assert len(self.joint_bounds) == self.num_dim
        return self.joint_bounds

    def update_state(self):
        state = []
        for joint in self.joint_idx:
            state.append(self.client.getJointState(self.id, joint)[0])
        self.state = state

    def get_cur_state(self):
        state = []
        for joint in self.joint_idx:
            state.append(self.client.getJointState(self.id, joint)[0])
        self.state = state
        return state

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            self.client.resetJointState(self.id, joint, value, targetVelocity=0)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim):
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, client, robot, obstacles = []):
        '''
        Args
            client: pybullet client
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.client = client
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        # print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("BITstar") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if pairwise_link_collision(self.client, self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if pairwise_collision(self.client, body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = get_self_link_pairs(self.client, robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in get_moving_links(self.client, robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME, interp_n = INTERPOLATE_NUM):
        '''
        plan a path to goal from the given robot start state
        '''
        
        orig_robot_state = start
        
        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)

        for i in range(len(start)):
            g[i] = float(goal[i])
            s[i] = start[i]
            
        self.ss.setStartAndGoalStates(s, g)
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        if solved:
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(interp_n)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)
            res = True
        self.ss.clear()
        self.robot.set_state(orig_robot_state)
        
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME, interp_n = INTERPOLATE_NUM):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time, interp_n=interp_n)
    

    def execute(self, path, dynamics=False, video=False, get_obs=None,
                gui_writer=None, get_gui_obs=None
                ):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        step = 0
        for q in path:
            if dynamics:
                gains = np.ones(len(self.robot.joint_idx))
                self.client.setJointMotorControlArray(
                    bodyIndex=self.robot_id,
                    jointIndices=self.robot.joint_idx,
                    controlMode=self.client.POSITION_CONTROL,
                    targetPositions=q,
                    positionGains=gains)
            else:
                self.robot.set_state(q)
            self.client.stepSimulation()
            step += 1
            if video and step % 20 == 19:
                obs_gui = get_gui_obs()
                gui_writer.append_data(obs_gui)
                get_obs()
                # video_writer.append_data(obs)
                step = 0



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]

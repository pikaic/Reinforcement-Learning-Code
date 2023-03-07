import pandas as pd
import numpy as np
import time
import random

import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist


ACTIONS     = ['up', 'down', 'left', 'right']
LENGTH      = None
N_STATES    = None
START       = None
HOLE1       = None
HOLE2       = None
TERMINAL    = None
EPSILON     = None
MAX_EPISODE = None
GAMMA       = None
ALPHA       = None
FIRST       = True

#############################################################################
#############################################################################

# Initial Q-Table
    
def build_q_table():
    global N_STATES
    global ACTIONS
    table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )
    return table

# Actor
# e-Greedy


def actor(observation, q_table):
    #self.check_state_exist(observation)
    # action selection
    if np.random.uniform() < EPSILON:
        # choose best action
        state_action = q_table.loc[observation, :]
        # print("####")
        # print(state_action)
        # some actions may have the same value, randomly choose on in these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
        # choose random action
        action = np.random.choice(ACTIONS)
    return action

# Enviroment Visual
def update_env(state, episode, step):
    view = np.array([['_ '] * LENGTH] * LENGTH)
    view[tuple(TERMINAL)] = '* '
    view[HOLE1] = 'X '
    view[HOLE2] = 'X '
    view[tuple(state)] = 'o '
    interaction = ''
    for v in view:
        interaction += ''.join(v) + '\n'
   
    
    #########################################
    #########################################
    #clear_output(wait=True)
    #print(interaction)
    #time.sleep(0.1)
    #########################################
    #########################################

# Enviroment Feedback

def init_env():
    global HOLE1
    global HOLE2
    global FIRST
    global START
    global TERMINAL
    start = START


    #HOLE1 = (2,2)
    #HOLE2 = (1,0)
    HOLE1 = (1,2)
    HOLE2 = (2,2)
    
    FIRST = False
    return start, False

def get_env_feedback(state, action):
    reward = 0.
    end = False
    a, b = state
    if action == 'up':
        a -= 1
        if a < 0:
            a = 0
        next_state = (a, b)
        if next_state == TERMINAL:
            reward = 1.
            end = True
        elif (next_state == HOLE1) or (next_state == HOLE2):
            reward = -1.
            end = True
    elif action == 'down':
        a += 1
        if a >= LENGTH:
            a = LENGTH - 1
        next_state = (a, b)
        if (next_state == HOLE1) or (next_state == HOLE2):
            reward = -1.
            end = True
    elif action == 'left':
        b -= 1
        if b < 0:
            b = 0
        next_state = (a, b)
        if (next_state == HOLE1) or (next_state == HOLE2):
            reward = -1.
            end = True
    elif action == 'right':
        b += 1
        if b >= LENGTH:
            b = LENGTH - 1
        next_state = (a, b)
        if next_state == TERMINAL:
            reward = 1.
            end = True
        elif (next_state == HOLE1) or (next_state == HOLE2):
            reward = -1.
            end = True
    #print("::next ::", next_state, " action ::: ", action)
    return next_state, reward, end

def playGame(q_table):
    maze_transitions = []
    state = (3,0)
    end = False
    LENGTH  = 4
    a, b = state
    i = 0
    while not end:
        #a, b = state
        #print("state ::", state)
        act = actor(a * LENGTH + b, q_table)
        #print("step::", i ," action ::", act)
        maze_transitions.append(act)
        next_state, reward, end = get_env_feedback(state, act)
        state = next_state
        a, b = state
        i += 1
    #print("==> Game Over <==")
    return maze_transitions

## following function replaces name of actions (string) to number (int)
## function is used for ther control motion of the drone, NOT for the SARSA algorithm!
def droneActions(maze_transitions):
    actions = []
    for action in maze_transitions:
        if action == 'up':
            actions.append(0)
        if action == 'down':
            actions.append(1)
        if action == 'right':
            actions.append(2)
        if action == "left":
            actions.append(3)        
    return actions



def droneMotions(drone_actions):
    pos_drone = 0
    head = [pos_drone] + drone_actions
    drone_move = []
    
    
    for i in range(len(head)-1):
        if head[i] == head[i+1]:
            drone_move.append(0)
        if head[i] != head[i+1]:
            if ((head[i] == 0) or (head[i] == 1)):
                if head[i+1] == 3:
                    drone_move.append(1)
                if head[i+1] == 2:
                    drone_move.append(-1)
                    
            if ((head[i] == 2) or (head[i] == 3)):
                if head[i+1] == 0:
                    drone_move.append(1)
                if head[i+1] == 1:
                    drone_move.append(-1)
    
    return drone_move

# Learn 

######learn SARSA algorithm

def learnSARSA():
    #build the Q-table (see the definition of the build_q_table() function )
    q_table = build_q_table()
    episode = 0
    
    #main learing loop
    while episode < MAX_EPISODE:
        state, end = init_env()
        step = 0
        #update the environment after each episode
        update_env(state, episode, step)
        # take a position of first state (start state)
        a, b = state
        # based on start state and actual Q-table agent take a action (by calling method actor() )
        act = actor(a * LENGTH + b, q_table)
        
        #when the agent does not win or drop into the HOLE run this loop
        while not end:
            #agent took a action: act and the agent receives the feedback from environment:
            #next_state, reward, end                
            next_state, reward, end = get_env_feedback(state, act)
            
            #position of next state the agent transits (state ==> next_state  ::: a,b ==> a_, b_
            a_, b_ = next_state
            
            #agent takes new action based on new target state
            act_ = actor(a_ * LENGTH + b_, q_table)

            # the agent takes predicted values from Q-table 
            q_predict = q_table.loc[a * LENGTH + b, act]
            
            # agent estimates the value of Q target (the agent is still in a,b but the agent
            #computes new state a_, b_)
            if next_state != TERMINAL:
                q_target = reward + GAMMA * q_table.loc[a_ * LENGTH + b_, act_]
            else:
                q_target = reward
            ######################################################        
            ### SARSA - compare with above formula (above cell)###
            ######################################################
            q_table.loc[a * LENGTH + b, act] += ALPHA * (q_target - q_predict)
            
            #agent formaly transits to new state and environment update
            state = next_state
            act = act_
            a, b = state
            step += 1
            update_env(state, episode, step)
            #print("step", step)
            if step > 30: # feel free to change this parameter
                #print("END")
                end = True
            

        episode += 1
    return q_table


###################################### END OF SARSA ######################################  
        
##########################################################################################
##########################################################################################

class MoveDroneClass(object):

    def __init__(self):

        self.ctrl_c = False
        self.rate = rospy.Rate(1)

    def publish_once_in_cmd_vel(self, cmd):
        """
        This is because publishing in topics sometimes fails teh first time you publish.
        In continuos publishing systems there is no big deal but in systems that publish only
        once it IS very important.
        """
        while not self.ctrl_c:
            connections = self._pub_cmd_vel.get_num_connections()
            if connections > 0:
                self._pub_cmd_vel.publish(cmd)
                rospy.loginfo("Publish in cmd_vel...")
                break
            else:
                self.rate.sleep()

    # function that stops the drone from any movement
    def stop_drone(self):
        rospy.loginfo("Stopping...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)

    # function that makes the drone turn 90 degrees
    def turn_drone(self, move):
        rospy.loginfo("Turning...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = -0.6 * move *2 # 
        self.publish_once_in_cmd_vel(self._move_msg)
        #self._move_msg.linear.z = 2
        #self.pub_position.publish(self._move_msg)

    # function that makes the drone move forward
    def move_forward_drone(self):
        rospy.loginfo("Moving forward...")
        self._move_msg.linear.x = 0.2 * 3
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)

    def move_drone(self, motion):
        actual_heading = 0

        # this callback is called when the action server is called.
        # this is the function that computes the Fibonacci sequence
        # and returns the sequence to the node that called the action server

        # helper variables
        r = rospy.Rate(5)

        # define the different publishers and messages that will be used
        self._pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._move_msg = Twist()
        self._pub_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._takeoff_msg = Empty()
        self._pub_land = rospy.Publisher('/drone/land', Empty, queue_size=1)
        self._land_msg = Empty()
        #self.pub_position = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        #var_twist = Twist()
        #position, orientation = self.get_odom()
        

        # define the seconds to move in each side of the square (which is taken from the goal) and the seconds to turn
        sideSeconds = 3.3
        turnSeconds = 1.5  # 3.09 #1.8

          # ===================DRONE TAKEOFF===========================================
        i = 0
        while not i == 2:
            self._pub_takeoff.publish(self._takeoff_msg)
            rospy.loginfo('Taking off...')
            time.sleep(1)
            i += 1

         # ==========================================================================

        
        for move in motion:

            #turning_time = 4 - actual_heading + move
            #print("motion ::::", move) #, "turning time : ", turning_time)

            if move == 0:
                self.move_forward_drone()
                time.sleep(sideSeconds)
                actual_heading = move

            if move != 0:
                
                self.turn_drone(move)
                time.sleep(turnSeconds)
                self.move_forward_drone()
                r.sleep()
                time.sleep(sideSeconds)
                actual_heading = move

            
            # the sequence is computed at 1 Hz frequency
            r.sleep()

        # ===================DRONE STOP AND LAND=====================================
        self.stop_drone()
        i = 0
        while not i == 3:
            self._pub_land.publish(self._land_msg)
            rospy.loginfo('Landing...')
            time.sleep(1)
            i += 1
        # =============================================================================
        
##########################################################################################
##########################################################################################


if __name__ == '__main__':
    LENGTH      = 4 
    N_STATES    = LENGTH * LENGTH
    START       = (LENGTH - 1, 0)
    TERMINAL    = (0,3)
    EPSILON     = .9
    MAX_EPISODE = 1000 
    GAMMA      = .9
    ALPHA       = .1 #0.1

    q_table = learnSARSA()

    maze_transitions = playGame(q_table)
    actions = droneActions(maze_transitions)
    print("maze_transitions ::", actions)
    drone_motions = droneMotions(actions)
    print("drone motion ::", drone_motions)

    #following motions are expected. 
    #if computed drone motions are different then so applied the expected

    #drone_motions_exp = [0, 0, 0, -1, 0, 0]
    #if (drone_motions != drone_motions_exp):
     #   drone_motions = drone_motions_exp


    
    rospy.init_node('move_drone')
    move_drone = MoveDroneClass()
    try:
        move_drone.move_drone(drone_motions)
    except rospy.ROSInterruptException:
        pass
    





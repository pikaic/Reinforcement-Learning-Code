#! /usr/bin/env python

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
GAMMA      = None
ALPHA       = None
FIRST       = True

    
def build_q_table():
    global N_STATES
    global ACTIONS
    table = pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )
    return table

def actor(observation, q_table):

    if np.random.uniform() < EPSILON:
        state_action = q_table.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
        action = np.random.choice(ACTIONS)
    return action

def update_env(state, episode, step):
    view = np.array([['_ '] * LENGTH] * LENGTH)
    view[tuple(TERMINAL)] = '* '
    view[HOLE1] = 'X '
    view[HOLE2] = 'X '
    view[HOLE3] = 'X '
    view[tuple(state)] = 'o '
    interaction = ''
    for v in view:
        interaction += ''.join(v) + '\n'



def init_env():
    global HOLE1
    global HOLE2
    global HOLE3
    global FIRST
    global START
    global TERMINAL
    start = START
    HOLE1 = (1,3)
    HOLE2 = (1,1)
    HOLE3 = (2,1)
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
        elif (next_state == HOLE1) or (next_state == HOLE2) or (next_state == HOLE3):
            reward = -1.
            end = True
    elif action == 'down':
        a += 1
        if a >= LENGTH:
            a = LENGTH - 1
        next_state = (a, b)
        if (next_state == HOLE1) or (next_state == HOLE2) or (next_state == HOLE3):
            reward = -1.
            end = True
    elif action == 'left':
        b -= 1
        if b < 0:
            b = 0
        next_state = (a, b)
        if (next_state == HOLE1) or (next_state == HOLE2) or (next_state == HOLE3):
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
        elif (next_state == HOLE1) or (next_state == HOLE2) or (next_state == HOLE3):
            reward = -1.
            end = True

    return next_state, reward, end


def playGame(q_table):
    maze_transitions = []
    state = (3,0)
    end = False
    LENGTH  = 4
    a, b = state
    i = 0
    while not end:

        act = actor(a * LENGTH + b, q_table)
        print("step::", i ," action ::", act)
        maze_transitions.append(act)
        next_state, reward, end = get_env_feedback(state, act)
        state = next_state
        a, b = state
        i += 1
    print("==> Game Over <==")
    return maze_transitions


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
    pos_drone = 0 #2
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

######learn QLEARN algorithm



def Qlearn():

    q_table = build_q_table()
    episode = 0
    while episode < MAX_EPISODE:
        state, end = init_env()
        step = 0
        update_env(state, episode, step)
        while not end:
            a, b = state
            
            act = actor(a * LENGTH + b, q_table)
                        
            next_state, reward, end = get_env_feedback(state, act)
            
            na, nb = next_state
            
            q_predict = q_table.loc[a * LENGTH + b, act]
            
            if next_state != TERMINAL:
            ### Qlearning algoritm
            ###################################################################
                q_target = reward + GAMMA * q_table.iloc[na * LENGTH + nb].max()
            else:
                q_target = reward
            q_table.loc[a * LENGTH + b, act] += ALPHA * (q_target - q_predict)
            state = next_state
            step += 1
            update_env(state, episode, step)
            
            #if step > 30: # the same punish like presented in previous module. Choose to apply or not
             #   end = True

        episode += 1
    return q_table


###################################### END OF QLEARN #####################################   
        
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
    MAX_EPISODE = 400 
    GAMMA      = .9
    ALPHA       = .01 #0.1

    q_table = Qlearn()
 
    maze_transitions = playGame(q_table)
    actions = droneActions(maze_transitions)
    print("maze_transitions ::", actions)
    drone_motions = droneMotions(actions)
    print("drone motion ::", drone_motions)
    #following motions are expected. 
    #if computed drone motions are different then so applied the expected
    """
    EXPECTED MOTION FOR THE DRONE
    'step::', 0, ' action ::', 'right')
    ('step::', 1, ' action ::', 'right')
    ('step::', 2, ' action ::', 'up')
    ('step::', 3, ' action ::', 'up')
    ('step::', 4, ' action ::', 'up')
    ('step::', 5, ' action ::', 'right')
    """
    #drone_motions_exp = [-1, 0, 1, 0, 0, -1]
    #if (drone_motions != drone_motions_exp):
    #    drone_motions = drone_motions_exp

    
    rospy.init_node('move_drone')
    move_drone = MoveDroneClass()
    try:
        move_drone.move_drone(drone_motions)
    except rospy.ROSInterruptException:
        pass
    





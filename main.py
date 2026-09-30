#!/usr/bin/env pybricks-micropython

import random
from pybricks.ev3devices import ColorSensor, InfraredSensor, Motor
from pybricks.hubs import EV3Brick
from pybricks.parameters import Button, Port
from pybricks.robotics import DriveBase
from pybricks.tools import wait
from pybricks.parameters import Stop
import pickle

# Constants
WHITE_VALUE = 25
BLACK_VALUE = 8
TURN_ANGLE = 2
DRIVE_SPEED = 20
MIN_SPEED = 70       # mm/s, forward speed after a turn (was 100)
MAX_SPEED = 120      # mm/s, forward speed cap (was 180)
FORWARD_WAIT = 100   # ms max length of one forward step (was 250)
SENSOR_CHECK = 20    # ms between sensor checks inside forward / turn actions

# --- Obstacle reverse + U-turn ---
# These are COMMANDED values in DriveBase units, NOT real mm / degrees
# (DriveBase is configured with wheel_diameter=40, axle_track=50, which does not
# match the real robot). Tune them with CALIBRATE_TURN = True (see bottom of file):
#   new U_TURN_CMD = U_TURN_CMD * 180 / (real angle turned, in degrees)
#   new REVERSE_CMD = REVERSE_CMD * (wanted mm) / (real mm reversed)
REVERSE_CMD = 36     # commanded distance to back away from the obstacle
U_TURN_CMD = 830     # commanded angle for the U-turn (666 gave ~130 degrees)
ALPHA = 0.1  # Learning rate
EPSILON = 1  # Exploration rate
GAMMA = 0.9  # Discount factor
E=2.7321
TEMP=1000


# Initialize hardware components
ev3 = EV3Brick()
left_motor = Motor(Port.A)
right_motor = Motor(Port.D)
light_sensor = ColorSensor(Port.S1)
ir_sensor = InfraredSensor(Port.S4)
robot = DriveBase(left_motor, right_motor, wheel_diameter=40, axle_track=50)

# utility functions
def save_q_dict(q_dict):
    temp = {}
    for key,value in q_dict.items():
        temp[(key[0],key[1],key[2].__name__)] = value
    with open('q_dict_v1.pkl', 'wb') as file:
        pickle.dump(temp, file)
       
def load_q_dict():
    # Reading the dictionary from the file
    with open('q_table.pkl', 'rb') as file:
        temp = pickle.load(file)
    q_dict = {}
    for key,value in temp.items():
        q_dict[(key[0],key[1],globals()[key[2]])] = value
    return q_dict

def get_light_state():
    light_sensor_reading = light_sensor.reflection()
    if(light_sensor_reading >= WHITE_VALUE):
        return 'WHITE'
    if(light_sensor_reading <= BLACK_VALUE):
        return 'BLACK'
    return 'MIDDLE'

# Robot actions
def forward(robot, previous_light_state):
    speed = min(max(MIN_SPEED,(robot.state()[1]))+5,MAX_SPEED)
    robot.drive(speed,0)
    # keep checking the sensor; end the step as soon as the robot leaves the edge
    for i in range(FORWARD_WAIT // SENSOR_CHECK):
        wait(SENSOR_CHECK)
        if get_light_state() != previous_light_state:
            break

def turn_left(robot, previous_light_state):
    robot.drive(10,-110)
    while previous_light_state == get_light_state() :
        wait(SENSOR_CHECK)
    robot.stop()  # stop turning as soon as the state changes (no overshoot)

def turn_right(robot,previous_light_state):
    robot.drive(10,110)
    while previous_light_state == get_light_state():
        wait(SENSOR_CHECK)
    robot.stop()  # stop turning as soon as the state changes (no overshoot)

def backward(robot, previous_light_state,mode):
    # 1) Reverse a bit to clear the obstacle
    robot.straight(-REVERSE_CMD)
    # 2) U-turn in place (left when mode is True, right when False)
    robot.turn((-1 if mode else 1) * U_TURN_CMD)
    ev3.speaker.beep()
    print("U-turn: reverse cmd", REVERSE_CMD, "turn cmd", U_TURN_CMD, "reflection", light_sensor.reflection())

def find_path(robot):
    # After the turn, go and find the line again (no RL), then hand back to the Q-table.
    # 1) creep forward until the sensor sees the line edge (grey)
    robot.drive(50,0)
    for i in range(100):  # about 2 seconds
        if get_light_state() == 'MIDDLE':
            robot.stop()
            return
        wait(20)
    # 2) not found: sweep left, right, left ... a bit wider each time
    direction = -1
    for span in (1000,2000,4000):  # ms
        robot.drive(0, direction*100)
        for i in range(span//20):
            if get_light_state() == 'MIDDLE':
                robot.stop()
                return
            wait(20)
        direction = -direction
    robot.stop()

actions = [forward, turn_left, turn_right]
modes = [True,False]
light_states = ['WHITE','MIDDLE','BLACK']
m_y = [('MIDDLE','turn_right','BLACK'),('BLACK','turn_left','MIDDLE'),('MIDDLE','turn_left','WHITE'),('WHITE','turn_right','MIDDLE')]
m_x = [('MIDDLE','turn_right','WHITE'),('WHITE','turn_left','MIDDLE'),('MIDDLE','turn_left','BLACK'),('BLACK','turn_right','MIDDLE')]


# Initialize Q-table
Q_table = {}
for mode in modes:
    for act in actions:
        for light in light_states:
            Q_table[(mode,light, act)] = 0

# Get reward for a given state
def get_reward(light_state):
    return -10 if light_state in ['BLACK','WHITE'] else 10
 
# Get best action for a given state
def get_best_action(Q_table, mode, light_state):
    max_q = -float("inf")
    best_action = None
    for act in actions:
        q = Q_table[(mode, light_state, act)]
        if q > max_q:
            best_action = act
            max_q = q
    return best_action, max_q

def learn():
    light_state = get_light_state()
    mode = True
    iterations = 0
    action = None
    
    while True:
        # Exploration vs. Exploitation
        if (E**(iterations/-TEMP) < 0.01):
            TRAINING = False
            save_q_dict(Q_table)
            break
        
        if random.uniform(0, 1) <  E**(iterations/-TEMP):
            action = random.choice(actions)  # Explore by choosing a random action
            print("random", action.__name__, mode)
        else:
            action = get_best_action(Q_table, mode,light_state)[0]  # Exploit by choosing the action with the highest Q-value
            print("greedy", action.__name__ , mode)

        action(robot, light_state)  # Execute the action and wait for the robot to finish

        new_light_state = get_light_state()

        new_mode = mode
        if((light_state,action.__name__,new_light_state) in m_x):
            new_mode = True
        elif ((light_state,action.__name__,new_light_state) in m_y):
            new_mode = False

        # Calculate max Q-value for the new state
        max_q_next = get_best_action(Q_table, new_mode, new_light_state)[1]

        # Calculate reward for the new state
        reward_next = get_reward(new_light_state)

        # Update Q-table
        Q_table[(mode, light_state, action)] += ALPHA * (reward_next + GAMMA * max_q_next - Q_table[( mode, light_state, action)])

        # Print iteration number on the EV3 screen
        ev3.screen.clear()
        ev3.screen.draw_text(30,40,iterations)
        ev3.screen.draw_text(20,50,E**(iterations/-TEMP))

        light_state = new_light_state
        mode = new_mode
        iterations += 1

        save_q_dict(Q_table)


def obstacle_avoidance(mode):
    backward(robot, light_sensor, mode)
    find_path(robot)

def line_following(Q_table, mode, light_state):
    action = get_best_action(Q_table, mode,light_state)[0]  # Choose the action with the highest Q-value
    print("line following",mode, action.__name__, light_state)

    action(robot, light_state)  # Execute the action and wait for the robot to finish
    new_light_state = get_light_state() # Observe new state

    if((light_state,action.__name__,new_light_state) in m_x):
        mode = True
    elif ((light_state,action.__name__,new_light_state) in m_y):
        mode = False
    

    return mode, new_light_state

def run():
    light_state = get_light_state()
    mode = True

    # Load Q-table
    Q_table = load_q_dict()
    print(Q_table)

    # Find mode
    action = turn_right
    action(robot, light_state)  # Execute the action and wait for the robot to finish
    new_light_state = get_light_state()
    if((light_state,action.__name__,new_light_state) in m_x):
        mode = True
    elif ((light_state,action.__name__,new_light_state) in m_y):
        mode = False
    light_state = new_light_state

    # Run
    while True:
        if (ir_sensor.distance() < 15):
            print("obstacle")
            robot.stop()
            ev3.speaker.say("Avoiding Obstacle")
            obstacle_avoidance(mode)
            mode = not mode
            light_state = get_light_state()  # refresh after the U-turn
        else:
            mode, light_state = line_following(Q_table, mode, light_state)

TRAINING = False
CALIBRATE_TURN = False  # True: only do the reverse + U-turn once and stop (for tuning)

if(CALIBRATE_TURN):
    backward(robot, None, True)
else:
    if(TRAINING):
        learn()

    run()
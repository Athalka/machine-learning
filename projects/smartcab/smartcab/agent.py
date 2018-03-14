import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=True, epsilon=1.0, alpha=0.005):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.trial_number = 0


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
            self.trial_number += 1
            #epsilon decaying function
            
            #use slow decay for alpha
            """
            self.alpha = self.alpha  - 0.001
            if self.alpha < 0.01:
                self.alpha = 0.01
            """
            # self.epsilon = self.epsilon - 0.05 # linear decay - default learning
            # self.epsilon = 1.0 / (float(self.trial_number) ** 2)
            # self.epsilon = self.alpha ** float(self.trial_number)
            
            #exponential decay
            #self.epsilon = math.e ** (-1 * self.alpha  * float(self.trial_number)) 
            
            #cosine decay
            self.epsilon = ( math.cos( self.alpha * float(self.trial_number))) ## best result with epsilon=1.0, alpha=0.01, alpha constant
            #self.epsilon = abs( math.cos( self.alpha * (float(self.trial_number) ** 1.0/2.0))) 
            
            #streched decaying function alternative
            #self.epsilon = math.e ** (- (float(self.trial_number)** 3 ) *  self.alpha)

            # decay function on 100 zero e ^(- x^(3 )  * 0.05^4) 
            #self.epsilon =  math.e ** (-1 * (float(self.trial_number) ** 2) * (self.alpha ** 3))

            #slow decay 
            #self.epsilon = 1 - 0.5 * (( self.alpha * self.trial_number) ** 3)
            #self.epsilon = 1 + (- self.alpha * self.trial_number) ** 3
            
            #logarithmic decay
            #self.epsilon =  math.log(-1 * self.alpha * self.trial_number + 2 ,2 )
            #self.epsilon = math.log( (-1 * (self.alpha * self.trial_number) + 4) *2 + 2,  2) /5.0 
            #self.epsilon = (math.log(-1 *  self.alpha * self.trial_number  + 1, 2) * 3 + 10 ) / 10



        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        state = (inputs['light'], waypoint, inputs['oncoming'], inputs['left'] )

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        # 

        #simple dictionary maximum value
        maxQ = max(self.Q[state].values())

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        if not self.learning:
            return

        initial_state = {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0 }

        if state not in self.Q:
            #Each state will have possible action outcomes after learning
            self.Q[state] = initial_state 

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        # 
        if not self.learning:
            #choose random action from possible valid action list
            action = random.choice(self.valid_actions)
        else:
            #Epsilon <1 so , epsilon * 100 will have the probability
            #at beginning actions will be random
            #Explaining for verbosity (could have better code)
            #toss a number between 0 - 1
            random_percent = random.random() # will be between 0-1   

            if random_percent < (self.epsilon + 0.0005):
                #our guessed random is smaller than epsilon, do a random action
                #Epsilon should be <0 in trials so should not choose here on real test
                action = random.choice(self.valid_actions)
            else:
                actions_possible = []
                maxQ_of_this_state = self.get_maxQ(state)

                for action_choice in self.Q[state]:
                    if self.Q[state][action_choice]  == maxQ_of_this_state:
                        actions_possible.append(action_choice)


                #action table build, choose one randomly
                action = random.choice(actions_possible)



        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')


        

        if self.learning:
            #new_q = (1 - alpha) * old_q + alpha * learned_value 
            old_q = self.Q[state][action]
            self.Q[state][action] = (1 - self.alpha) * old_q + self.alpha  * reward 

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run()


if __name__ == '__main__':
    run()

import random as rand
import math as maths




#My approach to Backpropergation is: 
#1: Feedforward to get the error of the network. 
#2: Go to the outputs neurons/regular neurons that need changing
#3: Find the reverse of the activation function to figure out how much more the neuron needs to be activated.
#4: Find the most activated neurons in prev layer and see if you can change the connection strengths to get the required activation. 
#5: Repeat steps 2-4 until you get to the input layer
#6: Use the found changes to update the network



class network:
    def __init__(self,inputs,outputs,middle):
        self.inputs=inputs
        self.outputs=outputs
        self.middle=middle
        self.connections={}
    def randomise_network(self,strength_range):
        new_network={} #initilise a new empty network
        for input_num in range(self.inputs): #Go through each input neuron
            for middle_num in self.middle[0]: #Go through the first layer of middle neurons
                new_network[(f"i{input_num+1}",middle_num)]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #make the connection random within the specified range

        for middle_column in self.middle: #Go through each middle neuron column
            if middle_column!=self.middle[len(self.middle)-1]: #Check if the column is the last one, if not, continue
                for curr_1 in middle_column: #Go through each neuron in the column
                    for curr_2 in self.middle[self.middle.index(middle_column)+1]: #Go through each neuron in the next column
                        new_network[(curr_1,curr_2)]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #Eandomise the connection in the specified range
            else: #If it is the final column
                for curr_1 in middle_column: #Go through each neuron in the column
                    for output in range(self.outputs): #Go through each neuron in the output
                        new_network[(curr_1,f"o{output}")]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #Randomise the connection in the specified range
        return new_network
    def get_output(self,inputs,network):
        network_status={} #This is where we will save the activation status of the network.
        for input_num in inputs:

                


def backpropergation(network,middle_neurons,desired_output,state):
    """Network has to be a dict formatted like: {("i1","m1"):1.4, etc}, and the middle_neurons has to be a list formatted like [["m1","m2","m4"],["m3","m5"],["m6","m7","m8","m9"]]. This would show that there are 3 columns, the first one with 3 middle neurons, the second with 2, and the third with 4. desired_output is what output you desired it to have, and state is the inputs the network got when you tested it."""
    returned_output=[]



Network=network(4,2,[["m1","m2","m3","m4"],["m5","m6","m7","m8"]])
new_network=Network.randomise_network([-5,5])
print(new_network)
x=0
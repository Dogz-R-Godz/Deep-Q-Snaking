import random as rand
import math as maths
import time
import struct

def read_mnist(images_path, labels_path):
    # Read labels
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = [int.from_bytes(lbpath.read(1), byteorder='big') for _ in range(n)]
    print("Read the labels")
    # Read images
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = [[int.from_bytes(imgpath.read(1), byteorder='big') for _ in range(rows*cols)] for _ in range(num)]
    print("Read the images")
    # Generate expected_outputs and network_inputs lists
    expected_outputs = []
    network_inputs = []

    for label, image in zip(labels, images):
        # Generate expected output dictionary
        expected_output = {f"o{i}": (1 if i == label else 0) for i in range(10)}
        expected_outputs.append(expected_output)

        # Generate network input dictionary
        network_input = {f"i{j}": pixel / 255.0 for j, pixel in enumerate(image)}
        network_inputs.append(network_input)

    return expected_outputs, network_inputs






class network:
    def __init__(self,inputs,outputs,middle):
        self.inputs=inputs
        self.outputs=outputs
        self.middle=middle
    def sigm(self,num,rounding=8):
        """Sigmoid is the activation function that we will use for this neural network. Sigmoid is an non-linear equation that can turn any input number into a number from 0-1."""
        sig1=1/(1+maths.e**-num) #sig=1/(1+maths.e**(-num*0.3673682)) Uncomment this if needed for the super long sigmoid #The sigmoid equation. It uses e, Eulers number (2.718281828459045). The *0.2 is so that the equation ends quite a bit later.
        rounded_sig=round(sig1,rounding)
        return rounded_sig
    
    def RELU(self,num):
        return max(num,num*0.001)

    def dRELU(self,num):
        if num<0:
            return 0.001
        else:
            return 1

    
    def rsigm(self,num,rounding):
        if num==1:
            return 23.7191  #100.0 Uncomment this if needed for the super long sig.
        if num==0:
            return -23.7191 #-64.5647 Uncomment this if needed for the super long sig.
        sig = maths.log(num/(1-num))#/0.3673682 Uncomment this if needed for the super long sig. #The reverse of the sigmoid equation.
        rounded_sig=round(sig,rounding)
        return rounded_sig
    
    def dsigm(self,num,rounding=16):
        deriv=self.sigm(num,16) * (1-self.sigm(num,16))
        rounded_deriv=round(deriv,rounding)
        return rounded_deriv
    
    def randomise_network(self,strength_range,random):
        new_network={} #initilise a new empty network
        for input_num in range(self.inputs): #Go through each input neuron
            for middle_num in self.middle[0]: #Go through the first layer of middle neurons
                if random:
                    new_network[(f"i{input_num}",middle_num)]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #make the connection random within the specified range
                else:
                    new_network[(f"i{input_num}",middle_num)]=0

        for middle_column in self.middle: #Go through each middle neuron column
            if middle_column!=self.middle[len(self.middle)-1]: #Check if the column is the last one, if not, continue
                for curr_1 in middle_column: #Go through each neuron in the column
                    for curr_2 in self.middle[self.middle.index(middle_column)+1]: #Go through each neuron in the next column
                        if random:
                            new_network[(curr_1,curr_2)]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #Randomise the connection in the specified range
                        else:
                            new_network[(curr_1,curr_2)]=0

            else: #If it is the final column
                for curr_1 in middle_column: #Go through each neuron in the column
                    for output in range(self.outputs): #Go through each neuron in the output
                        if random:
                            new_network[(curr_1,f"o{output}")]=round(rand.randint(strength_range[0]*1000,strength_range[1]*1000)/1000,5) #Randomise the connection in the specified range
                        else:
                            new_network[(curr_1,f"o{output}")]=0
        for column in self.middle:
            for neuron in column:
                new_network[neuron]=0
        for neuron in range(self.outputs):
            new_network[f"o{neuron}"]=0
        return new_network
    
    def get_output(self,inputs=None,network=None,activation="sig"):
        """inputs have to be structured like {"i0":0,"i1":1,"i2":1,"i3":0}"""
        if inputs==None:inputs={"i0":0,"i1":1,"i2":1,"i3":0}
        if network==None:network={('i0', 'm0'): -4.074, ('i0', 'm1'): -0.439, ('i0', 'm2'): 4.312, ('i0', 'm3'): 0.082, ('i1', 'm0'): 3.124, ('i1', 'm1'): 4.936, ('i1', 'm2'): 1.297, ('i1', 'm3'): -1.838, ('i2', 'm0'): 3.708, ('i2', 'm1'): 4.164, ('i2', 'm2'): 0.708, ('i2', 'm3'): 2.334, ('i3', 'm0'): -0.86, ('i3', 'm1'): 0.818, ('i3', 'm2'): -3.855, ('i3', 'm3'): 0.45, ('m0', 'm4'): -2.849, ('m0', 'm5'): -1.893, ('m0', 'm6'): 2.064, ('m0', 'm7'): -1.112, ('m1', 'm4'): 2.522, ('m1', 'm5'): -2.14, ('m1', 'm6'): -0.473, ('m1', 'm7'): -3.217, ('m2', 'm4'): 1.115, ('m2', 'm5'): 0.591, ('m2', 'm6'): -4.429, ('m2', 'm7'): 0.657, ('m3', 'm4'): 2.685, ('m3', 'm5'): 2.451, ('m3', 'm6'): -1.794, ('m3', 'm7'): 0.255, ('m4', 'o0'): 4.636, ('m4', 'o1'): 0.326, ('m5', 'o0'): 2.004, ('m5', 'o1'): 0.763, ('m6', 'o0'): 2.772, ('m6', 'o1'): -1.172, ('m7', 'o0'): 2.548, ('m7', 'o1'): 3.98}
        connections={}
        #Finds all the connections from each neuron for later on.
        for connection in network:
            if connection[0] in connections:
                connections[connection[0]].append(connection[1])
            else:
                connections[connection[0]]=[connection[1]]
        network_status={} #This is where we will save the activation status of the network.
        network_order=[]
        network_status_unactivated={} #This is where we will save the non-activated status of the network. This is just so we can use backprop easier.
        #Fill network_status with the inputs, middle, and output neurons.
        #Also it generates the order that the neurons should be read from.
        for input_num in range(self.inputs):
            network_status[f"i{input_num}"]=inputs[f"i{input_num}"]
            network_status_unactivated[f"i{input_num}"]=inputs[f"i{input_num}"]
            network_order.append(f"i{input_num}")
        for middle_col in self.middle:
            for middle_num in middle_col:
                network_status[middle_num]=0
                network_status_unactivated[middle_num]=0
                network_order.append(middle_num)
        for output_num in range(self.outputs):
            network_status[f"o{output_num}"]=0
            network_status_unactivated[f"o{output_num}"]=0
        #Go through the neurons in order. 
        while len(network_order)>0:
            curr_neuron=network_order.pop(0) 
            if curr_neuron[0]!="i": #Check if the neuron is an input neuron
                if activation=="sig":
                    network_status[curr_neuron]=self.sigm(network_status[curr_neuron]+network[curr_neuron],8) #Use the Sigmoid activation function to find the activated value of the neuron
                else:
                    network_status[curr_neuron]=self.RELU(network_status[curr_neuron]+network[curr_neuron])
            for conn_2 in connections[curr_neuron]: #Go through all the connections from the current neuron
                pre_change=network_status[conn_2]
                network_status[conn_2]=round(pre_change+network[(curr_neuron,conn_2)]*network_status[curr_neuron],8) 
                network_status_unactivated[conn_2]=round(pre_change+network[(curr_neuron,conn_2)]*network_status[curr_neuron],8)
        #Find the most activated output
        highest_output_num=-1
        output_chosen={}
        multiple_outputs={}
        for output_num in range(self.outputs):
            if activation=="sig":
                network_status[f"o{output_num}"]=self.sigm(network_status[f"o{output_num}"],8)
            else:
                network_status[f"o{output_num}"]=self.RELU(network_status[f"o{output_num}"])
            multiple_outputs[f"o{output_num}"]=network_status[f"o{output_num}"]
            if network_status[f"o{output_num}"]>highest_output_num:
                output_chosen={f"o{output_num}":network_status[f"o{output_num}"]}
                highest_output_num=network_status[f"o{output_num}"]
            

        
        return output_chosen,multiple_outputs,network_status,network_status_unactivated

    def find_error(self,network,desired_outputs,states):
        total_error=0
        errors={}
        for state in range(len(states)):
            returned_output,full_outputs,status,unactivated_status=self.get_output(states[state],network)

            for x in full_outputs:
                if x in errors.keys():
                    errors[x]+=(desired_outputs[state][x]-full_outputs[x])**2
                else:
                    errors[x]=(desired_outputs[state][x]-full_outputs[x])**2
        errors_2={}
        for x in errors:
            errors_2[x]=errors[x]/len(states)
            total_error+=errors_2[x]
        
        return errors_2,total_error
    

    def dcost(self,output,expected_output,rounding=8):
        deriv=2*(output-expected_output)
        rounded_deriv=round(deriv,rounding)
        return rounded_deriv

    def dz(self,respect,activ,weight):
        """respect = 1 is for activation, respect = 2 is for weight, respect = 3 is for bias."""
        if respect==1: #activation
            return weight
        elif respect==2:
            return activ
        elif respect==3:
            return 1



    def backpropergation(self,network,desired_outputs,states,strength_range_total,learning_rate,activation="sig"):
        """You need to make the desired_outputs and states variables lists of lists."""
        

        connectable_neurons={}
        connected_neurons={}
        for connection in network:
            if type(connection)==tuple:
                if connection[1] in connectable_neurons:
                    connectable_neurons[connection[1]].append(connection[0])
                else:
                    connectable_neurons[connection[1]]=[connection[0]]
                if connection[0] in connected_neurons:
                    connected_neurons[connection[0]].append(connection[1])
                else:
                    connected_neurons[connection[0]]=[connection[1]]

        #Make local variables for input, middle, and output neurons
        outputs=[]
        middle=self.middle
        inputs=[]
        for neuron in range(self.outputs):
            outputs.append(f"o{neuron}")
        for neuron in range(self.inputs):
            inputs.append(f"i{neuron}")

        layers=[outputs]
        middle2=middle.copy()
        middle2.reverse()
        layers+=middle2
        total_error=0
        state_wishes={}
        importance={}
        for state in range(len(states)):
            derivs={}
            print(f"{state}/{len(states)}")
            error_thing=self.find_error(network,[desired_outputs[state]],[states[state]])
            _,output,status,unactiv_status=self.get_output(states[state],network) 
            importance[state]=error_thing[1]
            total_error+=error_thing[1]
            for layer in layers:
                for neuron in layer:
                    for con in connectable_neurons[neuron]:
                        
                        conn=(con,neuron)
                        der_z_w=self.dz(2,status[neuron],network[conn]) #Find the deriv for the unactivated neuron with respect to the weight
                        der_z_b=self.dz(3,status[neuron],network[conn]) #Find the deriv for the unactivated neuron with respect to the bias
                        if activation=="sig":
                            der_s=self.dsigm(unactiv_status[neuron],100) #Find the deriv for the sigmoided neuron with respect to the unactivated neuron
                        else:
                            der_s=self.dRELU(unactiv_status[neuron]) #Find the deriv for the RELUed neuron with respect to the unactivated neuron
                        if layer==outputs:
                            der_c=self.dcost(status[neuron],desired_outputs[state][neuron],100) #Find the deriv for the cost function with respect to the sigmoided neuron\
                            derivs[neuron]=der_z_b*der_s*der_c #Discourage the use of bias in the final layer
                            derivs[conn]=der_z_w*der_s*der_c
                            #Start at output, find deriv of the weights. 
                            #Go back a layer, find deriv of those weights * the output weights derivs.
                        else:
                            deriv_w=der_z_w*der_s #Find the derivative of the sigmoided neuron with respect to the weight
                            deriv_b=der_z_b*der_s #Find the derivative of the sigmoided neuron with respect to the bias.
                            summed_derivs=0
                            for conn_2 in connected_neurons[neuron]:
                                summed_derivs+=derivs[(neuron,conn_2)] #Add up all the derivatives
                            total_deriv_w=deriv_w*summed_derivs #Make a var for the total derivs
                            total_deriv_b=deriv_b*summed_derivs
                            derivs[neuron]=total_deriv_b #Add them to the dict
                            derivs[conn]=total_deriv_w
                            
                        x=0
            for conn in network:
                if state in state_wishes:
                    state_wishes[state][conn]=max(min(-(derivs[conn]),strength_range_total),-strength_range_total)
                else:
                    state_wishes[state]={conn:max(min(-(derivs[conn]),strength_range_total),-strength_range_total)}
        new_network=network.copy()
        for conn in network:
            total_wishes=0
            for wish in state_wishes:
                total_wishes+=state_wishes[wish][conn] #Get the average for the importance
                
            

            gradient=(total_wishes/(len(states)))
            


            new_network[conn]=max(min(new_network[conn]+gradient*learning_rate,strength_range_total),-strength_range_total)

        return new_network
    
    def find_step_rewards(self,replay_buffer,decay_factor):
        rewards={} #New Q value = 
        done={True:1,False:0}
        prev_reward=0
        for replay in replay_buffer:
            if tuple(replay[0].values()) in rewards:
                if replay[5]: #If that move was terminal
                    reward=replay[4]
                    prev_reward=reward
                else:
                    reward=(prev_reward*decay_factor)+replay[4]
                    prev_reward=reward
                    
                    #reward=
                    #reward=((replay[3] + (decay_factor*rewards[tuple(replay_buffer[replay_buffer.index(replay)-1][0].values())][replay[1]]))/2)*(1-done[replay[4]])
                outputs=rewards[tuple(replay[0].values())]
                outputs[replay[2]]=reward
                rewards[tuple(replay[0].values())]=outputs
            else:
                if replay[5]: #If that move was terminal
                    reward=replay[4]
                    prev_reward=reward
                else:
                    reward=(prev_reward*decay_factor)+replay[4]
                    prev_reward=reward
                outputs=replay[3]
                outputs[replay[2]]=reward
                rewards[tuple(replay[0].values())]=outputs
        return list(rewards.keys()), list(rewards.values())
        x=0


print("Thank you for using my incredibly scuffed backpropergation neural network library. Check out my youtube channel: https://www.youtube.com/channel/UCdysJizyP9Ww4UxuAjMjSwA. ")
if __name__=="__main__":
    print(time.time())
    print("started")
    middle_n=[16,16]
    middle=[]
    for neuron in middle_n:
        neurons=[]
        for num in range(neuron):
            neurons.append(f"m{num}")
        middle.append(neurons)
    Network=network(764,10,middle)

    strength_range_total=4
    strength_range=[-strength_range_total,strength_range_total]
    new_network=Network.randomise_network(strength_range)
    
    network_inputs=[] #[{"i0":0, etc}]
    expected_outputs=[]
    #-------------
    #|(0,1)|(1,1)|
    #|-----------|
    #|(0,0)|(1,0)|
    #-------------
    expected_outputs, network_inputs = read_mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

    test_image=network_inputs[0]
    resolution=28
    text=""
    translator={0:" ", 1:"#"}
    for pixel in test_image:
        if len(text)==resolution:
            print(text)
            text=""
        text+=translator[round(test_image[pixel])]
# Now you can use the expected_outputs and network_inputs lists in your neural network.

    
    eo=[]
    ni=[]
    scostic_eo=[]
    scostic_ni=[]
    eo2=expected_outputs.copy()
    ni2=network_inputs.copy()
    counter=0
    while len(eo2)>0:
        counter+=1
        number=eo2.index(rand.choice(eo2))
        eo.append(eo2[number])
        ni.append(ni2[number])
        eo2.pop(number)
        ni2.pop(number)
        if counter==100:
            scostic_ni.append(ni)
            scostic_eo.append(eo)
            eo=[]
            ni=[]
            counter=0
    if counter!=0:
        scostic_ni.append(ni)
        scostic_eo.append(eo)
    new_ni=[scostic_ni[0]]
    new_eo=[scostic_eo[0]]


    starting_time=time.time()
    output=Network.get_output(scostic_ni[0][0],new_network)
    _,initial_error=Network.find_error(new_network,new_eo[0],new_ni[0])
    ending_time=time.time()
    print(f"Found the output for the neural network in {ending_time-starting_time} seconds") #Run and debug found the output for the neural network in 0.0149993896484375 seconds (400 inputs). Run found the output for the neural network in 0.008999109268188477 seconds (400 inputs)
    backprop_loops=5
    strength_range_total=2/(backprop_loops/10)
    for counter in range(backprop_loops):
        for group in range(len(new_eo)):
            new_network=Network.backpropergation(new_network,new_eo[group],new_ni[group],strength_range_total,0.1) #,(strength_range_total*counter/10)+1,(1/(-(counter/50)-4))+0.251)
            print(f"Done mini-batch {group}.")
            error,new_error=Network.find_error(new_network,new_eo[0],new_ni[0])
            print(f"New error: {round(new_error,2)} vs old error: {round(initial_error,2)}. The error in full is {error}")
        

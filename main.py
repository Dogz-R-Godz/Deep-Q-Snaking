#Import all nessessary libraries
import math
import pygame
import random as rand
import neural_network as nn
import pickle

#Initialise Pygame
pygame.init()
#Define colours
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
LIME = ( 150, 255, 150)
RED = ( 255, 0, 0)
BLUE = ( 0, 0, 255)
DARK_GREY = ( 43, 45, 47)
#Define window size and initialise it.
size = (1000, 500)
screen = pygame.display.set_mode(size,pygame.RESIZABLE)


#Set window icon and name
icon = pygame.image.load('icon.png')
pygame.display.set_icon(icon)
pygame.display.set_caption("""Deep Q Snakin! Meet "An AI" (Amazingly named by @That Guy#6482) """) 

#Set variable to check if the program is done or not.
carryOn = True

#Set variable to check if the program is in fullscreen or not
fullscreen = False

#Set the snake body, snake head, apple, and board size vars
board_size=(50,40) #Size is a 50 by 40 board.

square_size=(round(min((size[0]/2)/board_size[0],(size[1])/board_size[1])),round(min((size[0]/2)/board_size[0],(size[1])/board_size[1])))

body=[(25,15),(24,15)]
head=(25,15)
apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))
while apple in body or apple == head:
    apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))




# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()
replay_buffer=[] #replays will have [state, next state, action, outputs, reward, terminal?].

# -------- Main Program Loop -----------
moves={pygame.K_UP:(0,-1),pygame.K_DOWN:(0,1),pygame.K_LEFT:(-1,0),pygame.K_RIGHT:(1,0)}
move=(1,0)
vision_square=21
vision_square_radius=10
#(-1,0) is left, (1,0) is right, (0,1) is down, (0,-1) is up 
valid_moves={(0,-1):[(0,-1),(1,0),(-1,0)], (0,1):[(0,1),(1,0),(-1,0)], (1,0):[(0,1),(0,-1),(1,0)], (-1,0):[(0,1),(0,-1),(-1,0)]}

middle_n=[32,32,24]
middle=[]
total_num=0
for neuron in middle_n:
    neurons=[]
    for num in range(neuron):
        neurons.append(f"m{total_num}")
        total_num+=1
    middle.append(neurons)
E_Greedy=1000
speed=10

reward_decay_rate=0.95
do_E_Greedy=True
if not do_E_Greedy:
    E_Greedy=0
Network=nn.network((vision_square*vision_square)+4,4,middle)
games=0
curr_games=0
apples=0

strength_range_total=2
strength_range=[-strength_range_total,strength_range_total]
new_network=Network.randomise_network(strength_range,True)
mover={"o0":(0,-1),"o1":(1,0),"o2":(0,1),"o3":(-1,0)}
rev_mover={(0,-1):"o0",(1,0):"o1",(0,1):"o2",(-1,0):"o3"}
actions=0
replays_per_pause=10
replay_length=30
hunger=0 #If the hunger is larger than double the area of the board, then it dies.
max_hunger=board_size[0]*board_size[1]*2
font = pygame.font.SysFont(None, 24)
while carryOn:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            carryOn = False 
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if event.key == pygame.K_F11:
                if fullscreen:
                    fullscreen=False
                    screen = pygame.display.set_mode(size)
                    screen = pygame.display.set_mode(size,pygame.RESIZABLE)
                    size2=screen.get_size()
                    square_size=(round(min((size2[0]/2)/board_size[0],(size2[1])/board_size[1])),round(min((size2[0]/2)/board_size[0],(size2[1])/board_size[1])))
                else:
                    fullscreen=True
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    size2=screen.get_size()
                    square_size=(round(min((size2[0]/2)/board_size[0],(size2[1])/board_size[1])),round(min((size2[0]/2)/board_size[0],(size2[1])/board_size[1])))
            if event.key == pygame.K_SPACE:
                if speed==10:
                    speed=30
                elif speed==30:
                    speed=1000
                else:
                    speed=10
            if event.key in moves:
                move=moves[event.key]
            if keys[pygame.K_LCTRL] and keys[pygame.K_s]: #Save the current brain
                with open('current_brain.pickle', 'wb') as handle:
                    pickle.dump(new_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Successfully saved the brain!")
            if keys[pygame.K_LCTRL] and keys[pygame.K_o]: #Open the saved brain
                with open('current_brain.pickle', 'rb') as handle:
                    new_network = pickle.load(handle)
                print("Successfully loaded the brain!")
    
    state={}
    counter=0
    for x in range(vision_square):
        for y in range(vision_square):
            x2=math.ceil(vision_square/2)+x
            y2=math.ceil(vision_square/2)+y
            if (x2,y2) in body:#For the inputs: a wall/out of bounds is 1, the body is 0.75, the apple is 0.5, anything else is 0.
                state[f"i{counter}"]=-1
            elif (x2,y2) == apple:
                state[f"i{counter}"]=1
            else:
                state[f"i{counter}"]=0
            if x2<0 or x2>=board_size[0] or y2<0 or y2>=board_size[1]:
                state[f"i{counter}"]=-1
            
            counter+=1
    state[f"i{counter}"]=0
    state[f"i{counter+1}"]=0
    state[f"i{counter+2}"]=0
    state[f"i{counter+3}"]=0
    #{(0,-1):"o0", (1,0):"o1", (0,1):"o2", (-1,0):"o3"}
    if move==(0,-1):
        state[f"i{counter}"]=1
    elif move==(1,0):
        state[f"i{counter+1}"]=1
    elif move==(0,1):
        state[f"i{counter+2}"]=1
    elif move==(-1,0):
        state[f"i{counter+3}"]=1

    actions+=1
    old_move=move
    fullmoves=Network.get_output(state,new_network)
    fullmove=list(fullmoves[0].keys())
    fullmove=mover[fullmove[0]]
    if fullmove in valid_moves[move]:
        move=fullmove
    else:
        move=old_move
    if rand.randint(0,1000)<E_Greedy: #Make it a toggleable option.
        move=rand.choice(valid_moves[old_move])
    
    E_Greedy*=0.9999
    #Get the output from the neural network


    body.insert(0,head)
    
    head=(head[0]+move[0],head[1]+move[1])

    if head==apple:
        apples+=1
        state2={}
        apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))
        while apple in body or apple == head:
            apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))
        counter=0
        for x in range(vision_square):
            for y in range(vision_square):
                x2=math.ceil(vision_square/2)+x
                y2=math.ceil(vision_square/2)+y
                if (x2,y2) in body:#For the inputs: a wall/out of bounds is 1, the body is 0.75, the apple is 0.5, anything else is 0.
                    state2[f"i{counter}"]=0.75
                elif (x2,y2) == apple:
                    state2[f"i{counter}"]=0.5
                else:
                    state2[f"i{counter}"]=0
                if x2<0 or x2>=board_size[0] or y2<0 or y2>=board_size[1]:
                    state2[f"i{counter}"]=1
                
                counter+=1
        state2[f"i{counter}"]=0
        state2[f"i{counter+1}"]=0
        state2[f"i{counter+2}"]=0
        state2[f"i{counter+3}"]=0
        #{(0,-1):"o0", (1,0):"o1", (0,1):"o2", (-1,0):"o3"}
        if move==(0,-1):
            state2[f"i{counter}"]=1
        elif move==(1,0):
            state2[f"i{counter+1}"]=1
        elif move==(0,1):
            state2[f"i{counter+2}"]=1
        elif move==(-1,0):
            state2[f"i{counter+3}"]=1
        reward=0.5
        hunger=0
        replay_buffer.append((state,state2,rev_mover[move],fullmoves[1],reward,False))
        
    elif head in body or head[0]<0 or head[0]>=board_size[0] or head[1]<0 or head[1]>=board_size[1] or hunger>=max_hunger:
        games+=1
        curr_games+=1
        print(f"Done {games} games, and have gotten {apples} apples during those games.")
        reward=-1
        replay_buffer.append((state,{},rev_mover[move],fullmoves[1],reward,True))
        #reset the board
        hunger=0
        body=[(25,15),(24,15)]
        head=(25,15)
        apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))
        while apple in body or apple == head:
            apple=(rand.randint(0,board_size[0]-1),rand.randint(0,board_size[1]-1))
        move=(1,0)
        if actions>500 and curr_games>5:
            curr_games=0
            buffer_2=replay_buffer.copy()
            buffer_2.reverse() #Reverse it so we can go in reverse.
            print("About to find the rewards for each step")
            buffer_2_2=buffer_2.copy()

            states,rewards=Network.find_step_rewards(buffer_2,reward_decay_rate)
            #Make the states work with the inputs.
            inputs=list(buffer_2[0][0].keys())
            states=list(states)
            final_states=[]
            for stater in states:
                final_states.append(dict(zip(inputs,stater)))

            print("Found the rewards for each timestep")
            _,error=Network.find_error(new_network,rewards,final_states)
            print("Found the initial error")
            new_network=Network.backpropergation(new_network,rewards,final_states,strength_range_total,0.1)
            print(f'Done')
            _,new_error=Network.find_error(new_network,rewards,final_states)
            print(f"Old error: {error}. New error: {new_error}")
            

            replay_buffer=[]
            actions=0

    else:
        body.pop()
        state2={}
        counter=0
        for x in range(vision_square):
            for y in range(vision_square):
                x2=math.ceil(vision_square/2)+x
                y2=math.ceil(vision_square/2)+y
                if (x2,y2) in body:#For the inputs: a wall/out of bounds is 1, the body is 0.75, the apple is 0.5, anything else is 0.
                    state2[f"i{counter}"]=0.75
                elif (x2,y2) == apple:
                    state2[f"i{counter}"]=0.5
                else:
                    state2[f"i{counter}"]=0
                if x2<0 or x2>=board_size[0] or y2<0 or y2>=board_size[1]:
                    state2[f"i{counter}"]=1
                
                counter+=1
        state2[f"i{counter}"]=0
        state2[f"i{counter+1}"]=0
        state2[f"i{counter+2}"]=0
        state2[f"i{counter+3}"]=0
        #{(0,-1):"o0", (1,0):"o1", (0,1):"o2", (-1,0):"o3"}
        if move==(0,-1):
            state2[f"i{counter}"]=1
        elif move==(1,0):
            state2[f"i{counter+1}"]=1
        elif move==(0,1):
            state2[f"i{counter+2}"]=1
        elif move==(-1,0):
            state2[f"i{counter+3}"]=1
        reward=-0.01
        replay_buffer.append((state,state2,rev_mover[move],fullmoves[1],reward,False))
        
        hunger+=1
    




    screen.fill(DARK_GREY)
    size1=min(vision_square,(board_size[0]-head[0])+vision_square_radius)
    size2=min(vision_square,(board_size[1]-head[1])+vision_square_radius)
    rendering_viewing_area=[
                            max(head[0]-vision_square_radius,0)*square_size[0],
                            max(head[1]-vision_square_radius,0)*square_size[1],
                            size1*square_size[0],
                            size2*square_size[1]
                            ]
    pygame.draw.rect(screen,WHITE,rendering_viewing_area,0)
    #Dedicate half of the screen to the game. Dedicate the other half to the neural network
    for x in range(board_size[0]):
        for y in range(board_size[1]):
            if (x,y) == head:
                pygame.draw.rect(screen,LIME,[square_size[0]*x,square_size[1]*y,square_size[0],square_size[1]],0)
            elif (x,y) in body:
                pygame.draw.rect(screen,GREEN,[square_size[0]*x,square_size[1]*y,square_size[0],square_size[1]],0)
            elif (x,y) == apple:
                pygame.draw.rect(screen,RED,[square_size[0]*x,square_size[1]*y,square_size[0],square_size[1]],0)


    pygame.draw.rect(screen,BLACK,[0,0,board_size[0]*square_size[0],board_size[1]*square_size[1]],1)
    pygame.display.update()
    pygame.display.flip()
     
    clock.tick(speed)
 
pygame.quit()



import os
import shutil
import numpy as np
import random

def play_game():
   game_history = []
   initial_state = np.array([0, 0, 1, 0, 0])
   game_history.append(initial_state)
   probability = 0.5
   current_state = 2
   while current_state != -1 and current_state != len(initial_state):
      # Less than 0.5 is left, greater than or equal to is right:
      direction = random.uniform(0, 1)
      if direction < probability:
         # Left:
         current_state -= 1
      else:
         # Right:
         current_state += 1
      if current_state == -1: # Terminated left
         label_state = np.array([0, 0, 0, 0, 0])
      elif current_state == len(initial_state): # Terminated right
         label_state = np.array([1, 1, 1, 1, 1])
      else:
         new_state = np.array([0, 0, 0, 0, 0])
         new_state[current_state] = 1
         game_history.append(new_state)
         if sum(new_state) != 1:
            print("Error, something went wrong")
            exit(0)
   # Write the label line:
   game_history.append(label_state)
   return game_history


if __name__ == '__main__':
   
   n_data_sets = 100
   start_increment = 1
   num_sequences_per_set = 10
   output_directory = 'training_data'
   actual_num_right = 0
   actual_num_left = 0
   average_num_right = 0
   average_num_left = 0
   for i in range(n_data_sets):
      output_dir_name = "data_set_"+str(start_increment+i)
      rel_path = os.path.join(output_directory,output_dir_name)
      # Make the directory:
      if os.path.exists(rel_path):
         shutil.rmtree(rel_path)
      os.mkdir(rel_path)
      # We want to have each training set have equal number
      restrict = True
      # of left side ends as right side ends
      #num_right = int(num_sequences_per_set/4)
      #num_left = int(num_sequences_per_set/2)
      #num_left = num_sequences_per_set - num_right
      #num_left = int(num_sequences_per_set*0.5)
      num_right = int(num_sequences_per_set*0.5)
      num_left = num_sequences_per_set - num_right
      if restrict:
         # Write the left ending test cases:
         for j in range(num_left):
            data_set_enum = j+1 # For writing the file
            temp = play_game()
            while temp[-1][0] != 0:
               temp = play_game()
            file_name = "sequence"+str(data_set_enum)+".txt"
            rel_path_file = os.path.join(rel_path,file_name)
            with open(rel_path_file,'w') as the_file:
               for item in temp:
                  the_file.write("%s\n"%' '.join(map(str, item)))
            actual_num_left+=1
         for j in range(num_right):
            data_set_enum = num_left+j+1 # For writing the file
            temp = play_game()
            while temp[-1][0] != 1:
               temp = play_game()
            file_name = "sequence"+str(data_set_enum)+".txt"
            rel_path_file = os.path.join(rel_path,file_name)
            with open(rel_path_file,'w') as the_file:
               for item in temp:
                  the_file.write("%s\n"%' '.join(map(str, item)))
            actual_num_right+=1
      else:
         # If not restricting number of left ends and right ends
         for j in range(num_sequences_per_set):
            data_set_enum = j+1 # For writing the file
            temp = play_game()
            file_name = "sequence"+str(data_set_enum)+".txt"
            rel_path_file = os.path.join(rel_path,file_name)
            with open(rel_path_file,'w') as the_file:
               for item in temp:
                  the_file.write("%s\n"%' '.join(map(str, item)))
            if temp[-1][0] == 0:
               actual_num_left+=1
            else:
               actual_num_right+=1
   print("The number of samples ended on the left/right side: {0} / {1}".format(actual_num_left,actual_num_right))
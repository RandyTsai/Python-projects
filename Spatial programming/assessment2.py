# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:38:44 2021

@author: Randy MSI
"""

import csv
import sys
import math
from pyproj import Transformer



"""ASM4.1=========================================================="""

def import_csv(filename):
    
    nested_list = []
    
    with open(filename, 'r', newline='') as file:
        data = csv.reader(file)
        for row in data:
            nested_list.append(row)
        
    return nested_list
    
# import_csv('trajectory_data.csv')
    
    

def project_coordinate(from_epsg, to_epsg, in_x, in_y):  
    
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True) 
                                                #from epsg4326 to epsg4796,  always_xy=True is longitude, latitude for 
                                                #geographic CRS and easting, northing for most projected CRS. Default is false.    
    
    return transformer.transform(in_x, in_y)
    
# project_coordinate('epsg:4326', 'epsg:4796', 116.31, 39.98) #注意 csv latitu緯度先 latitude是y
# not sure correct?    
    


def output_geoFile():    
    
    #start csv
    value = input('please enter original read-from file which under the same folder:\n')
    print(f'you entered {value}, wait for minutes to ouput the file')
    
    result_list = import_csv(value)
    field = result_list[0] #field name       
    rows = result_list[1:] #field data
    
    #append data to list
    field.append('X_UTM')
    field.append('Y_UTM')
    for row in rows:
        x, y = project_coordinate('epsg:4326', 'epsg:4796', row[3], row[2]) #row[2]=lati=y, row[3]=longti=x
        row.append(str(x).strip())
        row.append(str(y).strip())
       
    #write csv
    with open('assessment2.csv', 'w', newline='') as file: #prevent from a blank line between rows
        write = csv.writer(file)
        write.writerow(field)
        write.writerows(rows)
  
    return None
    
# output_geoFile() 



"""ASM4.2=========================================================="""

def compute_distance(from_x, from_y, to_x, to_y):
        
    from_x = float(from_x)
    from_y = float(from_y)
    to_x = float(to_x)
    to_y = float(to_y)
        
    if not 0< from_x <834000: sys.exit("%s out of the range of easting value 0~834000" %from_x)
    if not 0< to_x <834000: sys.exit("%s out of the range of easting value 0~834000" %to_x)
    if not 0< from_y <10000000: sys.exit("%s out of the range of northing value 0~1000000" %from_y)
    if not 0< to_y <10000000: sys.exit("%s out of the range of northing value 0~1000000" %to_y)
    
    Dst = (from_x - to_x)**2+(from_y - to_y)**2    
    return math.sqrt(Dst)
    
# compute_distance(10, 3, 1, 3) 



def compute_time_difference(start_time, end_time):    
    
    start_time = str(start_time).split(':')
    start_time = list(map(int, start_time))
    sys.exit("'start_time' time format is not correct") if len(start_time)>3 else print('',end='') #sanitize
    if not (0<=start_time[0]<=24 and 0<=start_time[1]<60 and 0<=start_time[2]<60): sys.exit("'start_time' time format is not correct") #sanitize
    
    start_time_sec = start_time[0]*3600 + start_time[1]*60 +  start_time[2]
    
    end_time = str(end_time).split(':')
    end_time = list(map(int, end_time))
    sys.exit("'end_time' time format is not correct") if len(end_time)>3 else print('',end='') #sanitize
    if not (0<=end_time[0]<=24 and 0<=end_time[1]<60 and 0<=end_time[2]<60): sys.exit("'end_time' time format is not correct") #sanitize
    
    end_time_sec = end_time[0]*3600 + end_time[1]*60 +  end_time[2]
    
    return abs(end_time_sec - start_time_sec)
    
# compute_time_difference('10:18:38', '10:20:38')   
     
    

def compute_speed(total_distance, total_time):    
        
    return round(total_distance / int(total_time), 2)
    
# compute_speed(1010, 33)   
    

    
    
"""ASM4.3=========================================================="""  



'''   
extract one trajectory according to traj_id, row number from the data_rows
cur_trajID : trajectoryID, type string, e.g '0' or '1' or '10'
row_start_num : the row started index of the trajectory(1st row field number not count), type number, 
                e.g. in asm2.csv, trajectory'0' starts at row0 , trajectory'1' starts at row10
trajectories : nested list of numeric data (without the original csv 1st row of field name)
'''
def extract_traj(cur_trajID, row_start_num, trajectories):
    trajlist = []
    for row in trajectories[row_start_num:]: 
        
        if row[0] == cur_trajID: #compare id
            trajlist.append(row)
            row_start_num+=1
            if row_start_num+1 > len(trajectories): #important check                 
                return trajlist, row_start_num            
        else:
            #print('will return: ',trajlist,' and start rowID: ', row_start_num)
            return trajlist, row_start_num
    return None
        
# =============================================================================
# tmplist, tmpnum = extract_traj('10', 309, data_rows)  
# print(tmplist)
# print(tmpnum)     
# =============================================================================



# compute single trajectory length by sum of its segments 4.3.1
def compute_traj_len(trajectory): 
    segments_len = []    
    for i in range(0, len(trajectory)-1):
        seg_len = compute_distance(trajectory[i][6], trajectory[i][7], trajectory[i+1][6], trajectory[i+1][7]) #index change depend on csv field
        segments_len.append(seg_len)
    
    return round(sum(segments_len), 2)

# =============================================================================
# tmplist, tmpnum = extract_traj('10', 309, data_rows)  
# compute_traj_len(tmplist)     
# =============================================================================



# find a longest segment in a single trajectory 4.3.2
# input a single trajectory with its nodes
def compute_longest_seg(trajectory):
    
    segments_len = {} #dictionary    
    for i in range(0, len(trajectory)-1):
        seg_len = compute_distance(trajectory[i][6], trajectory[i][7], trajectory[i+1][6], trajectory[i+1][7]) #index change depend on csv field
        segments_len[i] = seg_len #notice the key is the nodeID that this seg begin, and 10 nodes have 9 segments
    
    max_value_key = max( segments_len, key=segments_len.get ) #key= use dic.get() function (return all values) as basis for comparison
    return max_value_key, round(segments_len[max_value_key], 2)

# =============================================================================
# extracted_traj, tmpnum = extract_traj('0', 0, data_rows)
# a, b = compute_longest_seg(extracted_traj)
# print('the longest segment index=',a, ' the value=',b)
# =============================================================================



#4.3.3
def avg_sampling_rate(trajectory):
    
    total_time_dif = []
    for i in range(0, len(trajectory)-1):        
        time_dif = compute_time_difference(trajectory[i][5], trajectory[i+1][5])
        total_time_dif.append(time_dif)
    
    samples = len(trajectory) #how many samples, and samples-1 = how many time gaps
    
    return round(sum(total_time_dif) / (samples-1), 2)
    
# =============================================================================
# extracted_traj, tmpnum = extract_traj('0', 0, data_rows)
# print('average sample every', avg_sampling_rate(extracted_traj), 'seconds')  #4.3.3 answer
# =============================================================================



#4.3.4
def compute_extreme_speed(trajectory):
    
    speeds = []
    
    for i in range(0, len(trajectory)-1):
        seg_len = compute_distance(trajectory[i][6], trajectory[i][7], trajectory[i+1][6], trajectory[i+1][7])
        time_dif = compute_time_difference(trajectory[i][5], trajectory[i+1][5])
        
        speed = compute_speed(seg_len, time_dif)
        speeds.append(speed)
        
    return min(speeds), speeds.index(min(speeds)), max(speeds), speeds.index(max(speeds))

# =============================================================================
# extracted_traj, tmpnum = extract_traj('1', 10, data_rows)
# min_speed, min_idx, max_speed, max_idx = compute_extreme_speed(extracted_traj) 
# print(f'the min speed={min_speed}m/s belongs to idx:{min_idx} segment, and max speed ={max_speed}m/s belongs to idx:{max_idx} segment')    
#     
# =============================================================================
    


# compute length sum of all trajectories 4.3.5(a)
def compute_len_sum_trajectories(given_id_list, trajectories):   
    
    row_start_num = 0
    trajs_len = []
    if(trajectories[row_start_num]):
        for i in given_id_list:             
            #print('process trajectoryID = ',i,' and row_start_num = ',row_start_num)
            tmplist, tmpnum = extract_traj(i, row_start_num, trajectories)        
            length = compute_traj_len(tmplist)
            trajs_len.append(length)
            row_start_num = tmpnum    
    
    return round(sum(trajs_len), 2)

# =============================================================================        
# length = compute_len_sum_trajectories(dist_traj_id, data_rows)
# print('the total lengh of all trjectory is: ', length, ' meters') 
# =============================================================================



# compute length sum of all trajectories 4.3.5(b)
def compute_longest_traj(given_id_list, trajectories):
    
    row_start_num = 0
    trajs_len = [] #length of all trajectories, 4.3.5(a)
    trajs_time = [] #time of all trajectories
    if(trajectories[row_start_num]):
        for i in given_id_list: 
            
            #process distance, append to distance list
            current_traj, tmpnum = extract_traj(i, row_start_num, trajectories)        
            length = compute_traj_len(current_traj)
            trajs_len.append(length) 
            row_start_num = tmpnum 
            
            #process total time of single traj, append to time list
            seg_time_diff = []
            for i in range(0, len(current_traj)-1):
                time_dif = compute_time_difference(current_traj[i][5], current_traj[i+1][5])
                seg_time_diff.append(time_dif)
            trajs_time.append( sum(seg_time_diff) )
    
    longest_traj = max(trajs_len)
    longest_traj_idx = trajs_len.index(longest_traj)
    avg_speed = round( longest_traj / trajs_time[longest_traj_idx], 2)
            
    return longest_traj, longest_traj_idx, avg_speed

# ============================================================================= 
# a,b,c = compute_longest_traj(dist_traj_id, data_rows)
# print(f'the longest trajectory dist={a} meters belongs to idx:{b} trajectory, avg speed={c}m/s')
# =============================================================================



#4.3.5(c)
def print_asses2_out(dist_traj_id, trajectories):
    '''Compute the total length of all trajectories.'''           
    length = compute_len_sum_trajectories(dist_traj_id, trajectories)
    #print('the total lengh of all trjectory is: ', length, ' meters') 
    
    '''Identify the index of the longest trajectory, and compute its average speed.'''   
    a,b,c = compute_longest_traj(dist_traj_id, trajectories)
    #print(f'the longest trajectory dist={a} meters belongs to idx:{b} trajectory, avg speed={c}m/s')
    
    f = open("assessment2_out.txt", "w")
    f.write('total_length_trajs,  the_longest_traj_dst,  longest_traj_idx,  avg_speed\n')
    f.write(f'{round(length,2)},   {round(a, 2)},   {round(b,2)},   {round(c,2)} \n')
    f.close()

    return None







def main():
    
    #setup constant and preprocess nessesary data
    #compute total length
    csv_datalist = import_csv('assessment2.csv')  #!!!NOTICE!!!!, needs to generate assessment2.csv file in advance by executing ASM4.1 above
    field_name = csv_datalist[0] #str name
    data_rows = csv_datalist[1:] #all numeric data
    flip_data_rows = list(zip(*data_rows))
    
    dist_traj_id = list(set(flip_data_rows[0])) #unique value
    dist_traj_id = list(map(int, dist_traj_id)) #to int
    dist_traj_id.sort() #sort by int value
    dist_traj_id = list(map(str, dist_traj_id)) #back to string 
     
    
    row_start_num = 0
    trajs_len = []
    if(data_rows[row_start_num]):
        for i in dist_traj_id:             
            #print('process trajectoryID = ',i,' and row_start_num = ',row_start_num)
            extracted_traj, tmpnum = extract_traj(i, row_start_num, data_rows)        
            
            traj_len = compute_traj_len(extracted_traj) # task 4.3.1
            print(f"Trajectory {int(i)+1}'s length is {traj_len}m.")
            
            idx, val = compute_longest_seg(extracted_traj) # task 4.3.2
            print(f"The length of its longest segment is {val}m, and the index is {int(idx)+1}.")
                        
            avg_rate = avg_sampling_rate(extracted_traj) # task 4.3.3
            print(f"The average sampling rate for the trajectory is {avg_rate}s." )
            
            min_speed, min_idx, max_speed, max_idx = compute_extreme_speed(extracted_traj) # task 4.3.4
            print(f'For the segment index {int(min_idx)+1}, the minimal travel speed is reached.')    
            print(f'For the segment index {int(max_idx)+1}, the maximum travel speed is reached.') 
            
            row_start_num = tmpnum
            print('----')
    
    # task 4.3.5a        
    sum_length = compute_len_sum_trajectories(dist_traj_id, data_rows)
    print(f'The total length of all trajectories is {sum_length}m.')      
    
    # task 4.3.5b
    a,b,c = compute_longest_traj(dist_traj_id, data_rows)
    print(f'The index of the longest trajectory is {b+1}, and the average speed along the trajectory is {c}m/s.')
    
    # task 4.3.5c
    print_asses2_out(dist_traj_id, data_rows)

    #Finished
    print ("Program complete")

    
    
if __name__ == '__main__ ':
    main ()



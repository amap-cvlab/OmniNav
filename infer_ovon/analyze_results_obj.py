import os
import argparse
import json
import math

def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
)
args = parser.parse_args()

jsons = os.listdir(os.path.join(args.path, 'log'))
succ = 0
spl = 0
distance_to_goal = 0
j_list = []
for j in jsons:
    with open(os.path.join(args.path, 'log', j)) as f:
        try:
            data = json.load(f)
            succ += check_inf_nan(int(data['success']))
            if int(data['success'])==0:
                j_list.append(j)
            spl += check_inf_nan(data['spl'])
            distance_to_goal += check_inf_nan(data['distance_to_goal'])
            
            
            
        except:
            print(j)
print(f'Success rate: {succ}/{len(jsons)} ({succ/len(jsons):.3f})')
print(f'SPL: {spl:.3f}/{len(jsons)} ({spl/len(jsons):.3f})')
print(f'Distance to goal: {distance_to_goal/len(jsons):.3f}')
# print(sorted(j_list))
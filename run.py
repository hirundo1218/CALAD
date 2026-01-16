import os
import pandas



### MSL ###
for file_name in ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']:
    print(file_name)
    os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/msl.yml --fname {file_name}')

### SMAP ###
# for file_name in ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 'D-3', 'D-4', 'A-2', 'A-3',' A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'P-2', 'A-8', 'A-9', 'F-3']:
#     print(file_name)
#     os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/smap.yml --fname {file_name}')

### SMD ###
# for file_name in ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-1-4.txt', 'machine-1-5.txt', 'machine-1-6.txt', 'machine-1-7.txt', 'machine-1-8.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt', 'machine-2-4.txt', 'machine-2-5.txt', 'machine-2-6.txt', 'machine-2-7.txt', 'machine-2-8.txt', 'machine-2-9.txt', 'machine-3-1.txt', 'machine-3-2.txt', 'machine-3-3.txt', 'machine-3-4.txt', 'machine-3-5.txt', 'machine-3-6.txt', 'machine-3-7.txt', 'machine-3-8.txt', 'machine-3-9.txt', 'machine-3-10.txt', 'machine-3-11.txt']:
#     print(file_name)
#     os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/smd.yml --fname {file_name}')

### SWAT ###
# file_name='swat'
# print(file_name)
# os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/swat.yml --fname {file_name}')



### MSL ###
for file_name in ['M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']:
    print(file_name)
    os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/msl.yml --fname {file_name}')

### SMAP ###
# for file_name in ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 'D-3', 'D-4', 'A-2', 'A-3',' A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'P-2', 'A-8', 'A-9', 'F-3']:
#     print(file_name)
#     os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/smap.yml --fname {file_name}')

### SMD ###
# for file_name in ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-1-4.txt', 'machine-1-5.txt', 'machine-1-6.txt', 'machine-1-7.txt', 'machine-1-8.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt', 'machine-2-4.txt', 'machine-2-5.txt', 'machine-2-6.txt', 'machine-2-7.txt', 'machine-2-8.txt', 'machine-2-9.txt', 'machine-3-1.txt', 'machine-3-2.txt', 'machine-3-3.txt', 'machine-3-4.txt', 'machine-3-5.txt', 'machine-3-6.txt', 'machine-3-7.txt', 'machine-3-8.txt', 'machine-3-9.txt', 'machine-3-10.txt', 'machine-3-11.txt']:
#     print(file_name)
#     os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/smd.yml --fname {file_name}')

### SWAT ###
# file_name='swat'
# print(file_name)
# os.system(f'CUDA_VISIBLE_DEVICES=0 python classification.py --config_env configs/env.yml --config_exp configs/classification/swat.yml --fname {file_name}')
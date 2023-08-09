import numpy as np 

import matplotlib.pyplot as plt 


all_gtkp = []
for i in range(205, 11095):
    with open(f'./data/label_kp/img_{i}.txt') as fgtkp:
        gtkp = fgtkp.readlines()
        gtkp = [elem.replace(' ','').split(',') for elem in gtkp]
        gtkp = [[float(val) for val in elem] for elem in gtkp]
        gtkp = gtkp[:14]
        all_gtkp.append(gtkp)
all_gtkp = np.array(all_gtkp)
estkp = np.load("estimation.npy")[:all_gtkp.shape[0],:,:]

plt.figure()
plt.plot(np.linalg.norm(estkp-all_gtkp, axis=(1,2), ord=np.float('inf')), label='Errors')
plt.xlabel('Time step', fontsize=20)
plt.ylabel('Keypoint detection error', fontsize=20)
plt.legend()
plt.savefig('Keypoint_detection_error.png')

for i in range(0,14):
    if i==7:
        print("here")
    plt.figure()
    plt.plot(np.linalg.norm((estkp-all_gtkp)[:,i,:], axis=1, ord=np.float('inf')))
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel(f'Keypoint {i} detection error', fontsize=20)
    # plt.legend()
    plt.savefig(f'Keypoint_{i}_detection_error.png')

plt.show()
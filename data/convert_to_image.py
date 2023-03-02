import numpy as np 
import cv2
from scipy.spatial.transform import Rotation 

def gaussian(xL, yL, sigma, H, W):

    grid = np.meshgrid(list(range(W)), list(range(H)))
    channel = np.exp(-((grid[0] - xL) ** 2 + (grid[1] - yL) ** 2) / (2 * sigma ** 2))
    # channel = [np.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    # channel = np.array(channel, dtype=np.float32)
    # channel = np.reshape(channel, newshape=(H, W))

    return channel

def convertToHM(H, W, keypoints, sigma=5):
    # H = img.shape[0]
    # W = img.shape[1]
    nKeypoints = len(keypoints)

    img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

    for i in range(0, nKeypoints // 2):
        x = keypoints[i * 2]
        y = keypoints[1 + 2 * i]

        channel_hm = gaussian(x, y, sigma, H, W)

        img_hm[:, :, i] = channel_hm
    
    # img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints // 2, 1))
    img_hm = img_hm.transpose((2,0,1))
    return img_hm



def convert_to_image(world_pos, ego_pos, ego_ori):
    objectPoints = np.array(world_pos) 
    R = Rotation.from_quat(ego_ori)
    R_euler = R.as_euler('xyz')
    R_euler[-1] = np.pi+0.02
    R = Rotation.from_euler('xyz', R_euler)
    R2 = Rotation.from_euler('xyz',[-np.pi/2, -np.pi/2, 0])
    # Rm2 = R2.as_matrix()
    # Rm = R.as_matrix()
    R_roted = R2*R
    rvec = R_roted.as_rotvec()
    tvec = -R_roted.apply(np.array(ego_pos))
    cameraMatrix = np.array([[205.46963709898583, 0.0, 320], [0.0, 205.46963709898583, 240], [0.0, 0.0, 1.0]])
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pnt,_ = cv2.projectPoints(objectPoints,rvec,tvec,cameraMatrix,distCoeffs)
    return pnt  

if __name__ == "__main__":
    for i in range(0, 4000):
        idx = 3500+i
        print(idx)
        if idx == 6910:
            print("stop")

        with open('./landing_devel/data/data.txt','r') as f:
            data = f.read()
            data = data.strip('\n').split('\n')
            pose = data[idx]
            pose = pose.split(',')
            pose = [float(elem) for elem in pose]
        
        pix1 = [290, 327]
        pix2 = [285, 346]
        pix3 = [299, 346]
        pix4 = [304, 327]

        pix5 = [329, 327]
        pix6 = [333, 346]
        pix7 = [348, 346]
        pix8 = [340, 327]

        pix9 = [310, 442]
        pix10 = [314, 295]

        pix11 = [315, 323]

        kp1 = [1288.160000, 27.179600, 0.500000] # (290, 327)
        kp2 = [1311.840210, 27.275831, 0.500000] # (285, 346)
        kp3 = [1311.708862, 33.348366, 0.500000] # (299, 346)
        kp4 = [1288.055908, 33.212242, 0.500000] # (304, 327)

        kp5 = [1287.930908, 49.299370, 0.500000] # (329, 327)
        kp6 = [1311.833740, 49.194279, 0.500000] # (333, 346)
        kp7 = [1311.834595, 55.323536, 0.500000] # (348, 346)
        kp8 = [1288.078247, 55.501751, 0.500000] # (340, 327)

        kp9 = [1358.607788, 39.831665, 0.500000] # (310, 442)
        kp10 = [1205.007568, 40.819168, 0.500000] # (314, 295)

        kp11 = [1281.550781, 40.233845, 0.500000]

        object_points = np.array([kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10], dtype=np.float32)
        image_points = np.array([pix1,pix2,pix3,pix4,pix5,pix6,pix7,pix8,pix9,pix10], dtype=np.float32)

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        #     [object_points],
        #     [image_points],
        #     (640,480),
        #     np.array([[205.46963709898583, 0.0, 320], [0.0, 205.46963709898583, 240], [0.0, 0.0, 1.0]], dtype=np.float32), 
        #     np.array([0,0,0,0,0], dtype=np.float32),
        #     flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC
        # )
        # cv2.projectPoints(np.array(kp11),rvecs[0],tvecs[0],mtx,dist)
        offset_vec = np.array([-1.1*0.20,0,0.8*0.77])
        drone_pos = np.array([pose[1], pose[2], pose[3]]) # x,y,z
        drone_ori = [pose[4], pose[5], pose[6], pose[7]] # x,y,z,w
        R = Rotation.from_quat(drone_ori)
        # offset_vec = R.inv().apply(offset_vec)
        drone_pos += offset_vec
        p1 = convert_to_image(kp1, drone_pos, drone_ori)
        p2 = convert_to_image(kp2, drone_pos, drone_ori)
        p3 = convert_to_image(kp3, drone_pos, drone_ori)
        p4 = convert_to_image(kp4, drone_pos, drone_ori)
        p5 = convert_to_image(kp5, drone_pos, drone_ori)
        p6 = convert_to_image(kp6, drone_pos, drone_ori)
        p7 = convert_to_image(kp7, drone_pos, drone_ori)
        p8 = convert_to_image(kp8, drone_pos, drone_ori)
        p9 = convert_to_image(kp9, drone_pos, drone_ori)
        p10 = convert_to_image(kp10, drone_pos, drone_ori)

        u1 = 640-int(p1[0][0][0])
        v1 = 480-int(p1[0][0][1])
        u2 = 640-int(p2[0][0][0])
        v2 = 480-int(p2[0][0][1])
        u3 = 640-int(p3[0][0][0])
        v3 = 480-int(p3[0][0][1])
        u4 = 640-int(p4[0][0][0])
        v4 = 480-int(p4[0][0][1])
        u5 = 640-int(p5[0][0][0])
        v5 = 480-int(p5[0][0][1])
        u6 = 640-int(p6[0][0][0])
        v6 = 480-int(p6[0][0][1])
        u7 = 640-int(p7[0][0][0])
        v7 = 480-int(p7[0][0][1])
        u8 = 640-int(p8[0][0][0])
        v8 = 480-int(p8[0][0][1])
        u9 = 640-int(p9[0][0][0])
        v9 = 480-int(p9[0][0][1])
        u10 = 640-int(p10[0][0][0])
        v10 = 480-int(p10[0][0][1])
        img_fn = f'./landing_devel/imgs/img_{idx}.png'
        img = cv2.imread(img_fn)
        cv2.line(img, (u1,v1), (u2,v2),(0,0,255))
        cv2.line(img, (u2,v2), (u3,v3),(0,0,255))
        cv2.line(img, (u3,v3), (u4,v4),(0,0,255))
        cv2.line(img, (u4,v4), (u1,v1),(0,0,255))
        cv2.line(img, (u5,v5), (u6,v6),(0,0,255))
        cv2.line(img, (u6,v6), (u7,v7),(0,0,255))
        cv2.line(img, (u7,v7), (u8,v8),(0,0,255))
        cv2.line(img, (u8,v8), (u5,v5),(0,0,255))
        cv2.line(img, (u9,v9), (u10,v10),(0,0,255))
        cv2.imshow('aa', img)
        # cv2.waitKey(3)

        hms = convertToHM(480, 640, [u1,v1,u2,v2,u3,v3,u4,v4,u5,v5,u6,v6,u7,v7,u8,v8], sigma=2)
        # hm_img = np.vstack((np.hstack(hms[0:4,:,:]), np.hstack(hms[4:,:,:])))
        hm_img = np.sum(hms, axis=0)
        cv2.imshow('bb', hm_img)
        cv2.waitKey(3)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"./tmp/img_marker_{idx}.png",img)
        for j in range(hms.shape[0]):
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./tmp/img_{idx}_{j}.png",hms[j,:,:]*255)
        cv2.imwrite(f"./tmp/img_{idx}_hm.png", hm_img*255)
        with open(f"./tmp/img_{idx}.txt", "w+") as f:
            f.write(f"{u1},{v1},{u2},{v2},{u3},{v3},{u4},{v4},{u5},{v5},{u6},{v6},{u7},{v7},{u8},{v8}")
        # import matplotlib.pyplot as plt 
        # plt.imshow(img_rgb)
        # plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import numba
from numba import jit

frame0_blur = cv2.imread('frame0_blur.jpg') / 255.0
frame0_blur = cv2.resize(frame0_blur, (98, 98))

N = 6
M = 6
padding = 4
interval = 8
W = 49
H = 49

print("W", W, "H", H)


x = np.arange(0, N, 1) * interval + padding
y = np.arange(0, M, 1) * interval + padding
xx, yy = np.meshgrid(x, y)

# xx, yy = (np.random.random([N, M])*W).astype(np.int), (np.random.random([N, M])*H).astype(np.int)

def draw_square(img, x, y, mkr_size, xx, yy, theta):
    w, h = img.shape[0], img.shape[1]
    mkr_size_large = mkr_size * 2**0.5
    
    
#     lx_raw: the left boundary of the marker based on original marker size
    lx_raw, rx_raw = x - mkr_size, x + mkr_size
    ly_raw, ry_raw = y - mkr_size, y + mkr_size
    
#     lx: the leftmost boundary of the marker after rotation
    lx, rx = x - mkr_size_large , x + mkr_size_large
    ly, ry = y - mkr_size_large, y + mkr_size_large
    
#     remove the area outside the canvas
    lx, rx = np.clip(lx, 0, w), np.clip(rx, -1, w-1)
    ly, ry = np.clip(ly, 0, h), np.clip(ry, -1, h-1)
    
#     expand the boundary to integer
    lxi, lyi = int(lx), int(ly)
    rxi, ryi = int(np.ceil(rx)), int(np.ceil(ry))
    
#     Rotate the marker theta degree
    xx_r, yy_r = xx[lxi:rxi+1, lyi:ryi+1], yy[lxi:rxi+1, lyi:ryi+1]
    xx_r, yy_r = np.cos(theta) * (xx_r - x) - np.sin(theta) * (yy_r - y) + x, np.sin(theta) * (xx_r - x)  + np.cos(theta) * (yy_r - y) + y
    
#     calculate the percentage of occupied area by the marker for each pixel

    def intensity(x, lx, rx):
        return 1-np.clip(np.maximum(lx-x, x-rx), 0, 1)
    
    darkness = 0.3+0.7*np.random.random()
    scale = (1 - darkness*intensity(xx_r, lx_raw, rx_raw) * intensity(yy_r, ly_raw, ry_raw))
    for c in range(3):
        img[lxi:rxi+1, lyi:ryi+1, c] *= scale
    
    

def generate(xx, yy, img_blur=None, rng=0., W=48, H=48, N=6, M=6, degree=None):
#     img = np.ones((W, H, 3))
#     img = frame0_blur.copy()

    scale_up = 1
    W_large = W * scale_up
    H_large = H * scale_up
    if img_blur is None:
#         img_blur = (np.random.random((15, 15, 3)) * 0.7) + 0.3
        img_blur = (np.random.random((W//3, H//3, 3)) * 0.9) + 0.1
        img_blur = cv2.resize(img_blur, (H, W))
    
    
#     w, h = img.shape[0], img.shape[1]
    yy_whole, xx_whole = np.meshgrid(np.arange(H), np.arange(W))
    
    img = img_blur + np.random.randn(W, H, 3)*0.05 - 0.025
    missing = np.random.random() * 3
    
    for i in range(N):
        for j in range(M):
            if np.random.random() < missing / N / M:
                continue
                
            r = yy[i, j]
            c = xx[i, j]
#             r = int(np.round(yy[i, j]))
#             c = int(np.round(xx[i, j]))
#             if r >= W or r < 0 or c >= H or c < 0:
#                 continue
            
            mkr_sz = 3 // 2
            
            if degree is None:
                theta = np.random.normal(0, 0.5) * 45 / 180 * np.pi
            else:
                theta = degree
                
            draw_square(img, r, c, 0.5+rng*1, xx_whole, yy_whole, theta)
#             img[r-mkr_sz:r+mkr_sz+1, c-mkr_sz:c+mkr_sz+1, :] = img_blur[r-mkr_sz:r+mkr_sz+1, c-mkr_sz:c+mkr_sz+1, :] * rng
#             img[r, c, :] = img_blur[r, c, :] * (random.random() * 0.)
            
#             for channel in range(3):
#                 img[r-2:r+3, c-2:c+3, channel] = img_blur[r-2:r+3, c-2:c+3, channel] * random.random()*0.2
            
#             for pi in range(-1,2):
#                 for pj in range(-1,2):
#                     if r + pi<0 or r + pi >= W or c + pj <0 or c + pj >= H:
#                         continue
#                     if random.random()<0.7:
#                         img[r+pi,c+pj,:] = random.random()*0.2
#             img[r-1:r+1, c-1:c+1, :] = 0.

    img[:,:1] *= np.random.random(img[:,:1].shape) * 0.5
    img = cv2.GaussianBlur(img, (3,3), 0)
    img[img<0]=0.
    img[img>1]=1.
    return img


def generate_large(xx, yy, img_blur=None, rng=0., W=48, H=48, N=6, M=6):
#     img = np.ones((W, H, 3))
#     img = frame0_blur.copy()

    scale_up = 1
    W_large = W * scale_up
    H_large = H * scale_up
    if img_blur is None:
#         img_blur = (np.random.random((15, 15, 3)) * 0.7) + 0.3
        img_blur = (np.random.random((W//3, H//3, 3)) * 0.9) + 0.1
#         img_blur = (np.random.random((W_large, H_large, 3)) * 0.9) + 0.1
        img_blur = cv2.resize(img_blur, (H_large, W_large))
    
    
#     st = time.time()
    img = img_blur + np.random.randn(W_large, H_large, 3)*0.05 - 0.025
    for i in range(N):
        for j in range(M):
            r = int(yy[i, j]*scale_up)
            c = int(xx[i, j]*scale_up)
            if r >= W_large or r < 0 or c >= H_large or c < 0:
                continue
            
            mkr_sz = int(scale_up * (1 + rng * 3) / 2)
            img[r-mkr_sz:r+mkr_sz+1, c-mkr_sz:c+mkr_sz+1, :] = 0.2+img_blur[r-mkr_sz:r+mkr_sz+1, c-mkr_sz:c+mkr_sz+1, :] * np.random.random() * 0.5
            img[r, c, :] = img_blur[r, c, :] * (random.random() * 0.)
            
#             for channel in range(3):
#                 img[r-2:r+3, c-2:c+3, channel] = img_blur[r-2:r+3, c-2:c+3, channel] * random.random()*0.2
            
#             for pi in range(-1,2):
#                 for pj in range(-1,2):
#                     if r + pi<0 or r + pi >= W or c + pj <0 or c + pj >= H:
#                         continue
#                     if random.random()<0.7:
#                         img[r+pi,c+pj,:] = random.random()*0.2
#             img[r-1:r+1, c-1:c+1, :] = 0.

#     img = cv2.GaussianBlur(img, (3,3), 0)
#     img = cv2.resize(img, (H, W))
    img[:,:1] *= np.random.random(img[:,:1].shape) * 0.5
    img[img<0]=0.
    img[img>1]=1.
#     print(time.time()-st)
    return img


x = np.arange(0, N, 1) * interval + padding
y = np.arange(0, M, 1) * interval + padding
xx, yy = np.meshgrid(x, y)

frame0 = generate(xx, yy)


x = np.arange(0, W, 1)
y = np.arange(0, H, 1)
xx, yy = np.meshgrid(x, y)


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy):
    g = np.exp(-( ((xx-center_x)**2 + (yy-center_y)**2)) / ( 2.0 * sigma**2 ) ) 
     
#     rng = 0.05+np.random.random()*0.9
#     g[g>=g.max()*rng] = g[g>=g.max()*rng].mean()
    
    xx_ = xx + shear_x * g
    yy_ = yy + shear_y * g
    return xx_, yy_
    
    
def twist(center_x, center_y, sigma, theta, xx, yy):
    g = np.exp(-( ((xx-center_x)**2 + (yy-center_y)**2)) / ( 2.0 * sigma**2 ) ) 
    
#     rng = 0.05+np.random.random()*0.9
#     g[g>=g.max()*rng] = g[g>=g.max()*rng].mean()
    
    dx = (xx - center_x)
    dy = (yy - center_y)
    
    rotx = dx * np.cos(theta) - dy * np.sin(theta)
    roty = dx * np.sin(theta) + dy * np.cos(theta)
    
    xx_ = xx + (rotx - dx) * g 
    yy_ = yy + (roty - dy) * g
    return xx_, yy_


def dilate(center_x, center_y, sigma, k, xx, yy):
    g = np.exp(-( ((xx-center_x)**2 + (yy-center_y)**2)) / ( 2.0 * sigma**2 ) ) 
    
#     rng = 0.05+np.random.random()*0.9
#     g[g>=g.max()*rng] = g[g>=g.max()*rng].mean()
    
    dx = (xx - center_x)
    dy = (yy - center_y)
    
    
    xx_ = xx + (k * dx) * g 
    yy_ = yy + (k * dy) * g
    return xx_, yy_

x = np.arange(0, W, 1)
y = np.arange(0, H, 1)
xx, yy = np.meshgrid(x, y)



xx0, yy0 = xx.copy(), yy.copy() 
X = []
Y = []

def random_shear(xx, yy, W, H):
    shear_ratio = 5
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    
    if np.random.random() < 0.3:
        normal = np.array([center_x - W/2, center_y - H/2])
        normal = normal / ((np.sum(normal**2))**0.5 + 1e-6)
        
        shear_x = random.random() * interval * shear_ratio * normal[0]
        shear_y = random.random() * interval * shear_ratio * normal[1]
    else:
        shear_x = random.random() * interval * shear_ratio - interval * shear_ratio / 2
        shear_y = random.random() * interval * shear_ratio - interval * shear_ratio / 2
        
#     print(center_x, center_y, shear_x, shear_y)
        
    
    xx_, yy_ = shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy)
    return xx_, yy_
    
    
def random_twist(xx, yy, W, H):
    twist_degree = 100
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    theta = (random.random()*twist_degree - twist_degree/2.) / 180.0 * np.pi
    
    xx_, yy_ = twist(center_x, center_y, sigma, theta, xx, yy)
    return xx_, yy_

    
def random_dilate(xx, yy, W, H):
    k_rng = 0.2
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    k = random.random() * k_rng
#     k = k_rng
    
    xx_, yy_ = dilate(center_x, center_y, sigma, k, xx, yy)
    return xx_, yy_



def preprocessing(img, W, H):
#     Brightness
    ret = img
    
    
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx, yy = np.meshgrid(y, x)
    
    for _ in range(5):
        sz_x = int(2 + random.random() * 15)
        sz_y = int(2 + random.random() * 15)
        x = int(random.random() * (W - sz_x))
        y = int(random.random() * (H  - sz_y))
        theta = np.random.random() * np.pi
        rng = 0.7
        xr = (xx - x) * np.cos(theta) - (yy - y) * np.sin(theta)
        yr = (xx - x) * np.sin(theta) + (yy - y) * np.cos(theta)
        
        mask = np.logical_and.reduce([(xr>=-sz_x), (xr<=sz_x), (yr >= -sz_y), (yr <= sz_y)])
        
        ret[mask] *= (1 + np.random.random(3) * rng * 2 - rng)
        
#         ret[x:x+sz_x//2, y:y+sz_y//2, :] *= (1 + np.random.random(3) * rng * 2 - rng)
#         ret[x+sz_x//2:x+sz_x, y:y+sz_y//2, :] *= (1 + np.random.random(3) * rng * 2 - rng)
        
#         ret[x:x+sz_x//2, y+sz_y//2:y+sz_y, :] *= (1 + np.random.random(3) * rng * 2 - rng)
#         ret[x+sz_x//2:x+sz_x, y+sz_y//2:y+sz_y, :] *= (1 + np.random.random(3) * rng * 2 - rng)
    
    
    
    return ret

def generate_img(batch_size=32, setting=None):
    
    while True:

        X, Y = [], []
        
#         if np.random.random() < 0:
#             W, H = 48, 48
#             N, M = 6, 6
#         else:
#             W, H = 48, 64
#             N, M = 6, 8
            
#         N, M = np.random.randint(4,15), np.random.randint(4,15)
#         W = np.random.randint(N * 6, 96)
#         H = np.random.randint(M * 6, 96)
#         W = (W//16+1)*16
#         H = (H//16+1)*16

        N, M = 10, 14
        W, H = 80, 112
        
#         W, H = 48, 48
#         N, M = 6, 6
        
        if not(setting is None):
            W, H, N, M = setting
        
        
        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)
        xx0, yy0 = np.meshgrid(y, x)
        
        
        interval_x = W/(N)
        interval_y = H/(M)

        x = np.arange(interval_x/2, W, interval_x)[:N]
        y = np.arange(interval_y/2, H, interval_y)[:M]
        xind, yind = np.meshgrid(y, x)
#         print(xind)
#         print(yind)
        
        xind = (xind.reshape([1,-1])[0]).astype(np.int)
        yind = (yind.reshape([1,-1])[0]).astype(np.int)
        xind += (np.random.random(xind.shape)*2-1).astype(np.int)
        yind += (np.random.random(xind.shape)*2-1).astype(np.int)
        
        for i in range(batch_size):
            xx = xx0 + (np.random.random(xx0.shape)*2-1)
            yy = yy0 + (np.random.random(yy0.shape)*2-1)
            rng = np.random.random()
#             rng = 0

            img_blur = (np.random.random((15, 15, 3)) * 0.7) + 0.3
            img_blur = cv2.resize(img_blur, (W, H))



        #     Random markers
#             xind = (np.random.random(N*M) * W).astype(np.int)
#             yind = (np.random.random(N*M) * H).astype(np.int)

        #     Grid markers
#             x_grid = np.arange(0, N, 1) * interval + padding
#             y_grid = np.arange(0, M, 1) * interval + padding
#             xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

#             xind, yind = np.reshape(xx_grid,[-1]), np.reshape(yy_grid,[-1])
#             xind += (np.random.random(xind.shape)*4-2).astype(np.int)
#             yind += (np.random.random(xind.shape)*4-2).astype(np.int)


            xx_marker, yy_marker = xx[yind,xind].reshape([N,M]), yy[yind, xind].reshape([N,M])
            img0 = generate(xx_marker, yy_marker, img_blur=None, rng=rng, W=W, H=H, N=N, M=M, degree=0)

        #     Random distortion
            xx_, yy_ = xx, yy
#             xx_, yy_ = random_dilate(xx_, yy_, W, H)
            xx_, yy_ = random_shear(xx_, yy_, W, H)
            xx_, yy_ = random_twist(xx_, yy_, W, H)

#             xx_, yy_ = random_shear(xx_, yy_, W, H)
#             xx_, yy_ = random_twist(xx_, yy_, W, H)
            
            xx_ += np.random.random(xx_.shape) * 1 - 0.5
            yy_ += np.random.random(yy_.shape) * 1 - 0.5


        #     Distorted markers
            xx_marker_, yy_marker_ = xx_[yind,xind].reshape([N,M]), yy_[yind, xind].reshape([N,M])
            img = generate(xx_marker_, yy_marker_, img_blur=None, rng=rng, W=W, H=H, N=N, M=M)
#             img = generate_large(xx_marker_, yy_marker_, img_blur=None, rng=rng, W=W, H=H, N=N, M=M)
            img = preprocessing(img, W, H)

            t = np.zeros([xx_marker_.shape[0], xx_marker_.shape[1], 2])
            t[:,:,0] = xx_marker_# - xx_marker
            t[:,:,1] = yy_marker_# - yy_marker

#             X.append(np.dstack([img0-0.5, img-0.5]))#, np.reshape(xx,[W,H,1]), np.reshape(yy,[W,H,1])]))
            X.append(np.dstack([img0-0.5, img-0.5]))#, np.reshape(xx,[W,H,1]), np.reshape(yy,[W,H,1])]))
#             X.append(np.dstack([img-0.5]))#, np.reshape(xx,[W,H,1]), np.reshape(yy,[W,H,1])]))
            Y.append(t)

        X = np.array(X)
        Y = np.array(Y)

#         X = X[:,:W,:H] - 0.5
#         print(X.shape, xx.shape, yy.shape)
#         X = np.dstack([X])
        Y = Y[:,:W,:H]
        
        Y_list = Y


        yield X, Y_list

    

# for i in range(500000):
# for i in range(100000):
#     if i % 10000 == 0: print(i)
#     xx_, yy_ = xx, yy
#     xx_, yy_ = random_shear(xx_, yy_)
#     xx_, yy_ = random_twist(xx_, yy_)
    
# #     xx_, yy_ = twist(21, 21, 3, -40/180.*np.pi, xx_, yy_)
#     img = generate(xx_, yy_)
#     X.append(img)
    
#     t = np.zeros([xx_.shape[0], xx_.shape[1], 2])
#     t[:,:,0] = xx_.astype(np.int)
#     t[:,:,1] = yy_.astype(np.int)
#     Y.append(t)
    

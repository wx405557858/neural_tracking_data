import argparse
import cv2
import numpy as np
import time
import math

from keras.models import load_model

N = 6
M = 8

# N = 6
# M = 6

padding = 7
interval = 7
W = 64
H = 48 * 2


traj = []
moving = False
rotation = False

K = 10
mkr_rng = 0.0


x = np.arange(0, W, 1)
y = np.arange(0, H, 1)
xx, yy = np.meshgrid(y, x)

# frame0_blur = cv2.imread('frame0_blur.jpg') / 255.
# frame0_blur = cv2.imread('frame0_blur.jpg') / 255.
# frame0_blur = cv2.resize(frame0_blur, (W, H))


img_blur = (np.random.random((3, 3, 3)) * 0.7) + 0.3
frame0_blur = cv2.resize(img_blur, (H, W))

# frame0_blur = np.ones((49, 49, 3), dtype=np.float32)

crazy = True


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    # rng = 0.05+np.random.random()*0.95
    # g[g>g.max()*rng] = g[g>g.max()*rng].mean()

    dx = shear_x * g
    dy = shear_y * g
    if crazy == False:
        thres = 0.7 * interval
        mag = (dx ** 2 + dy ** 2) ** 0.5
        mask = mag > thres
        dx[mask] = dx[mask] / mag[mask] * thres
        dy[mask] = dy[mask] / mag[mask] * thres

    xx_ = xx + dx
    yy_ = yy + dy
    return xx_, yy_


def twist(center_x, center_y, sigma, theta, xx, yy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    # rng = 0.05+np.random.random()*0.95
    # g[g>g.max()*rng] = g[g>g.max()*rng].mean()
    dx = xx - center_x
    dy = yy - center_y

    rotx = dx * np.cos(theta) - dy * np.sin(theta)
    roty = dx * np.sin(theta) + dy * np.cos(theta)

    xx_ = xx + (rotx - dx) * g
    yy_ = yy + (roty - dy) * g
    return xx_, yy_


xx0 = xx.copy()
yy0 = yy.copy()

wx = xx0.copy()
wy = yy0.copy()

changing_x, changing_y = 0, 0

# def generate(xx, yy):
#     # img = np.ones((W, H, 3))
#     img = frame0_blur.copy()
#     for i in range(N):
#         for j in range(M):
#             r = int(xx[i, j])
#             c = int(yy[i, j])
#             if r >= W or r < 0 or c >= H or c < 0:
#                 continue
#             img[r-2:r+2, c-2:c+2, :] = 0.
#     # img[20:30,20:30,:] = 1
#     # img[5:10,5:10,:] = 1
#     return img


def generate(xx, yy):
    # img = np.ones((W, H, 3))
    img = frame0_blur.copy()

    # img = img + np.random.randn(W, H, 3)*0.05 - 0.025

    for i in range(N):
        for j in range(M):
            r = int(yy[i, j])
            c = int(xx[i, j])
            if r >= W or r < 0 or c >= H or c < 0:
                continue
            shape = img[r - 1 : r + 2, c - 1 : c + 2, :].shape

            img[r - 1 : r + 2, c - 1 : c + 2, :] = (
                frame0_blur[r - 1 : r + 2, c - 1 : c + 2, :] * mkr_rng
            )
            img[r, c, :] = frame0_blur[r, c, :] * -2
            # img[r-1:r+2, c-1:c+2, :] = np.ones(shape)

            # for channel in range(3):
            #     img[r-1:r+2, c-1:c+2, channel] = img[r-1:r+2, c-1:c+2, channel] * random.random()*0.2

    #             for pi in range(-1,2):
    #                 for pj in range(-1,2):
    #                     if r + pi<0 or r + pi >= W or c + pj <0 or c + pj >= H:
    #                         continue
    #                     if random.random()<0.7:
    #                         img[r+pi,c+pj,:] = random.random()*0.2
    #             img[r-1:r+1, c-1:c+1, :] = 0.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    print("IMG SHAPE", img.shape)
    # img = cv2.resize(img, (H, W))
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    img = img[:W, :H]
    print("IMG SHAPE", img.shape)
    return img


# def draw_flow(img, flow):
#     global K
#     img_ = cv2.resize(img, (W * K, H * K))
#     for i in range(N):
#         for j in range(M):
#             pt1 = (int(yy0[i, j] * K + K // 2), int(xx0[i, j] * K + K // 2))
#             pt2 = (int(flow[i, j, 1] * K + K // 2), int(flow[i, j, 0] * K + K // 2))
#             cv2.arrowedLine(img_, pt1, pt2, (0, 255, 255), 2, 8, 0, 0.4)
#     return img_


def draw_dense_flow(img, flow, scale=1.0, K=5):
    img_ = cv2.resize(img, (img.shape[1] * K, img.shape[0] * K))
    row, col = img.shape[0], img.shape[1]
    print("FLOW SHAPE", flow.shape, "IMG SHAPE", img.shape)
    for i in range(6, row - 2, 7):
        for j in range(6, col - 2, 7):
            d = (flow[i, j] * scale * K).astype(int)
            cv2.arrowedLine(
                img_,
                (int(j * K + K // 2), int(i * K + K // 2)),
                (int(j * K + d[0] + K // 2), int(i * K + d[1] + K // 2)),
                (0, 255, 255),
                2,
                8,
                0,
                0.4,
            )
    return img_


def contrain(xx, yy):
    dx = xx - xx0
    dy = yy - yy0
    if crazy == False:
        thres = 0.35 * interval
        mag = (dx ** 2 + dy ** 2) ** 0.5
        mask = mag > thres
        dx[mask] = dx[mask] / mag[mask] * thres
        dy[mask] = dy[mask] / mag[mask] * thres
    xx = xx0 + dx
    yy = yy0 + dy
    return xx, yy


def cross_product(Ax, Ay, Bx, By, Cx, Cy):
    len1 = ((Bx - Ax) ** 2 + (By - Ay) ** 2) ** 0.5
    len2 = ((Cx - Ax) ** 2 + (Cy - Ay) ** 2) ** 0.5
    return ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) / (len1 * len2 + 1e-6)


def motion_callback(event, x, y, flags, param):
    global traj, moving, xx, yy, wx, wy, rotation, changing_x, changing_y

    x, y = x / K, y / K

    if event == cv2.EVENT_LBUTTONDOWN:
        traj.append([x, y])

        wx = xx0.copy()
        wy = yy0.copy()

        rotation = False
        moving = True

    elif event == cv2.EVENT_LBUTTONUP:
        traj = []
        moving = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if moving == True:
            traj.append([x, y])
            sigma = 10
            # sigma = 20
            if rotation == False:
                xx, yy = shear(
                    traj[0][0],
                    traj[0][1],
                    sigma,
                    x - traj[0][0],
                    y - traj[0][1],
                    wx,
                    wy,
                )
            else:
                sigma = 20
                theta = math.asin(
                    cross_product(traj[0][0], traj[0][1], changing_x, changing_y, x, y)
                )
                theta = max(min(theta, 50 / 180.0 * math.pi), -50 / 180.0 * math.pi)
                xx, yy = twist(traj[0][0], traj[0][1], sigma, theta, wx, wy)
            if crazy == False:
                xx, yy = contrain(xx, yy)


cv2.namedWindow("image")
cv2.setMouseCallback("image", motion_callback)

flag_first = False

# model = load_model('models/6x6_1/tracking_195_0.039.h5')

# model = load_model('models/6x6_gelsight/tracking_061_0.064.h5')
# model = load_model('models/random_ae/tracking_029_0.631.h5')
# model = load_model('models/random_ae_2/tracking_008_0.550.h5')
# model = load_model('models/random_ae_size/tracking_028_0.402.h5')
# model = load_model('models/random_ae_grid/tracking_018_1.554.h5')
# model = load_model('models/random_ae_grid_abs_k5/tracking_029_1.704.h5')
# model = load_model('models/random_ae_multi_3/tracking_015_0.771.h5')
# model = load_model('models/random_ae_multi_add/tracking_016_0.751.h5')

# model = load_model("models/random_ae_random_multi_scratch_4/tracking_049_0.439.h5")
# model = load_model("models/random_ae_var_2/tracking_097_0.098.h5")

# model = load_model('models/random_ae_var_multi_8_rela_single/tracking_004_0.473.h5')
# model = load_model("models/random_ae_var_multi_8/tracking_042_0.237.h5")

# model = load_model("models/random_ae_var_multi_8_rela/tracking_029_0.179.h5")

model = load_model("models/random_ae_var_grid/tracking_041_0.508.h5")  # Star

# model = load_model("models/random_ae_var_grid/tracking_050_0.134.h5")

# model = load_model('models/random_ae_random_single/tracking_011_0.610.h5')


svid = 0

xind = (np.random.random(N * M) * W).astype(np.int)
yind = (np.random.random(N * M) * H).astype(np.int)


# T = 4
interval_x = W / (N + 1)
interval_y = H / (M + 1)

print(interval_x, interval_y)
x = np.arange(interval_x, W, interval_x)[:N]
y = np.arange(interval_y, H, interval_y)[:M]
print(x, y)
# exit()
xind, yind = np.meshgrid(x, y)
xind = (xind.reshape([1, -1])[0]).astype(np.int)
yind = (yind.reshape([1, -1])[0]).astype(np.int)

# print("xind",xind)


xx_marker, yy_marker = xx[xind, yind].reshape([N, M]), yy[xind, yind].reshape([N, M])
img0 = generate(xx_marker, yy_marker)


while True:
    xx_marker_, yy_marker_ = xx[xind, yind].reshape([N, M]), yy[xind, yind].reshape(
        [N, M]
    )

    img = generate(xx_marker_, yy_marker_)
    # img += np.random.random(img.shape) * 0.1
    st = time.time()
    pred = model.predict(
        np.array(
            [
                np.dstack(
                    [
                        img0 - 0.5,
                        img - 0.5,
                        np.reshape(xx, [W, H, 1]),
                        np.reshape(yy, [W, H, 1]),
                    ]
                )
            ]
        )
    )[0][0]
    print(time.time() - st)

    # pred += 0.5

    if flag_first == False:
        flag_first = True
        # xx0 = pred[0][:,:,0]
        # yy0 = pred[0][:,:,1]
        img0 = img.copy()

        xx = xx0.copy()
        yy = yy0.copy()

        pred0 = pred.copy()
        continue

    pred = pred - pred0 + np.dstack([xx0, yy0])
    # display_img = cv2.resize(img, (H*K, W*K))
    # display_img = draw_flow(img, pred[0])

    # relative
    # flow = np.swapaxes(pred[0, :, :, ::-1], 0, 1)

    # absolute
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx0, yy0 = np.meshgrid(y, x)
    # flow = np.swapaxes(pred[:,:,::-1]-np.dstack([yy0, xx0]),0,1)
    flow = pred[:, :] - np.dstack([xx0, yy0])
    # flow = np.rot90(np.swapaxes(flow, 0, 1), k=2, axes=(0, 1))

    # display_img = optical_flow_opencv.draw_nn(img, pred[0,:,:,::-1], scale=1, K=10)
    display_img = draw_dense_flow(img, flow, scale=1, K=10)

    # flow_of = optical_flow_opencv.of(img0, img)
    # display_img_of = optical_flow_opencv.draw(img, flow_of, scale=1, K=10)

    # gt = draw_flow(img, np.dstack([xx, yy]))
    flow_gt = np.dstack([xx - xx0, yy - yy0])
    # flow_gt = np.swapaxes(flow_gt, 0, 1)
    gt = draw_dense_flow(img, flow_gt, scale=1, K=10)

    # cv2.imshow("image_of", display_img_of)
    cv2.imshow("gt", gt)
    cv2.imshow("image", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        xx = xx0.copy()
        yy = yy0.copy()

    if key == ord("s"):
        rotation = rotation ^ True
        if len(traj) > 0:
            changing_x = traj[-1][0]
            changing_y = traj[-1][1]
        if rotation == False:
            traj = []
        wx, wy = xx.copy(), yy.copy()

    elif key == ord("q"):
        break
    elif key == ord("c"):
        img_blur = (np.random.random((3, 3, 3)) * 0.7) + 0.3
        frame0_blur = cv2.resize(img_blur, (H, W))

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)

    elif key == ord("p"):
        cv2.imwrite("im{}.jpg".format(svid), display_img * 255)
        svid += 1

    elif key == ord("k"):
        crazy = crazy ^ True

    elif key == ord("d"):
        N = np.random.randint(6, 10)
        M = np.random.randint(6, 14)

        interval_x = W / (N + 1)
        interval_y = H / (M + 1)

        x = np.arange(interval_x, W, interval_x)[:N]
        y = np.arange(interval_y, H, interval_y)[:M]

        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1, -1])[0]).astype(np.int)
        yind = (yind.reshape([1, -1])[0]).astype(np.int)

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)

    elif key == ord("z"):
        mkr_rng = mkr_rng - 0.5
        if mkr_rng < 0:
            mkr_rng = 1

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)
        # flag_first = False

# close all open windows
cv2.destroyAllWindows()

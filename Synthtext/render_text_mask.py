"""
Rendering text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""

import math
import os
import random
import time

import cv2
import numpy as np
import pygame
import pygame.locals
from pygame import freetype


def center2size(surf, size):

    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    canvas[(size_h-surf_h)//2:(size_h-surf_h)//2+surf_h,
           (size_w-surf_w)//2:(size_w-surf_w)//2+surf_w] = surf
    return canvas


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]),
          min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        return arr, bbs
    else:
        return arr


def render_normal(font, text):

    # get the number of lines
    lines = text.split('\n')
    lengths = [len(l) for l in lines]

    # font parameters:
    line_spacing = font.get_sized_height() + 1

    # initialize the surface to proper size:
    line_bounds = font.get_rect(lines[np.argmax(lengths)])
    fsize = (round(2.0 * line_bounds.width),
             round(1.25 * line_spacing * len(lines)))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    bbs = []
    space = font.get_rect('O')
    x, y = 0, 0
    for l in lines:
        x = 0  # carriage-return
        y += line_spacing  # line-feed

        for ch in l:  # render each character
            if ch.isspace():  # just shift
                x += space.width
            else:
                # render the character
                ch_bounds = font.render_to(surf, (x, y), ch)
                print(ch, x, y, ch_bounds)
                ch_bounds.x = x + ch_bounds.x
                ch_bounds.y = y - ch_bounds.y
                x = ch_bounds.x + ch_bounds.width
                bbs.append(np.array(ch_bounds))

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # get the words:
    words = ' '.join(text.split())

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(
        pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
    surf_arr = surf_arr.swapaxes(0, 1)

    # self.visualize_bb(surf_arr,bbs)
    return surf_arr, bbs


def render_curved(font, text, curve_rate, curve_center=None):

    wl = len(text)
    isword = len(text.split()) == 1

    # do curved iff, the length of the word <= 10
    if not isword or wl > 10:
        return render_normal(font, text)

    # create the surface:
    lspace = font.get_sized_height() + 1
    lbound = font.get_rect(text)
    #fsize = (round(2.0*lbound.width), round(3*lspace))
    fsize = (round(3.0*lbound.width), round(5*lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    # baseline state
    if curve_center is None:
        curve_center = wl // 2
    curve_center = max(curve_center, 0)
    curve_center = min(curve_center, wl - 1)
    mid_idx = curve_center  # wl//2
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(wl)]
    curve[mid_idx] = -np.sum(curve) / max(wl-1, 1)
    rots = [-int(math.degrees(math.atan(2 * curve_rate *
                                        (i-mid_idx)/(font.size/2)))) for i in range(wl)]

    bbs = []
    # place middle char
    rect = font.get_rect(text[mid_idx])
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery + rect.height
    rect.centery += curve[mid_idx]
    ch_bounds = font.render_to(
        surf, rect, text[mid_idx], rotation=rots[mid_idx])
    print(text[mid_idx], rect, ch_bounds)
    ch_bounds.x = rect.x + ch_bounds.x
    ch_bounds.y = rect.y - ch_bounds.y
    mid_ch_bb = np.array(ch_bounds)

    # render chars to the left and right:
    last_rect = rect
    ch_idx = []
    for i in range(wl):
        # skip the middle character
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)
            continue

        if i < mid_idx:  # left-chars
            i = mid_idx-1-i
        elif i == mid_idx + 1:  # right-chars begin
            last_rect = rect

        ch_idx.append(i)
        ch = text[i]

        newrect = font.get_rect(ch)
        newrect.y = last_rect.y
        if i > mid_idx:
            newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
        else:
            newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
        newrect.centery = max(newrect.height, min(
            fsize[1] - newrect.height, newrect.centery + curve[i]))
        try:
            bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
        except ValueError:
            bbrect = font.render_to(surf, newrect, ch)
        bbrect.x = newrect.x + bbrect.x
        bbrect.y = newrect.y - bbrect.y
        bbs.append(np.array(bbrect))
        last_rect = newrect

    # correct the bounding-box order:
    bbs_sequence_order = [None for i in ch_idx]
    for idx, i in enumerate(ch_idx):
        bbs_sequence_order[i] = bbs[idx]
    bbs = bbs_sequence_order

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(
        pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, bbs


def center_warpPerspective(img, H, center, size, bb):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size,
                              cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    points = bb2points(bb)
    perspected_points = M.dot(points)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    bb = points2bb(perspected_points)
    return img, bb


def bb2points(bb):
    points = np.ones((3, 4 * bb.shape[2]), dtype=np.float32)
    for i in range(bb.shape[2]):
        x = bb[0, :, i]
        y = bb[1, :, i]
        points[:2, 0+4*i] = np.array([x[0], y[0]], dtype=np.float32).T
        points[:2, 1+4*i] = np.array([x[1], y[1]], dtype=np.float32).T
        points[:2, 2+4*i] = np.array([x[2], y[2]], dtype=np.float32).T
        points[:2, 3+4*i] = np.array([x[3], y[3]], dtype=np.float32).T
    bb = points2bb(points)
    return points


def points2bb(points):
    n = int(points.shape[1] / 4)
    bb = np.zeros((2, 4, n))
    for i in range(0, points.shape[1], 4):
        bb[0, :, i//4] = np.int0(points[0, i:i+4])
        bb[1, :, i//4] = np.int0(points[1, i:i+4])
    return bb


def update_bb(bb, x_offset, y_offset):
    n = bb.shape[2]
    for i in range(n):
        bb[0, :, i] += x_offset
        bb[1, :, i] += y_offset
    return bb


def center_pointsPerspective(points, H, center):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    return M.dot(points)


def perspective(img, bb, rotate_angle, zoom, shear_angle, perspect, pad):  # w first

    rotate_angle = rotate_angle * math.pi / 180.
    shear_x_angle = shear_angle[0] * math.pi / 180.
    shear_y_angle = shear_angle[1] * math.pi / 180.
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect

    H_scale = np.array([[scale_w, 0, 0],
                        [0, scale_h, 0],
                        [0, 0, 1]], dtype=np.float32)
    H_rotate = np.array([[math.cos(rotate_angle), math.sin(rotate_angle), 0],
                         [-math.sin(rotate_angle), math.cos(rotate_angle), 0],
                         [0, 0, 1]], dtype=np.float32)
    H_shear = np.array([[1, math.tan(shear_x_angle), 0],
                        [math.tan(shear_y_angle), 1, 0],
                        [0, 0, 1]], dtype=np.float32)
    H_perspect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [perspect_x, perspect_y, 1]], dtype=np.float32)

    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)
    points = np.ones((3, 4), dtype=np.float32)
    points[:2, 0] = np.array([0, 0], dtype=np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype=np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype=np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype=np.float32).T
    perspected_points = center_pointsPerspective(points, H, img_center)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    canvas_w = int(2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]),
                           np.max(perspected_points[0, :]) - img_center[0])) + 10
    canvas_h = int(2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]),
                           np.max(perspected_points[1, :]) - img_center[1])) + 10

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tly = (canvas_h - img_h) // 2
    tlx = (canvas_w - img_w) // 2
    canvas[tly:tly+img_h, tlx:tlx+img_w] = img

    bb = update_bb(bb, tlx, tly)

    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    canvas, bb = center_warpPerspective(
        canvas, H, canvas_center, canvas_size, bb)
    loc = np.where(canvas > 127)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    text_w = maxx - minx + 1
    text_h = maxy - miny + 1
    resimg = np.zeros(
        (text_h + pad[2] + pad[3], text_w + pad[0] + pad[1])).astype(np.uint8)
    resimg[pad[2]:pad[2]+text_h, pad[0]:pad[0] +
           text_w] = canvas[miny:maxy+1, minx:maxx+1]

    bb = update_bb(bb, -minx + pad[0], -miny + pad[2])
    return resimg, bb


def render_text(font, text, param):

    if param['is_curve']:
        return render_curved(font, text, param['curve_rate'], param['curve_center'])
    else:
        return render_normal(font, text)


def bb_xywh2coords(bbs):
    """
    Takes an nx4 bounding-box matrix specified in x,y,w,h
    format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
    """
    n, _ = bbs.shape
    coords = np.zeros((2, 4, n))
    for i in range(n):
        coords[:, :, i] = bbs[i, :2][:, None]
        coords[0, 1, i] += bbs[i, 2]
        coords[:, 2, i] += bbs[i, 2:4]
        coords[1, 3, i] += bbs[i, 3]
    return coords


def paint_rotate_rectangle(img, bbs, color=(0, 0, 255)):
    new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        rect = cv2.minAreaRect(c)

        box = cv2.boxPoints(rect)

        box = np.int0(box)

        cv2.drawContours(new_img, [box], 0, color, 1)

    return new_img


def paint_rectangle(img, bbs, color=(0, 0, 255)):
    new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for bb in bbs:
        cv2.rectangle(new_img, (bb[0], bb[1]),
                      (bb[0] + bb[2], bb[1] + bb[3]), color)
    return new_img


def paint_boundingbox(img, bb, color=(0, 0, 255)):
    new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(bb.shape[2]):
        x = bb[0, :, i]
        y = bb[1, :, i]
        points = [[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]]
        points = np.array(points, dtype=np.int64)
        cv2.drawContours(new_img, [points], 0, color, 1, lineType=cv2.LINE_AA)
    return new_img


def main():
    import data_cfg
    pygame.init()
    freetype.init()
    font_name = '/data1/yfx/datasets/fonts/english_ttf/arial.ttf'
    font = freetype.Font(font_name)
    font.antialiased = True
    font.origin = True

    text = 'transfect'

    # font.size = np.random.randint(
    #     data_cfg.font_size[0], data_cfg.font_size[1] + 1)
    font.size = 60
    param = {
        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn()
        + data_cfg.curve_rate_param[1],
        'curve_center': np.random.randint(0, len(text))
    }
    # surf, bbs = render_normal(font, text)
    surf, bbs = render_curved(font, text, 1, 4)
    print(bbs)
    print(bbs[:, 3])
    print(surf.shape)

    bb = bb_xywh2coords(bbs)
    print(bb)

    img = paint_boundingbox(surf, bb)
    cv2.imshow('img', img)
    cv2.waitKey()

    # get padding
    padding_ud = np.random.randint(
        data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
    padding_lr = np.random.randint(
        data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
    padding = np.hstack((padding_ud, padding_lr))

    rotate = data_cfg.rotate_param[0] * \
        np.random.randn() + data_cfg.rotate_param[1]
    zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
    shear = data_cfg.shear_param[0] * \
        np.random.randn(2) + data_cfg.shear_param[1]
    perspect = data_cfg.perspect_param[0] * \
        np.random.randn(2) + data_cfg.perspect_param[1]
    # surf, bb = perspective(surf, bb, 20, [1.11, 0.9], [-0.49,-3.8], [0.0006, -0.0002], [2,8,11,8])
    surf, bb = perspective(surf, bb, 20, zoom, shear, perspect, padding)
    print(bb)

    img = paint_boundingbox(surf, bb)
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    main()

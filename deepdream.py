import numpy as np
import argparse
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import caffe
import random
import sys

"""
fn: file name
"""
model_path = "bvlc_googlenet/"
net_fn = model_path + "deploy.prototxt"
param_fn = model_path + "bvlc_googlenet.caffemodel"
img = ""
guide = ""
guide_features = None

def set_arg():
    global img
    global guide
    parser = argparse.ArgumentParser(
        description="Deep Style"
    )
    parser.add_argument(
        "-s",
        "--src",
        help="source img",
        dest="img"
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="desination img",
        dest="guide"
    )
    args = parser.parse_args()
    
    if args.img is None:
        print("Error. Please see -h for more helps.")
        sys.exit(1)
    else:
        img = args.img

    if args.guide is not None:
        guide = args.guide 

set_arg()

# Use protobuf obj to load caffe model 
model = caffe.io.caffe_pb2.NetParameter()

"""
text_format.Merge(text,message)
> Merges an ASCII representation of a protocol message into a message.

Args:
  text: Message ASCII representation.
  message: A protocol buffer message to merge into.
"""
text_format.Merge(open(net_fn).read(), model)

# Enable patching model to compute gradient
model.force_backward = True
open("tmp.prototxt", "w").write(str(model))

# Use network struct(net_fn) and trained model(param_fn) to build classifier
"""
mean: ImageNet mean, training set dependent
channel_swap: BGR -> RGB
"""
net = caffe.Classifier(
    "tmp.prototxt",
    param_fn,
    mean=np.float32([104.0,116.0,122.0]),
    channel_swap=(2,1,0)
)

def save_img(a, fmt="jpeg"):
    # float32 to uint8
    a = np.uint8(np.clip(a, 0, 255))
    name = str(random.randint(0,100000))
    PIL.Image.fromarray(a).save(name+"."+fmt, fmt)
    print(name+"."+fmt)

# A couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    """
    np.rollaxis: move array axes to new positions
    
    base_img.shape: 720, 1080, 3
    np.rollaxis(img, 2): 3, 720, 1080
    """
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean["data"]

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean["data"])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    print("ch is " + str(ch))
    x = x.reshape(ch, -1)
    y = y.reshape(ch, -1)
    # Compute the matrix of dot-products with guide features
    A = x.T.dot(y)
    # Select ones that match best
    dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]

# Gradient ascent
def make_step(net, step_size=1.5, end="inception_4c/output",
            jitter=32, clip=True, objective=objective_L2):
    # Input data
    src = net.blobs["data"]
    # Target layer(default: inception_4c/output)
    dst = net.blobs[end]

    # Generate jitter
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    # Apply jitter shift
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
    
    # Forward to target layer
    net.forward(end=end)
    # Specify the optimization objective
    objective(dst)

    # Backward to target layer (optimize the target layer)
    net.backward(start=end)

    # Apply normalized ascent step to input image
    g = src.diff[0]
    src.data[:] += step_size/np.abs(g).mean() * g

    # Unshift jitter
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)

    if clip:
        bias = net.transformer.mean["data"]
        src.data[:] = np.clip(src.data, -bias, 255-bias)
"""
iteration: 10
octave: 4
"""
def deepdream(net, base_img, objective=objective_L2, iter_n=10, octave_n=4, octave_scale=1.4,
            end="inception_4c/output", clip=True, **step_params):
    # Prepare base images for all octave
    octaves = [preprocess(net, base_img)]
    
    # Zoom in image for obvate
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale, 1.0/octave_scale), order=1))

    src = net.blobs["data"]
    
    # Return an array that its shape is same as octaves[-1]
    detail = np.zeros_like(octaves[-1])
    
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        
        # Upscale detail from the previous octave
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)

        # Resize src as image size
        src.reshape(1, 3, h, w)
        src.data[0] = octave_base + detail
        
        # 10 times gradient ascent
        for i in range(iter_n):
            make_step(net, end=end, clip=clip, objective=objective, **step_params)

            # Visualize image
            vis = deprocess(net, src.data[0])

            # When clipping is disabled, adjust image contrast
            if not clip:
                vis = vis*(255.0/np.percentile(vis, 99.98))

            save_img(vis)
            print(octave)
            print(i)
            print(end)
            print(vis.shape)
        
        # Extract detail
        detail = src.data[0] - octave_base
    
    # Return the result image
    return deprocess(net, src.data[0])

    

if __name__=="__main__":
    img = np.float32(PIL.Image.open(img))
    
    if guide != "":
        guide = np.float32(PIL.Image.open(guide))
        
        end = "inception_3b/output"
        h, w = guide.shape[:2]
        src, dst = net.blobs["data"], net.blobs[end]
        src.reshape(1, 3, h, w)
        src.data[0] = preprocess(net, guide)
        net.forward(end=end)
        guide_features = dst.data[0].copy()

        deepdream(net, img, end=end, objective=objective_guide)
    else:
        deepdream(net, img)

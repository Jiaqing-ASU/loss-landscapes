from numpy import load
import numpy as np
import argparse

STEPS = 40

parser = argparse.ArgumentParser()
# input corruption dataset:
# original, brightness, canny_edges, dotted_line, fog, glass_blur, identity
# impulse_noise, motion_blur, rotate, scale, shear, shot_noise
# spatter, stripe, translate, zigzag
parser.add_argument('--dataset', default='original', help='dataset')             
args = parser.parse_args()

# load loss values
if args.dataset == 'original':
    loss_data_fin_3d = load('mnist_results/loss_data_fin_3d_mnist.npy')
    loss_data_fin = load('mnist_results/loss_data_fin_mnist.npy')
    # export the loss values for 3d
    np.array(loss_data_fin_3d,np.float32).tofile("points/original/points_loss_3d.raw")
    # export the loss values for 2d
    np.array(loss_data_fin,np.float32).tofile("points/original/points_loss_2d.raw")
else:
    path = 'mnistc_results' + '/' + args.dataset + '/'
    loss_data_fin_3d = load(path+'loss_data_fin_3d_mnistc.npy')
    loss_data_fin = load(path+'loss_data_fin_mnistc.npy')
    output_path = 'points' + '/' + args.dataset + '/'
    # export the loss values for 3d
    np.array(loss_data_fin_3d,np.float32).tofile(output_path+"points_loss_3d.raw")
    # export the loss values for 2d
    np.array(loss_data_fin,np.float32).tofile(output_path+"points_loss_2d.raw")

# generate testing numpy array for ParaView
test_array = np.arange(100).reshape(5,20).astype(np.float)

np.savetxt("points/test/points_test.csv", test_array, delimiter=",")
np.array(test_array,np.float32).tofile("points/test/points_test.raw")
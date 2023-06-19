import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from model.sphereglue import SphereGlue
from utils.demo_mydataset import MyDataset 
import numpy
from utils.Utils import draw_matches

torch.cuda.empty_cache()
torch.manual_seed(1)

def split_name(name):
    i1_i2, f = name.split('.')
    i1, i2 = i1_i2.split('_') 
    return i1, i2, i1_i2  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default='data/', 
        help=' Input path', metavar='')  
    parser.add_argument('--detector', type=str, nargs='+', default= 'superpoint',  # ['superpoint', 'akaze', 'superpoint_tf', 'kp2d', 'sift'],
        help=' Detector', metavar='')  
    parser.add_argument('--output', type=str, default='output/', 
        help=' Output path', metavar='')
    parser.add_argument('--matches', type=str, default='matches/', 
        help=' Matches path', metavar='') 
    parser.add_argument('--images', type=str, default='images/', 
        help=' Image path', metavar='')
    parser.add_argument('--save_npz', type=bool, default=True, 
        help=' Saving npz', metavar='') 
    parser.add_argument('--draw_matches', type=bool, default=True, 
        help=' Draws matches', metavar='') 
    parser.add_argument('--save_drawn_matches', type=bool, default=True, 
        help=' Saving drawn matches', metavar='')                                                                           
    parser.add_argument('--match_threshold', type=float, default=0.2,
        help=' Match threshold ', metavar='')
    parser.add_argument('--batch_size', type=int, default=1,
        help=' Batch size of training images', metavar='')
    parser.add_argument('--knn', type=int, default=20,
        help=' K nearest neighbour for creating edges', metavar='')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
        help=' Sinkhorn iterations', metavar='')
    parser.add_argument('--aggregation', type=str, nargs='+', default='add',
        help=' Aggregation', metavar='') 
    parser.add_argument('--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    args = parser.parse_args()
    print('\nargs',args)

    # Connecting device to cuda or cpu
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print('\nRunning inference on device \"{}\"'.format(device))

    # Config file
    default_config = {'K': 2, #Chebyshev filter size
                    'GNN_layers': ['cross'], 
                    'match_threshold': args.match_threshold,
                    'sinkhorn_iterations':args.sinkhorn_iterations,
                    'aggr': args.aggregation,
                    'knn': args.knn,
                }


    if args.detector  == 'kp2d' or args.detector  == 'superpoint' or args.detector  == 'superpoint_tf':
        default_config['descriptor_dim'] = 256
        default_config['output_dim'] = 256*2
    if args.detector  == 'sift':
        default_config['descriptor_dim'] = 128
        default_config['output_dim'] = 128*2
    if args.detector  == 'akaze':
        default_config['descriptor_dim'] = 64
        default_config['output_dim'] = 64*2

    # Data processing and Data loader    
    dataset = MyDataset(args.knn, args.input, args.detector , device)
    print(f'Size of Dataset {len(dataset)}')
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Loading the model
    matching_test = SphereGlue(default_config).to(device)              

    model_path = 'saved_model/' + args.detector + '/autosaved.pt'
    ckpt_data = torch.load(model_path)
    matching_test.load_state_dict(ckpt_data["MODEL_STATE_DICT"])
    matching_test.eval()
    print('Loading the model weights for detector ' + str(args.detector ) +  ' from->', model_path)
    

    with torch.no_grad():
        for data, y_true in data_loader:
            # run the model on the given dataset
            y_pred = matching_test(data)
            
            output_data = { 'keypointCoords0': data['keypointCoords0'][0],
                                'keypointCoords1': data['keypointCoords1'][0],
                                'keypointScores0': data['scores1'][0].detach().cpu().numpy(),
                                'keypointScores1': data['scores2'][0].detach().cpu().numpy(),
                                'correspondences': y_pred['matches0'][0].to(torch.float32).detach().cpu().numpy(),
                                'scores': y_pred['matching_scores0'][0].to(torch.float32).detach().cpu().numpy()
                            }

            # Saves npz files with keypoints information and predictions 
            if args.save_npz is True:
                # Output path
                output_path = os.path.join(args.output, args.detector)

                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                numpy.savez(os.path.join(output_path,str(data['name'][0])), **output_data)


            if args.draw_matches is True:
                matches_path = os.path.join(args.matches, args.detector)

                if not os.path.exists(matches_path):
                    os.mkdir(matches_path)

                i1, i2, i1_i2 = split_name(data['name'][0])
                draw_matches(os.path.join(args.images, i1 + '.jpg'), os.path.join(args.images, i1 +'.jpg'), output_data, os.path.join(matches_path, i1_i2), args)

            exit()

    if args.save_npz is True:
         print('Predictions are saved in -> ', args.output)
    if args.draw_matches is True:
         print('Matches are saved in -> ', matches_path)


#-------------------------------------------- End --------------------------------------------#




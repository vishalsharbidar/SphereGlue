import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from model.chebglue import ChebGlue
from utils.demo_mydataset import MyDataset 
import numpy

torch.cuda.empty_cache()
torch.manual_seed(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, nargs='+', default='data/', 
        help=' Input path', metavar='')   
    parser.add_argument('--output', type=str, nargs='+', default='output/', 
        help=' Output path', metavar='')                                                                            
    parser.add_argument('--match_threshold', type=float, default=0.2,
        help=' Match threshold ', metavar='')
    parser.add_argument('--batch_size', type=int, default=1,
        help=' Batch size of training images', metavar='')
    parser.add_argument('--knn', type=int, default=20,
        help=' K nearest neighbour for creating edges', metavar='')
    parser.add_argument('--K', type=int, default=2,
        help=' Chebyshev filter size K', metavar='')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
        help=' Sinkhorn iterations', metavar='')
    parser.add_argument('--GNN_layers', type=str, nargs='+', default=['cross'],
        help=' GNN layers', metavar='') # ['self', 'cross']
    parser.add_argument('--detector', type=str, nargs='+', default= 'akaze',  # ['superpoint', 'akaze', 'superpoint_tf', 'kp2d', 'sift'],
        help=' Detector', metavar='') 
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
    default_config = {'K': args.K, 
                    'match_threshold': args.match_threshold,
                    'GNN_layers':args.GNN_layers, 
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

    ### testing
    # Loading the model
    matching_test = ChebGlue(default_config).to(device)              

    model_path = 'saved_model/' + args.detector + '/autosaved.pt'
    ckpt_data = torch.load(model_path)
    matching_test.load_state_dict(ckpt_data["MODEL_STATE_DICT"])
    matching_test.eval()
    

    print('Loading the model weights for detector ' + str(args.detector ) +  ' from->', model_path)
    

    with torch.no_grad():
        for data, y_true in data_loader:
            # run the model on the given dataset
            y_pred = matching_test(data)
            output = args.output  + str(data['name'][0]) 
            out = {'correspondences': y_pred['matches0'][0].to(torch.float32).detach().cpu().numpy(),
                   'scores': y_pred['matching_scores0'][0].to(torch.float32).detach().cpu().numpy()}

            numpy.savez(output, **out)

    print('Predictions are saved in -> ', args.output)


#-------------------------------------------- End --------------------------------------------#




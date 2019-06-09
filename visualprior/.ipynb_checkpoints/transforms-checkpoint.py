from .taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork, TASKONOMY_PRETRAINED_URLS
import multiprocessing.dummy as mp
import torch


def representation_transform(img, feature_task='normal'):
    return VisualPrior.to_representation(img, feature_tasks=[feature_task])

def multi_representation_transform(img, feature_tasks=['normal']):
    return VisualPrior.to_representation(img, feature_tasks)

def max_coverage_featureset_transform(img, k=4):
    return VisualPrior.max_coverage_transform(img, feature_tasks)


class VisualPrior(object):

    max_coverate_featuresets = [
        ['autoencoding'],
        ['segment_unsup2d', 'segment_unsup25d'],
        ['edge_texture', 'reshading', 'curvature'],
        ['normal', 'keypoints2d', 'segment_unsup2d', 'segment_semantic'],
    ]
    model_dir = None
    viable_feature_tasks = [
             'autoencoding',
             'colorization',
             'curvature',
             'denoising',
             'edge_texture',
             'edge_occlusion',
             'egomotion', 
             'fixated_pose', 
             'jigsaw',
             'keypoints2d',
             'keypoints3d',
             'nonfixated_pose',
             'point_matching', 
             'reshading',
             'depth_zbuffer',
             'depth_euclidean',
             'normal',
             'room_layout',
             'segment_unsup25d',
             'segment_unsup2d',
             'segment_semantic',
             'class_object',
             'class_scene',
             'inpainting',
             'vanishing_point']


    @classmethod
    def to_representation(cls, img, feature_tasks=['normal']):
        '''
            Transforms an RGB image into a feature driven by some vision task(s)
            Expects inputs:
                shape  (batch_size, 3, 256, 256)
                values [-1,1]
            Outputs:
                shape  (batch_size, 8, 16, 16)

            This funciton is technically unsupported and there are absolutely no guarantees. 
        '''
        VisualPriorRepresentation._load_unloaded_nets(feature_tasks)
        nets = [VisualPriorRepresentation.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)

    @classmethod
    def to_predicted_label(cls, img, feature_tasks=['normal']):
        '''
            Transforms an RGB image into a predicted label for some task.
            Expects inputs:
                shape  (batch_size, 3, 256, 256)
                values [-1,1]
            Outputs:
                shape  (batch_size, C, 256, 256)
                values [-1,1]

            This funciton is technically unsupported and there are absolutely no guarantees. 
        '''
        VisualPriorPredictedLabel._load_unloaded_nets(feature_tasks)
        nets = [VisualPriorPredictedLabel.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)
    
    @classmethod
    def max_coverage_transform(cls, img, k=4):
        assert k > 0, 'Number of features to use for the max_coverage_transform must be > 0'
        if k > 4:
            raise NotImplementedError("max_coverage_transform featureset not implemented for k > 4")
        return cls.transform(img, feature_tasks=max_coverate_featuresets[k - 1])

    @classmethod
    def set_model_dir(model_dir):
        cls.model_dir = model_dir


    @classmethod
    def _unload_network(cls, encoder_name):
        pass

    @classmethod
    def _set_network_cache_dir(cls, dict_of_feature_task_to_path):
        for feature_task, path in dict_of_feature_task_to_path.items():
            cls.feature_task_to_path[feature_task] = path

    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, use_decoder=False):
        if use_decoder:
            VisualPriorWithDecoding._load_unloaded_nets(feature_tasks)
        else:
            VisualPriorEncoding._load_unloaded_nets(feature_tasks)


class VisualPriorRepresentation(object):
    '''
        Handles loading networks that transform images into encoded features.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    feature_task_to_net = {}

    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                net_paths_to_load.append(TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder'])
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            cls.feature_task_to_net[feature_task] = net

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None):
        return [cls._load_encoder(url, model_dir) for url in network_paths]

    @classmethod
    def _load_encoder(cls, url, model_dir=None, progress=True):
        net = TaskonomyEncoder() #.cuda()
        net.eval()
        print(url)
        checkpoint = torch.utils.model_zoo.load_url(url, model_dir=model_dir, progress=progress)
        net.load_state_dict(checkpoint['state_dict'])
        for p in net.parameters():
            p.requires_grad = False
        # net = Compose(nn.GroupNorm(32, 32, affine=False), net)
        return net
             


class VisualPriorPredictedLabel(object):
    '''
        Handles loading networks that transform images into transformed images.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, C, 256, 256)
            values [-1,1]
            
        This class is technically unsupported and there are absolutely no guarantees. 
    '''
    feature_task_to_net = {}
    
    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                net_paths_to_load.append((TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder'],
                                          TASKONOMY_PRETRAINED_URLS[feature_task + '_decoder']))
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            print(f"Loading {feature_task}")
            cls.feature_task_to_net[feature_task] = net

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None, progress=True):
        nets = []
        for encoder_path, decoder_path in network_paths:
            nets.append(TaskonomyNetwork(
                    load_encoder_path=encoder_path,
                    load_decoder_path=decoder_path,
                    model_dir=model_dir,
                    progress=progress))
        return nets
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import cox
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from matplotlib.colors import LinearSegmentedColormap
from cox.utils import Parameters, string_to_obj
import numbers
import math

from tqdm.notebook import tqdm
import torch as ch
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from sklearn.metrics import f1_score

from .tools import constants as consts
from .tools.utils import UnNormalize
from .tools.confusion_matrix import plot_confusion_matrix_from_data
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from captum.attr import LayerAttribution, LayerGradCam, Occlusion
from captum.attr import visualization as viz
import os
from robustness import defaults, model_utils, train
from robustness.datasets import CIFAR
from robustness.tools.utils import fix_random_seed


def restore_model(LOGS_PATH, exp_id, model_name, dataset, device, arch='resnet18', checkpoint_type='latest'):
    """
    LOGS_PATH (str): path tot he log file
    exp_id (str): the ID of the experiment
    model_name (str): 
    dataset (robustness.DataSet)
    device (str): 'cpu'
    base_arch (str): 'resnet18'
    checkpoint_type (str): defines the type of the checkpoint to restrieve (`latest` or `best`)
    """
    model, _ = model_utils.make_and_restore_model(
        arch=arch,
        dataset=dataset,
        resume_path=f"{LOGS_PATH}/{exp_id}/checkpoint.pt.{checkpoint_type}",
        device=device)

    # read results
    train_results = None
    if os.path.exists(os.path.join(LOGS_PATH, exp_id, 'train_results.csv')):
        train_results = pd.read_csv(os.path.join(LOGS_PATH, exp_id, 'train_results.csv'))

    test_results = pd.read_csv(os.path.join(LOGS_PATH, exp_id, 'test_results.csv'))
    accuracies = pd.read_csv(os.path.join(LOGS_PATH, exp_id, 'accuracies.csv'))

    return {
        'model': model,
        'name': model_name,
        'exp_id': exp_id,
        'train_results': train_results,
        'test_results': test_results,
        'test_acc': accuracies.test_acc[0]
    }


def compute_accuracies_per_class(all_models, dataset, plot_accuracies=False, x_axis=None):
    """
    Compute the standard accuracy per class for a list of models.

    - all_models (list): list of models, in their dictionary format used in evaluate_model.ipynb
    - dataset (DataSet)
    - plot_accuracies (boolean, optinal): If `True`, plot the accuracies as the adversary power increases
    - x_axis (list): list of adversary power of the models in `all_models`, to plot the accuracy 
            depending on the adversary power (e.g. [0, 0.5, 1, 1.5, 2])
    """
    model_names = []
    for model in all_models:
        model_names.append(model['name'])
    accuracies_per_class = pd.DataFrame({'model': model_names})

    for cls_id, cls_name in dataset.label_mapping.items():
        accuracies = []
        for model in all_models:
            res = model['test_results']
            acc = len(res.loc[(res['y_true'] == cls_id) & (
                res['y_pred'] == cls_id)]) / len(res.loc[res['y_true'] == cls_id])
            accuracies.append(acc)

        accuracies_per_class.insert(
            len(accuracies_per_class.columns), cls_name, accuracies)

    accuracies_per_class['average_acc'] = accuracies_per_class.mean(axis=1)


    if plot_accuracies:
        for cls_name in dataset.label_mapping.values():
            plt.plot(
                x_axis, accuracies_per_class[cls_name], label=f'{cls_name} accuracy')

        plt.xlabel("Power of the adversary")
        plt.ylabel("Standard accuracy per class")
        plt.title("Standard accuracy per class relating to the adversary power")
        plt.legend()
        plt.show()

    return accuracies_per_class

def compute_f1_score(models, f1_type='macro'):
    """
    Compute the F1 score for multiple models

    Args:
    - models (list of dicts): each element is a dictionary model
    - f1_type (str): type of F1 average

    Returns:
    - Pandas series with F1 score for each model
    """
    data, index = [], []
    for model in models:
        f1 = f1_score(model['test_results'].y_true,
                      model['test_results'].y_pred, average=f1_type)
        data.append(f1)
        index.append(model['name'])

    return pd.Series(data, index)


##############    Interpretability    #############


def evaluate_model(exp_id, dataset, train_dataset, test_dataset,
                   LOG_DIR, device='cpu', arch='resnet18', workers=8, batch_size=16,
                   checkpoint_type='latest'):
    """
    Compute accuracy and also Compute results for every sample in test.
    
    test_acc  - standard

    - exp_id (str) - the experiment id (the model's name)
    - dataset (instance of DataSet (from robustness)). Used to instatiate the model.
    - train_dataset (Dataset) - Train dataset.
    - test_dataset (Dataset) - Test dataset.
    - LOG_DIR (str) - path where to load the model and save the accuracies
    """
    fix_random_seed(42)

    #load model
    model, _ = model_utils.make_and_restore_model(
        arch=arch,
        dataset=dataset,
        resume_path=os.path.join(LOG_DIR, exp_id, f'checkpoint.pt.{checkpoint_type}'),
        device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    
    model.eval()
    model.module.only_prediction = True

    # compute predictions
    train_results = get_results(model, train_loader, train_dataset)
    test_results = get_results(model, test_loader, test_dataset)
    train_acc = np.average(train_results['y_pred'] == train_results['y_true']) * 100
    test_acc = np.average(test_results['y_pred'] == test_results['y_true']) * 100

    # save predictions
    train_results.to_csv(os.path.join(LOG_DIR, exp_id, 'train_results.csv'), index=False)
    test_results.to_csv(os.path.join(LOG_DIR, exp_id, 'test_results.csv'), index=False)
    
    # Add accuracies to log
    accs = pd.DataFrame({
        'train_acc': [train_acc],
        'test_acc': [test_acc]
    })
    accs.to_csv(os.path.join(LOG_DIR, exp_id, 'accuracies.csv'), index=False)
    
    return accs

    
def get_results(model, loader, dataset_object, do_tqdm=True):
    """
    **internal function, called from `evaluate_model_to_store`**

    Compute the predictions on a given set. 

    Args:
    - model (DataParallel)
    - loader (torch.DataLoader)

    Return:
        - DataFrame with columns (image_id, y_true, y_pred, probability),
          having one row for each prediction
    """
    y_true_list, y_pred_list, probabilities_true, probabilities_pred = [], [], [], []
    
    iterator = enumerate(loader)
    if do_tqdm:
        iterator = tqdm(iterator, total=len(loader), position=0, leave=True)

    with ch.no_grad():
        for i, (inp, y_true) in iterator:
            y_pred = model(inp)

            softmax = ch.nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()

            y_true = y_true.detach().numpy().tolist()  # transform it to numpy
            y_true_list.extend(y_true)
            probabilities_true.extend(softmax[np.arange(len(y_true)), y_true])

            y_pred_list.extend(np.argmax(softmax, axis=1).tolist())
            probabilities_pred.extend(np.max(softmax, axis=1).tolist())

    res = pd.DataFrame({
        'image_id': list(dataset_object.df.image_id), # because the loader has shuffle=False, and read them in order 
        'y_true': y_true_list,
        'y_pred': y_pred_list,
        'probability_y_pred': probabilities_pred,
        'probability_y_true': probabilities_true
    })
    return res


def plot_top1_predicted_probability(results, labels, cumulative=True):
    g_acc = sns.FacetGrid(results, col='y_true', col_wrap=3)
    g_acc.map(partial(plt.hist, bins=100, range=(0, 1), density=True, cumulative=True), 'probability_y_true')
    g_acc.fig.suptitle('Cumulative distributions for the predicted probability of the true class')
    for label, ax in zip(labels, g_acc.axes):
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Predicted probability", fontsize=12)
    plt.subplots_adjust(top=0.80)
    plt.plot()


def plot_confusion_matrix(results, labels):
    """
    This function is mainly copied from the sklearn docs

    - prediction: DataFrame with columns ('y_true', 'y_pred', 'probability')
    - label: list with ordered labels
    """
    y_true = results['y_true'].to_numpy()
    y_pred = results['y_pred'].to_numpy()

    plot_confusion_matrix_from_data(y_true, y_pred, columns=labels)


def tensor_img_to_numpy(tensor):
    """
    Take a tensor image (C, H, W) and transpose it to (H, W, C)
    """
    return np.transpose(tensor.cpu().detach().numpy(), (1, 2, 0))


def attribute_features(model, algorithm, input, target, **kwargs):
    """
    Simple method to run an attribution algorithm
    """
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=target, **kwargs)

    return tensor_attributions


def compute_smoothness(map, norm="L1"):
    """
    Compute metric to quantify the smoothness of the saliency map. 
    - map (numpy.ndarray)
    - norm (str) in "L1" or "L2"
    """
    map_hor = np.roll(map, shift=1, axis=1)  # shift horizontal
    map_ver = np.roll(map, shift=1, axis=0)  # shift vertical

    # sum the horizontal and vertical differences

    if norm == "L2":
        sum = np.square(map - map_hor) + np.square(map - map_ver)
    elif norm == "L1":
        sum = np.abs(map - map_hor) + np.abs(map - map_ver)
    else:
        raise ValueError(f"norm should be L1 or L2, not {norm}")

    return np.average(sum)


def compute_gradcam(model, input_for_captum, y_true, original_shape):
    layer = model.module.model.layer4
    layer_gc = LayerGradCam(model, layer)
    attr_gradcam = attribute_features(model, layer_gc, input_for_captum, y_true,
                                relu_attributions=True)
    
    return LayerAttribution.interpolate(
        attr_gradcam, original_shape, 'bilinear')


def compute_occlusion(model, input_for_captum, y_true):
    occlusion = Occlusion(model)
    saliency_map = attribute_features(model, occlusion, input_for_captum, y_true, sliding_window_shapes=(
        3, 50, 50), strides=25, baselines=0)
    return saliency_map


def compute_CAM(net, input, y_true, original_shape):
  """
  Compute CAM

  - model (dictionary): use key `model`
  - input (ch.Tensor (1, C, H, W))
  """
  # hook the feature extractor
  net = net.module.model
  if next(net.parameters()).is_cuda:
    device = 'cuda'
  else:
    device = 'cpu'
  features_extracted = []

  def hook_feature(module, input, output):
      features_extracted.append(output.data.cpu().numpy())

  handle = net._modules.get('layer4').register_forward_hook(hook_feature)
  # forward pass
  net(input.to(device))
  handle.remove()

  # get the softmax weight
  params = list(net.parameters())
  weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

  def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample=(448, 448)):
    # generate and upsample CAM
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(1, 1, h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = ch.tensor(cam_img)

    return LayerAttribution.interpolate(cam_img, size_upsample, 'bilinear')

  return returnCAM(features_extracted[0], weight_softmax, y_true)


class GaussianSmoothing(ch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = ch.meshgrid(
            [
                ch.arange(size, dtype=ch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                ch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / ch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """        
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


def plot_saliency(image, y_true, image_id, dataset, models, 
                  saliency_methods,
                  results_type='test_results',
                  title="",
                  saliency_abs=True, viz_sign='absolute_value',
                  sg_stdev_perc=0.2, sg_samples=50, nt_type='smoothgrad_sq',
                  ig_baseline='black', ig_sigma=10,
                  occlusion_window=(3, 50, 50), occlusion_stride=25,
                  outlier_perc=1, plot_saliency_distribution=False,
                  show_prediction_color=True,
                  plot_image = True):
    """
    Plot saliencies as follows:
        First column has only the original image.
        Each row represents a model
        Each columns represents a saliency method.

    Arguments:
    - image (tensor C, H, W) - transformed image
    - y_true - the image's label with which you want the saliency w.r.t (usually the true label)
    - image_id: indice of the image of interest
    - models: list of dictionary {'model', 'name'} for the models to compute the saliency
    - dataset: DataSet object (e.g. class HAM10000)
    - dataset_object: torch.utils.data.Dataset (e.g. class HAM10000_dataset)
    - title (str): title for the main figure
    - results_type(str) in `train_results`, `test_results`: Represents the key in the model dictionary
                            to get its predictions
    - saliency_methods ([str]): list with the saliency methods to plot 
            (supported methods: 'saliency', 'saliency_squared', 'saliency_sg', 'ig', 'gradcam', 'occlusion')
    - nt_type (str): The type of SmoothGrad. Values in ['smoothgrad', 'smoothgrad_sq', 'vargrad']
    - ig_baseline (str): The baseline to use in Integrated Gradients. One of 'black', 'uniform', 'white', 'blur'
    - outlier_perc: percentage of outliers to clip when plotting the saliency maps
    - show_prediction_color: if `True`, then show a red/green backgrounf arounf the image if the predictions is correct/wrong
    - plot_image: if `True`then plot the results image of saliency maps to compare them, else don't plot anything

    Returns:
    - list of np.ndaray of 224x224 - the summed-per-channel values of all saliency maps
    """
    no_rows = len(saliency_methods)
    no_cols = len(models) + 1
    if plot_saliency_distribution:
        no_rows = no_rows + len(saliency_methods)
    fig = plt.figure(figsize=(4*no_cols, 3*no_rows))
    axs = fig.subplots(no_rows, no_cols, squeeze=False)
    saliency_data = list() # list to store the saliecny values distribution

    original_image = tensor_img_to_numpy(image)
    input_for_captum = image.unsqueeze(0)       # (1, C, H, W)
    input_for_captum.requires_grad = True    

    for it, saliency_method in enumerate(saliency_methods):
        axs[it][0].imshow(original_image)
        axs[it][0].set_title("%s: %s\n%s" % (image_id, dataset.label_mapping[y_true], saliency_method))
        axs[it][0].axis('off')
    
        def visualize_map(row, col, saliency, title, **kwargs):
            """
            Plot the saliency map on a given axis

            Attributes:
            - saliency: Tensor of (B, C, H, W), the saliency map
            - pos (int): column position of the plot
            """
            saliency = tensor_img_to_numpy(saliency[0])  # (H, W, C)

            _ = viz.visualize_image_attr(saliency, original_image, plt_fig_axis=(fig, axs[row][col]),
                                            sign=viz_sign,
                                            outlier_perc=outlier_perc,
                                            show_colorbar=show_prediction_color,
                                            title=title,
                                            use_pyplot=False,
                                            **kwargs)

            if plot_saliency_distribution:
                # sum the gradient on the channel
                data = axs[row][col].get_children()[-2]._A.data        
                assert len(data.shape)==2, ("Plotting the saliency distribution expects a 2D saliency map (not in 3D, i.e. RGB)")

                axs[row-1][col].hist(data.ravel(), bins=50)
                axs[row-1][col].set_xlabel("Saliency values")
                axs[row-1][col].set_ylabel("Count")
                axs[row-1][col].set_title("Values distribution") # \n Smooth: L1=%.3f  L2=%0.3f" % (compute_smoothness(data, 'L1'), compute_smoothness(data, 'L2')))

        for j, model_name in enumerate(models):
            model = model_name['model']

            model.eval()
            model.module.only_prediction = True

            # Get prediction
            aux = model_name[results_type]
            result = aux.loc[aux['image_id'] == image_id]
            y_pred = result['y_pred'].iloc[0]
            
            saliency_map = None
            if saliency_method == 'saliency' or saliency_method == 'saliency_squared':
                saliency = Saliency(model)
                saliency_map = attribute_features(model, saliency, input_for_captum, y_true, abs=saliency_abs)  # Tensor of (N, C, H, W)

                if saliency_method == 'saliency_squared':
                  saliency_map = saliency_map**2
            elif saliency_method == 'saliency_sg':
                saliency = Saliency(model) 
                nt = NoiseTunnel(saliency)
                stdev = (ch.max(input_for_captum).item() -
                         ch.min(input_for_captum).item()) * sg_stdev_perc
                saliency_map = attribute_features(model, nt, input_for_captum, y_true,
                                                  nt_type=nt_type, n_samples=sg_samples, stdevs=stdev,
                                                  abs=saliency_abs)
            elif saliency_method == 'ig':
                ig = IntegratedGradients(model)

                if ig_baseline == 'black':
                    baseline = 0
                elif ig_baseline == 'uniform':
                    baseline = ch.rand_like(input_for_captum)
                elif ig_baseline == 'white':
                    white_noise = ch.randn_like(input_for_captum)
                    baseline = ch.clamp(input_for_captum + white_noise, 0, 1)
                elif ig_baseline == 'blur':
                    smoothing = GaussianSmoothing(3, 5, ig_sigma)
                    baseline = smoothing(input_for_captum)
                else:
                    raise ValueError("Invalid value for the parameter `ig_baseline`")

                saliency_map, delta = attribute_features(model, ig, input_for_captum, y_true, baselines=baseline,  # black baseline
                                                    n_steps=50, return_convergence_delta=True)        
            elif saliency_method == 'gradcam':
                saliency_map = compute_gradcam(
                    model, input_for_captum, y_true, (original_image.shape[0], original_image.shape[1]))
            elif saliency_method == 'occlusion':
                occlusion = Occlusion(model)
                saliency_map = attribute_features(model, occlusion, input_for_captum, y_true, sliding_window_shapes=occlusion_window, strides=occlusion_stride, baselines=0)
            elif saliency_method == 'cam':
                saliency_map = compute_CAM(model, input_for_captum, y_true, (original_image.shape[0], original_image.shape[1]))
            else: 
                raise ValueError(f"The saliency method {saliency_method} is not implemented.")
            
            saliency_map_2d = ch.sum(saliency_map.squeeze(0), 0).cpu().data.numpy()
            saliency_data.append(saliency_map_2d)

            visualize_map(it, j+1, saliency_map,
                          model_name['name'] + "\n prob y_true: %.2f" % result['probability_y_true'].iloc[0])

            # Set background to green/red if the prediction is correct/wrong
            if show_prediction_color:
                if y_true != y_pred:
                    color = 'red'
                else:
                    color = 'green'

                bbox = axs[it][j+1].get_position()
                rect = matplotlib.patches.Rectangle((bbox.x0, bbox.y0), bbox.x1 - bbox.x0, bbox.height, color=color, zorder=-1, alpha=0.2)
                fig.add_artist(rect)

    fig.suptitle(title, fontsize=16, y=1.2)    
    # fig.tight_layout()
    if plot_image:
        plt.plot()
    else:
        plt.close()
    
    return saliency_data


def plot_curves_from_file(LOG_DIR, EXP_ID, print_table=False):
    log = cox.store.Store(LOG_DIR, EXP_ID)  # open the log
    log = plot_curves_from_log(log, print_table)
    print(string_to_obj(log.tables['metadata'].df.args[0]))
    log.close() # close the log

    print(pd.read_csv(os.path.join(log.path, 'accuracies_best.csv')))

    return log


def plot_curves_from_log(log, print_table=False):
    print(log.tables)
    metadata = log['metadata'].df

    df = log['logs'].df
    if print_table:
        print(df)

    x = df.epoch.tolist()

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.title("Accuracy")
    plt.plot(x, df.train_prec1.to_list(), label='Train')
    plt.plot(x, df.nat_prec1.to_list(), label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    # plt.ylim(0, 100)
    # plt.yticks(list(np.arange(0, 101, 10)), list(np.arange(0, 101, 10)))

    plt.subplot(122)
    plt.title("Loss")
    plt.plot(x, df.train_loss.to_list(), label='Train')
    plt.plot(x, df.nat_loss.to_list(), label='Validation')
    plt.legend()
    plt.xlabel('Epoch')

    plt.show()

    return log

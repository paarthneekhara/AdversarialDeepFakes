# Import basic required libraries
import foolbox
import torch
import torchvision.models as models
import numpy as np
from foolbox.criteria import TargetClassProbability
from foolbox import criteria
import random 
import os    
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA as RandomizedPCA
import pickle
from skimage import io, transform
from PIL import Image
from sklearn.decomposition import KernelPCA

path = '/home/malhar/Desktop/data/'
# filenames = [line.rstrip('\n') for line in open('filenames.txt')]

# Define helper function 1. get_classes() takes in the filename of an image.
# Input: Image filename 
# Behavior: Read in filename and calculate label of that file
# Output: Return index, class and label number of that image
def get_classes(filenames):
    idxs = []
    for i in range(len(filenames)):
        idxs.append(int(filenames[i].split('_')[2].split('.')[0])-1)
    nlabels = [None] * len(filenames)
    f = open(path+"imagenet_2012_validation_synset_labels.txt", "r")
    for i, line in enumerate(f):
        if i in idxs:
            nlabels[idxs.index(i)] = line.strip()
    classes = [None] * len(filenames)
    f = open(path+"synset_labels_to_words.txt", "r")
    for i, line in enumerate(f):
        ln = line.split(' ')[0]
        if ln in nlabels:
            indx = [index for index, value in enumerate(
                nlabels) if value == ln]
            for x in indx:
                classes[x] = i
    return idxs, classes, nlabels

def convert_numpy_to_torch_for_inference(input_numpy, mean, std):
    # Rescale it
    img_numpy_scaled = np.copy(input_numpy)
    img_numpy_scaled[:,:,0] = (input_numpy[:,:,0] - mean[0]) / std[0]
    img_numpy_scaled[:,:,1] = (input_numpy[:,:,1] - mean[1]) / std[1]
    img_numpy_scaled[:,:,2] = (input_numpy[:,:,2] - mean[2]) / std[2]
    
    
    # Convert it to torch
    img_torch = np.zeros((1,3,224,224))
    img_torch[0,0,:,:] = img_numpy_scaled[:,:,0]
    img_torch[0,1,:,:] = img_numpy_scaled[:,:,1]
    img_torch[0,2,:,:] = img_numpy_scaled[:,:,2]
    
    return torch.from_numpy(img_torch)


def obtain_image_from_index(index_filename, filenames_all):
    filename = filenames_all[index_filename]
    filepath = (path+'/imagenet_validation_images/' + filename)
    image = Image.open(filepath)
    image = image.resize((224,224))
    image = np.asarray(image, dtype=np.float32)
    image /= 255.
    return image

def softmax(x):
    import numpy as np
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def generate_CW_adversarial_sample(label_target, image, fmodel):
    attack = foolbox.attacks.CarliniWagnerL2Attack(
        model=fmodel, 
        criterion=criteria.TargetClass(label_target))
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, label_target)
    return adversarial

def generate_deepfool_adversarial_sample(original_label, image, fmodel):
    attack = foolbox.attacks.DeepFoolAttack(model=fmodel)
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, original_label, steps = 100)
    return adversarial

def generate_randomPGD_adversarial_sample(original_label, image, fmodel):
    attack = foolbox.attacks.CarliniWagnerL2Attack(
        model=fmodel, 
        criterion=criteria.TargetClass(label_target))
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, label_target)
    return adversarial

def generate_GradientAttack_adversarial_sample(original_label, img, fmodel):
    attack = foolbox.attacks.GradientAttack(model=fmodel)
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, original_label, steps = 100)
    return adversarial

def generate_JSMA_adversarial_sample(label_target, image, fmodel):
    print ('Generating attack for ', label_target)
    attack = foolbox.attacks.SaliencyMapAttack(
        model=fmodel, 
        criterion=criteria.TargetClass(label_target))
    print ('Generated attack...')
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, label_target)
    print ('Generated image...', type(adversarial))
    return adversarial

def generate_PGD_no_random_adversarial_sample(label_target, image, fmodel, linf_val):
    print ('Generating PGD no random attack for ', label_target, ' with linf=', linf_val)
    attack = foolbox.attacks.ProjectedGradientDescentAttack(
        model=fmodel, 
        criterion=criteria.TargetClass(label_target),
        distance=foolbox.distances.Linfinity)
    print ('Generated attack...')
    image_for_adversarial = np.transpose(image, (2, 0, 1))
    adversarial = attack(image_for_adversarial, label_target, epsilon=linf_val)
    print ('Generated image...', type(adversarial))
    return adversarial

def generate_PGD_with_random_adversarial_sample(label_target, image, fmodel, linf_val, iters, binary_choice, stepsize_choice):
#     print ('Generating PGD with random attack for %d ' % label_target)
    attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(
        model=fmodel, 
        criterion=criteria.TargetClass(label_target),
        distance=foolbox.distances.Linfinity)
#     print ('Generated attack...')
    image_for_adversarial = np.transpose(image, (2, 0, 1))
#     print (type(image_for_adversarial))
    adversarial = attack(image_for_adversarial, 
                         label_target, 
                         iterations=iters, 
                         epsilon=linf_val, 
                         binary_search=binary_choice,
                        stepsize=stepsize_choice)
#     print ('Generated image...', type(adversarial))
    return adversarial

# Define helper function 2. PCA_transform().
# Input: image, size of image, number of PCA components to transform.
# Output: (dim,dim,3) shaped array
# Problem: This creates an array with image duplicated in a 3x3 grid
def PCA_transform(img, dim, N):
    img_r = np.reshape(img, (dim, dim*3))
    ipca = RandomizedPCA(N).fit(img_r)
    ratio = np.sum(ipca.explained_variance_ratio_)
    img_c = ipca.transform(img_r)
    img_pca = ipca.inverse_transform(img_c)
    img_pca = np.reshape(img_pca, (dim, dim,3))
    return img_pca, ratio

# Problem: This creates an array with image duplicated in a 3x3 grid
def kernel_PCA_transform(img, dim, N):
    # Declare the object
    kernel = KernelPCA(kernel="rbf", n_components=N, fit_inverse_transform=True, gamma=20)
    
    # Reshape 
    img_r = np.reshape(img, (dim, dim*3))
    
    # Fit kernel to data 
    ipca = kernel.fit_transform(img_r)
    explained_variance = np.var(ipca, axis=0)
    ratio = explained_variance / np.sum(explained_variance)
    
    # Perform the inverse transform
    img_pca = kernel.inverse_transform(ipca)
    
    img_pca = np.reshape(img_pca[:], (dim, dim,3))

    return img_pca

# Behavior of adversarial samples: given model name, attack type, and number used, return the top1, top5 and adversarial rpedictions
def behavior_of_adversarial_samples(model_name, attack_name, number_used):
    # Import basic required libraries
    import os 
    import numpy as np
    import re
    
    # Import paths and filenames
    path = attack_name + '/' + model_name + '/'
    filenames = os.listdir(path)
    
    # assertions
    assert number_used <= len(filenames)
    
    
    k_preds_top_1 = []
    k_preds_top_5 = []
    adv_preds = []

    # Iterate over the components
    for k in range(224):

        top_1_correct_preds = 0
        top_5_correct_preds = 0
        adv_correct_preds = 0
        count = 0

        # Iterate over the filenames
        for filename in filenames:
            if (count >= number_used):
                break
            if ('.csv' not in filename):
                continue

            predictions = np.genfromtxt(path + filename, delimiter=',', invalid_raise = False)
            if (predictions.shape != (224,1000)):
                continue

            # Extract the target class and original class of the file from the filename
            numbers = re.findall(r'\d+', filename)
            target = int(numbers[2])
            original = int(numbers[3])

            # Extract the predictions for that particular component
            predictions_k = predictions[k,:]

            # Get top class
            top_class = int(np.argmax(predictions_k))
            top_5_predictions = predictions[-1:][0].argsort()[-5:][::-1]
            
#             print (top_5_predictions)
            print (k, 'Topper:', top_class, ' Target:', target, ' Original:', original, filename)
            count += 1
            if (top_class == target):
                adv_correct_preds += 1
            if (top_class == original):
                top_1_correct_preds += 1
            if (target in top_5_predictions):
                top_5_correct_preds += 1

        print ('---------------------------')
        print ('Component: ', k)
        print ('Component wise top-1 accuracy: ', top_1_correct_preds / (1.0 * count), top_1_correct_preds, count)
#         print ('Component wise top-5 accuracy: ', top_5_correct_preds / (1.0 * count), top_5_correct_preds, count)
        print ('Component wise adversarial accuracy: ', adv_correct_preds / (1.0 * count), adv_correct_preds, count)

        k_preds_top_1.append(top_1_correct_preds / (1.0 * count))
#         k_preds_top_5.append(top_5_correct_preds / (1.0 * count))
        adv_preds.append(adv_correct_preds / (1.0 * count))
        
    return k_preds_top_1, adv_preds, range(224)

from torch import autograd
import torch
import torch.nn as nn
from dataset.transform import xception_default_data_transforms, mesonet_default_data_transforms
import robust_transforms as rt
import random

def predict_with_model(preprocessed_image, model, model_type, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real), output probs, logits
    """
    
    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    if model_type == "xception":
        resized_image = nn.functional.interpolate(preprocessed_image, size = (299, 299), mode = "bilinear", align_corners = True)
        norm_transform = xception_default_data_transforms['normalize']
    elif model_type == "meso":
        resized_image = nn.functional.interpolate(preprocessed_image, size = (256, 256), mode = "bilinear", align_corners = True)
        norm_transform = mesonet_default_data_transforms['normalize']
    
    normalized_image = norm_transform(resized_image)
    
    logits = model(normalized_image)
    output = post_function(logits)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def robust_fgsm(input_img, model, model_type, cuda = True, 
    max_iter = 100, alpha = 1/255.0, 
    eps = 16/255.0, desired_acc = 0.95,
    transform_set = {"gauss_noise", "gauss_blur", "translation", "resize"}
    ):


    def _get_transforms(apply_transforms = {"gauss_noise", "gauss_blur", "translation", "resize"}):
        
        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda = cuda),
            ]

        if "gauss_blur" in apply_transforms:
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size = (5, 5), sigma=(5., 5.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (5, 5), sigma=(10., 10.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (7, 7), sigma=(5., 5.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (7, 7), sigma=(10., 10.), cuda = cuda),
            ]

        if "translation" in apply_transforms:
            transform_list += [
                lambda x: rt.translate_image(x, 10, 10, cuda = cuda),
                lambda x: rt.translate_image(x, 10, -10, cuda = cuda),
                lambda x: rt.translate_image(x, -10, 10, cuda = cuda),
                lambda x: rt.translate_image(x, -10, -10, cuda = cuda),
                lambda x: rt.translate_image(x, 20, 20, cuda = cuda),
                lambda x: rt.translate_image(x, 20, -20, cuda = cuda),
                lambda x: rt.translate_image(x, -20, 10, cuda = cuda),
                lambda x: rt.translate_image(x, -20, -20, cuda = cuda),
            ]

        if "resize" in apply_transforms:
            transform_list += [
                lambda x: rt.compress_decompress(x, 0.1, cuda = cuda),
                lambda x: rt.compress_decompress(x, 0.2, cuda = cuda),
                lambda x: rt.compress_decompress(x, 0.3, cuda = cuda),
            ]

        return transform_list

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    loss_criterion = nn.CrossEntropyLoss()

    while iter_no < max_iter:
        transform_functions = _get_transforms(transform_set)
        
        
        loss = 0

        all_fooled = True
        print ("**** Applying Transforms ****")
        for transform_fn in transform_functions:
            
            transformed_img = transform_fn(input_var)
            prediction, output, logits = predict_with_model(transformed_img, model, model_type, cuda=cuda)

            if output[0][0] < desired_acc:
                all_fooled = False
            loss += torch.clamp( logits[0][1]-logits[0][0] + 10, min = 0.0)
            # loss += loss_criterion(logits, target_var)

        print ("*** Finished Transforms **, all fooled", all_fooled)
        if all_fooled:
            break

        loss /= (1. * len(transform_functions))
        if input_var.grad is not None:
            input_var.grad.data.zero_() # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)
        
        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print ("L infinity norm", l_inf_norm, l_inf_norm * 255.0)
    
    meta_data = {
        'attack_iterations' : iter_no,
        'l_inf_norm' : l_inf_norm,
        'l_inf_norm_255' : round(l_inf_norm * 255.0)
    }

    return input_var, meta_data

def iterative_fgsm(input_img, model, model_type, cuda = True, max_iter = 100, alpha = 1/255.0, eps = 16/255.0, desired_acc = 0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var, model, model_type, cuda=cuda)    
        if (output[0][0] - output[0][1]) > desired_acc:
            break
            
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
        if input_var.grad is not None:
            input_var.grad.data.zero_() # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)
        
        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print ("L infinity norm", l_inf_norm, l_inf_norm * 255.0)
    
    meta_data = {
        'attack_iterations' : iter_no,
        'l_inf_norm' : l_inf_norm,
        'l_inf_norm_255' : round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


def carlini_wagner_attack(input_img, model, model_type, cuda = True, 
    max_attack_iter = 500, alpha = 0.005, 
    const = 1e-3, max_bs_iter = 5, confidence = 20.0):
    
    def torch_arctanh(x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5
        
    attack_w = autograd.Variable(torch_arctanh(input_img.data - 1), requires_grad = True)
    bestl2 = 1e10
    bestscore = -1

    lower_bound_c = 0
    upper_bound_c = 1.0
    bestl2 = 1e10
    bestimg = None
    optimizer = torch.optim.Adam([attack_w], lr=alpha)
    for bsi in range(max_bs_iter):
        for iter_no in range(max_attack_iter):
            adv_image = 0.5 * ( torch.tanh(input_img + attack_w) + 1. )
            prediction, output, logits = predict_with_model(adv_image, model, model_type, cuda=cuda)
            loss1 = torch.clamp( logits[0][1]-logits[0][0] + confidence, min = 0.0)
            loss2 = torch.norm(adv_image - input_img, 2)

            loss_total = loss2 + const * loss1
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if iter_no % 50 == 0:
                print ("BSI {} ITER {}".format(bsi, iter_no), output )
                print ("Losses", loss_total, loss1.data, loss2)


        # binary search for const
        if (logits[0][0] - logits[0][1] > confidence):
            if loss2 < bestl2:
                bestl2 = loss2
                print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Best l2", bestl2)
                bestimg = adv_image.detach().clone().data

            upper_bound_c = min(upper_bound_c, const)
        else:
            lower_bound_c = max(lower_bound_c, const)

        const = (lower_bound_c + upper_bound_c)/2.0

    meta_data = {}
    if bestimg is not None:
        meta_data['l2_norm'] = bestl2.detach().item()
        return bestimg, meta_data
    else:
        meta_data['l2_norm'] = loss2.detach().item()
        return adv_image, meta_data


def black_box_attack(input_img, model, model_type, 
    cuda = True, max_iter = 100, alpha = 1/255.0, 
    eps = 16/255.0, desired_acc = 0.90, 
    transform_set = {"gauss_blur", "translation"}):

    def _get_transforms(apply_transforms = {"gauss_noise", "gauss_blur", "translation", "resize"}):
        
        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda = cuda),
            ]

        if "gauss_blur" in apply_transforms:
            kernel_size = random.randint(3,6)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size - 1
            sigma = random.randint(5, 7)
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size = (kernel_size, kernel_size), sigma=(sigma * 1., sigma * 1.), cuda = cuda)
            ]

        if "translation" in apply_transforms:
            x_translate = random.randint(-20,20)
            y_translate = random.randint(-20,20)
            
            transform_list += [
                lambda x: rt.translate_image(x, x_translate, y_translate, cuda = cuda),
            ]

        if "resize" in apply_transforms:
            compression_factor = random.randint(4, 6)/10.0
            transform_list += [
                lambda x: rt.compress_decompress(x, compression_factor, cuda = cuda),
            ]

        return transform_list

    def _find_nes_gradient(input_var, transform_functions, model, model_type, num_samples = 20, sigma = 0.001):
        g = 0
        _num_queries = 0
        for sample_no in range(num_samples):
            for transform_func in transform_functions:
                rand_noise = torch.randn_like(input_var)
                img1 = input_var + sigma * rand_noise
                img2 = input_var - sigma * rand_noise

                prediction1, probs_1, _ = predict_with_model(transform_func(img1), model, model_type, cuda=cuda)

                prediction2, probs_2, _ = predict_with_model(transform_func(img2), model, model_type, cuda=cuda)

                _num_queries += 2
                g = g + probs_1[0][0] * rand_noise
                g = g - probs_2[0][0] * rand_noise
                g = g.data.detach()

                del rand_noise
                del img1
                del prediction1, probs_1
                del prediction2, probs_2

        return (1./(2. * num_samples * len(transform_functions) * sigma)) * g, _num_queries

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    # give it a warm start by crafting by fooling without any transformations -> easier
    warm_start_done = False
    num_queries = 0
    while iter_no < max_iter:

        if not warm_start_done:
            _, output, _= predict_with_model(input_var, model, model_type, cuda=cuda)
            num_queries += 1
            if output[0][0] > desired_acc:
                warm_start_done = True

        if warm_start_done:
            # choose all transform functions
            transform_functions = _get_transforms(transform_set)
        else:
            transform_functions = _get_transforms({}) # returns identity function

        all_fooled = True
        print ("Testing transformation outputs", iter_no)
        for transform_fn in transform_functions:
            _, output, _= predict_with_model(transform_fn(input_var), model, model_type, cuda=cuda)
            num_queries += 1
            print (output)
            if output[0][0] < desired_acc:
                all_fooled = False
        
        print("All transforms fooled:", all_fooled, "Warm start done:", warm_start_done)
        if warm_start_done and all_fooled:
            break
        
        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var, transform_functions, model, model_type)
        num_queries += _num_grad_calc_queries
        step_adv = input_var.detach() + alpha * torch.sign(step_gradient_estimate.data.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)
        
        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print ("L infinity norm", l_inf_norm, l_inf_norm * 255.0)
    
    meta_data = {
        'num_network_queries' : num_queries,
        'attack_iterations' : iter_no,
        'l_inf_norm' : l_inf_norm,
        'l_inf_norm_255' : round(l_inf_norm * 255.0)
    }

    return input_var, meta_data

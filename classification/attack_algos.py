from torch import autograd
import torch
import torch.nn as nn
from dataset.transform import xception_default_data_transforms
import robust_transforms as rt

def predict_with_model(preprocessed_image, model, post_function=nn.Softmax(dim=1), cuda=True):
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
    resized_image = nn.functional.interpolate(preprocessed_image, size = (299, 299), mode = "bilinear", align_corners = True)
    norm_transform = xception_default_data_transforms['normalize']
    normalized_image = norm_transform(resized_image)
    
    logits = model(normalized_image)
    output = post_function(logits)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())
    print ("prediction", prediction)
    print ("output", output)
    return int(prediction), output, logits


def robust_fgsm(input_img, model, cuda = True, max_iter = 100, alpha = 1/255.0, eps = 16/255.0, desired_acc = 0.7):


    def _get_transforms():
        return [
            lambda x: x,
            lambda x: rt.add_gaussian_noise(x, 0.01, cuda = cuda),
            # lambda x: rt.add_gaussian_noise(x, 0.02, cuda = cuda),
            lambda x: rt.gaussian_blur(x, kernel_size = (7, 7), sigma=(5.0, 5.0), cuda = cuda),
            lambda x: rt.gaussian_blur(x, kernel_size = (5, 5), sigma=(10.5, 10.5), cuda = cuda),
            # lambda x: rt.gaussian_blur(x, kernel_size = (11, 11), sigma=(10.5, 10.5), cuda = cuda),
            # lambda x: rt.gaussian_blur(x, kernel_size = (11, 11), sigma=(5.0, 5.0), cuda = cuda),
            lambda x: rt.translate_image(x, 10, 10, cuda = cuda),
            lambda x: rt.translate_image(x, 10, -10, cuda = cuda),
            lambda x: rt.translate_image(x, -10, 10, cuda = cuda),
            lambda x: rt.translate_image(x, -10, -10, cuda = cuda),
            lambda x: rt.translate_image(x, 20, 20, cuda = cuda),
            lambda x: rt.translate_image(x, 20, -20, cuda = cuda),
            lambda x: rt.translate_image(x, -20, 10, cuda = cuda),
            lambda x: rt.translate_image(x, -20, -20, cuda = cuda),
            lambda x: rt.compress_decompress(x, 0.1, cuda = cuda),
            lambda x: rt.compress_decompress(x, 0.2, cuda = cuda),
            lambda x: rt.compress_decompress(x, 0.3, cuda = cuda),
        ]

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    loss_criterion = nn.CrossEntropyLoss()

    while iter_no < max_iter:
        transforms = _get_transforms()
        loss = 0

        all_fooled = True
        print ("**** Applying Transforms ****")
        for transform_fn in transforms:
            transformed_img = transform_fn(input_var)
            prediction, output, logits = predict_with_model(transformed_img, model, cuda=cuda)

            if output[0][0] < desired_acc:
                all_fooled = False
            loss += torch.clamp( logits[0][1]-logits[0][0] + 10, min = 0.0)
            # loss += loss_criterion(logits, target_var)

        print ("*** Finished Transforms **, all fooled", all_fooled)
        if all_fooled:
            break

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

def iterative_fgsm(input_img, model, cuda = True, max_iter = 100, alpha = 1/255.0, eps = 16/255.0, desired_acc = 0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var, model, cuda=cuda)    
        if (output[0][0] - output[0][1]) > desired_acc:
            break
            
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
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


def carlini_wagner_attack(input_img, model, cuda = True, 
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
            prediction, output, logits = predict_with_model(adv_image, model, cuda=cuda)
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



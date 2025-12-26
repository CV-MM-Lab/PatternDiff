import torch
import torchvision.transforms.functional as F

def compose_img(gt_img, fake_img, im_parse):
    # fake_img = fake_img.cpu()

    seg_head = torch.logical_or(im_parse == 1, im_parse == 2)
    seg_head = torch.logical_or(seg_head, im_parse == 4)
    seg_head = torch.logical_or(seg_head, im_parse == 13)

    true_head = gt_img * seg_head
    true_parts = true_head
    
    generated_body = (F.pil_to_tensor(fake_img) / 255) * (~(seg_head))

    return true_parts + generated_body

def compose_img_cpu(gt_img, fake_img, im_parse):
    # fake_img = fake_img.cpu()

    seg_head = torch.logical_or(im_parse == 1, im_parse == 2)
    #seg_head = torch.logical_or(im_parse == 11, im_parse == 1)
    seg_head = torch.logical_or(seg_head, im_parse == 4)
    seg_head = torch.logical_or(seg_head, im_parse == 13)

    true_head = gt_img * seg_head
    true_parts = true_head
    
    generated_body = (F.pil_to_tensor(fake_img) / 255) * (~(seg_head))

    return true_parts + generated_body




def compose_dresscode_cpu(gt_img, fake_img, im_parse):

    seg_head = torch.logical_or(im_parse == 1, im_parse == 11)
    #seg_head = torch.logical_or(seg_head, im_parse == 11)
    seg_head = torch.logical_or(seg_head, im_parse == 12)
    seg_head = torch.logical_or(seg_head, im_parse == 13)

    true_head = gt_img * seg_head
    true_parts = true_head
    
    generated_body = (F.pil_to_tensor(fake_img) / 255) * (~(seg_head))

    return true_parts + generated_body

def compose_img_dresscode(gt_img, fake_img, im_head):

    seg_head = im_head
    true_head = gt_img * seg_head
    generated_body = fake_img * ~(seg_head)

    return true_head + generated_body 
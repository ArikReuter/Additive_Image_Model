import torch
from dataloaders import CustomDataset, CustomDataset_img_features
from metrics import mad_explained, var_exp_score, coef_det

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test_split_images(images, targets, train_frac, val_frac, batch_size, transforms_train, transforms_val_test):
    tot_len = len(images)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_img = images[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_img = images[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_img = images[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = CustomDataset(train_img, train_y, transforms = transforms_train)
    val_set = CustomDataset(val_img, val_y, transforms = transforms_val_test)
    test_set = CustomDataset(test_img, test_y, transforms = transforms_val_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_test_split_features(features, targets, train_frac, val_frac, batch_size):
    tot_len = len(features)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_features = features[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_features = features[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_features = features[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = list(zip(train_features, train_y))
    val_set = list(zip(val_features, val_y))
    test_set = list(zip(test_features, test_y))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_test_split_features_images(features, images, targets, train_frac, val_frac, batch_size, transforms_train, transforms_val_test):
    tot_len = len(features)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_features = features[:train_max_idx]
    train_images = images[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_features = features[train_max_idx:val_max_idx]
    val_images = images[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_features = features[val_max_idx:]
    test_images = images[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = CustomDataset_img_features(train_images, train_features, train_y, transforms = transforms_train)
    val_set = CustomDataset_img_features(val_images, val_features, val_y, transforms = transforms_val_test)
    test_set = CustomDataset_img_features(test_images, test_features, test_y, transforms = transforms_val_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader



import numpy as np
import time
from tqdm import tqdm
#Validation function

def validate(model, dataloader, loss_fun):
    val_loss_lis = []

    target_lis = []
    pred_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

          x, f, y = batch
          x = x.to(device)
          f = f.to(device)
          y = y.to(device)
          pred = model(x, f)
          pred = pred.squeeze(-1)
          loss = loss_fun(pred, y)
          val_loss_lis.append(loss.cpu().detach())

          target_lis.append(y.detach().cpu())
          pred_lis.append(pred.detach().cpu())

    mean_loss = np.mean(np.array(val_loss_lis))
    median_loss = np.median(np.array(val_loss_lis))

    target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
    var_exp = var_exp_score(pred_ten, target_ten)
    mad_exp = mad_explained(pred_ten, target_ten)
    r_score = coef_det(pred_ten, target_ten)
    return mean_loss, median_loss, var_exp, mad_exp, r_score



# Training function
def train_loop(model, optimizer, loss_fun, trainset, val_loader, print_mod, device, n_epochs, save_path = None, early_stopping = True, n_epochs_early_stopping = 5):
    """
    train the model
    Args:
        model: The model to train
        optimizer: The used optimizer
        loss_fun: The used loss function
        trainset: The dataset to train on
        valset: The dataset to use for validation
        print_mod: Number of epochs to print result after
        device: Either "cpu" or "cuda"
        n_epochs: Number of epochs to train
        save_path: Path to save the model's state dict
        config: config file from the model to train
        sparse_ten (bool): if a sparse tensor is used for each batch
    """
    if early_stopping == True:
      n_early_stopping = n_epochs_early_stopping
      past_val_losses = []

    loss_lis = []
    target_lis = []
    pred_lis = []

    loss_lis_all = []
    val_loss_lis_all = []

    model = model.to(device)

    model.train()
    for epoch in range(n_epochs):
      start = time.time()
      for iter, batch in enumerate(tqdm(trainset)):

        x, f, y = batch
        x = x.to(device)
        f = f.to(device)
        y = y.to(device)
        pred = model(x, f)
        pred = pred.squeeze(-1)


        loss = loss_fun(pred, y)
        #print(loss)

        optimizer.zero_grad()       # clear previous gradients
        loss.backward()             # backprop

        optimizer.step()

        loss_lis.append(loss.cpu().detach())
        target_lis.append(y.detach().cpu())
        pred_lis.append(pred.detach().cpu())

      if epoch % print_mod == 0:

        end = time.time()
        time_delta = end - start

        mean_loss = np.mean(np.array(loss_lis))
        median_loss = np.median(np.array(loss_lis))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
        var_exp = var_exp_score(pred_ten, target_ten)
        mad_exp = mad_explained(pred_ten, target_ten)
        r_score = coef_det(pred_ten, target_ten)

        target_lis = []
        pred_lis = []



        loss_lis_all += loss_lis

        loss_lis = []



        mean_loss_val, median_loss_val, var_exp_val, mad_exp_val, val_r_score = validate(model, val_loader, loss_fun = loss_fun)

        val_loss_lis_all.append(mean_loss_val)



        print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, median_train_loss = {median_loss}, train_var_exp = {var_exp}, train_mad_exp = {mad_exp}, train_r = {r_score}, elapsed time: {time_delta}')
        print(f'Epoch nr {epoch}: mean_valid_loss = {mean_loss_val}, median_valid_loss = {median_loss_val}, valid_var_exp = {var_exp_val}, valid_mad_exp = {mad_exp_val},  valid_r = {val_r_score}')



        # early stopping based on median validation loss:
        if early_stopping:
          if len(past_val_losses) == 0 or mean_loss_val < min(past_val_losses):
            print("save model")
            torch.save(model.state_dict(), save_path)

          if len(past_val_losses) >= n_early_stopping:
            if mean_loss_val > max(past_val_losses):
              print(f"Early stopping because the median validation loss has not decreased since the last {n_early_stopping} epochs")
              return loss_lis_all, val_loss_lis_all
            else:
              past_val_losses = past_val_losses[1:] + [mean_loss_val]
          else:
            past_val_losses = past_val_losses + [mean_loss_val]



    return loss_lis_all, val_loss_lis_all
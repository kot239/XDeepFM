import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def get_input_batch(batch):
    return [
            torch.stack(batch['city']).t().float().to(device), 
            torch.stack(batch['bd']).t().float().to(device), 
            torch.stack(batch['gender']).t().float().to(device), 
            torch.stack(batch['registered_via']).t().float().to(device), 
            torch.stack(batch['language']).t().float().to(device),
            torch.stack(batch['source_system_tab']).t().float().to(device), 
            torch.stack(batch['source_screen_name']).t().float().to(device), 
            torch.stack(batch['source_type']).t().float().to(device),  
            torch.stack(batch['song_year']).t().float().to(device), 
            torch.stack(batch['song_country']).t().float().to(device), 
            torch.stack(batch['genre_ids']).t().float().to(device),
            batch['song_length'][0].float().unsqueeze(0).t().to(device),
            batch['membership_days'][0].float().unsqueeze(0).t().to(device),
            batch['number_of_time_played'][0].float().unsqueeze(0).t().to(device),
            batch['user_activity'][0].float().unsqueeze(0).t().to(device),
            batch['lyricists_count'][0].float().unsqueeze(0).t().to(device),
            batch['composer_count'][0].float().unsqueeze(0).t().to(device),
            batch['artist_count'][0].float().unsqueeze(0).t().to(device),
            batch['number_of_genres'][0].float().unsqueeze(0).t().to(device),
            batch['artist_composer_lyricist'][0].float().unsqueeze(0).t().to(device),
            batch['is_featured'][0].float().unsqueeze(0).t().to(device),
            batch['artist_composer'][0].float().unsqueeze(0).t().to(device)
        ]


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0.
    outputs = []
    targets = []

    for i, batch in tqdm(enumerate(data_loader), desc='Train'):
        output = model(get_input_batch(batch)).reshape(-1)

        outputs = outputs + output.tolist()
        targets = targets + batch['target'][0].tolist()

        loss = model.loss(output, batch['target'][0].float()).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    auc = auc_score(targets, outputs)
    return total_loss, auc


def evaluate(model, data_loader, optimizer):
    model.eval()
    total_loss = 0.
    outputs = []
    targets = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), desc='Evaluate'):
            output = model(get_input_batch(batch)).reshape(-1)

            outputs = outputs + output.tolist()
            targets = targets + batch['target'][0].tolist()

            loss = model.loss(output, batch['target'][0].float()).to(device)
            total_loss += loss.item()
            
    auc = auc_score(targets, outputs)
    return total_loss, auc


def train_model(model, optimizer, epochs=10):
    train_auc = []
    val_auc = []
    train_loss = []
    val_loss = []
    for e in range(epochs):
        t_loss, t_auc = train(model, train_loader, optimizer)
        train_auc.append(t_auc)
        train_loss.append(t_loss)

        e_loss, e_auc = evaluate(model, test_loader, optimizer)
        val_auc.append(e_auc)
        val_loss.append(e_loss)
        
        if e % 1 == 0:
            print()
            print('-' * 99)
            print(f'| end of epoch {(e + 1):3d} | '
                f'valid loss {e_loss:5.3f} | train loss {t_loss:5.3f} |'
                f'valid auc {e_auc:5.3f} | train auc {t_auc:5.3f} |')
            print('-' * 99)
    
    return train_auc, val_auc, train_loss, val_loss


def plot_training_results(train_loss, val_loss, train_auc, val_auc):
    x = [x for x in range(1, len(train_loss) + 1)]
    plt.figure(figsize=[16, 12])
    plt.subplot(2, 2, 1)
    plt.plot(x, train_loss)
    plt.title('Train loss')
    plt.subplot(2, 2, 2)
    plt.plot(x, val_loss)
    plt.title('Val loss')
    plt.subplot(2, 2, 3)
    plt.plot(x, train_auc)
    plt.title('Train AUC score')
    plt.subplot(2, 2, 4)
    plt.plot(x, val_auc)
    plt.title('Val AUC score')
    # plt.savefig(plots)
    plt.show()

import torch
from torch.utils.data import DataLoader

from utils import get_train_and_test_df, get_input_tensor, KKBoxDataset, train_model, device, plot_training_results
from models import CinModel, DnnModel, XDeepFMModel

if __name__ == '__main__':

    train_subs, test_subs = get_train_and_test_df()

    X_train, X_test, y_train, y_test, features_name = get_input_tensor(train_subs)

    features = [(f, len(x)) for f, x in zip(features_name, X_train[0])]

    train_dataset = KKBoxDataset(X_train, y_train)
    test_dataset = KKBoxDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)


    # train cin
    cin_model = CinModel().to(device)
    cin_optimizer = torch.optim.Adam(cin_model.parameters(), lr=0.001)
    train_auc_cin, val_auc_cin, train_loss_cin, val_loss_cin = train_model(cin_model, cin_optimizer)
    plot_training_results(train_auc_cin, val_auc_cin, train_loss_cin, val_loss_cin)

    # train dnn
    dnn_model = DnnModel().to(device)
    dnn_optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)
    train_auc_dnn, val_auc_dnn, train_loss_dnn, val_loss_dnn = train_model(dnn_model, dnn_optimizer)
    plot_training_results(train_auc_dnn, val_auc_dnn, train_loss_dnn, val_loss_dnn)

    # train xdeepfm
    xdeepfm_model = XDeepFMModel().to(device)
    xdeepfm_optimizer = torch.optim.Adam(xdeepfm_model.parameters(), lr=0.001)
    train_auc, val_auc, train_loss, val_loss = train_model(xdeepfm_model, xdeepfm_optimizer)
    plot_training_results(train_auc, val_auc, train_loss, val_loss)

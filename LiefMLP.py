import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LiefMLP(nn.Module):
    def __init__(self):
        super(LiefMLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2381),
            
            nn.Linear(2381, 512),
            nn.Tanh(),

            nn.Linear(512, 128),
            nn.Tanh(),

            nn.BatchNorm1d(128),

            nn.Linear(128, 8),
            nn.Tanh(),

            nn.Linear(8, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, input):
        x = self.classifier(input)
        return x
    

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved in ", path)

def load_model(path):
    model = LiefMLP()
    model.load_state_dict(torch.load(path))
    print("Model loaded")
    return model

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs, show_validation_metrics = False, early_stopping = None, model_path = None, plots_path = None):
    # model in training mode

    min_valid_loss = np.inf
    train_losses = []
    valid_losses = []
    criterion = criterion.to(device)
    model = model.to(device)

    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):   
            optimizer.zero_grad()

            data = batch[0].to(device)
            target = batch[1].long().to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()
        predictions = []
        actual_labels = []

        for batch in tqdm(valid_loader):  
            data = batch[0].to(device)
            target = batch[1].long().to(device)
            output = model(data)
            _, preds = torch.max(output, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(target.cpu().tolist())
            loss = criterion(output, target)
            valid_loss += loss.item()

        # calculate average losses
        valid_loss = valid_loss / len(valid_loader)
        train_loss = train_loss / len(train_loader)

        print('Epoch: {} \tTraining Loss: {:.8f} \t Validation Loss: {:.8f}'.format(epoch, train_loss, valid_loss))
        
        if show_validation_metrics:
            clf_report = classification_report(actual_labels, predictions, digits=4, output_dict=True) 
            print("Accuracy Score: ", accuracy_score(actual_labels, predictions), 
                "\tPrecision: ", clf_report["macro avg"]["precision"],
                "\tRecall:", clf_report["macro avg"]["recall"],
                "\tF1: ", clf_report["macro avg"]["f1-score"])
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < min_valid_loss:
            print("Validation Loss Decreased, saving the model.")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        try:
            if early_stopping(valid_loss):
                print("Early Stopping.")
                break
        except:
            pass

        if scheduler is not None:
            try:
                scheduler.step(valid_loss)
            except:
                scheduler.step()

    # model in evaluation mode
    model.eval()

    
    x = range(1, epoch + 1)
    plt.plot(x, train_losses, label='train_loss')
    plt.plot(x, valid_losses, label='val_loss')
    plt.xticks(x)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if plots_path is not None:
        plt.savefig(os.path.join(plots_path, "losses.png"))
    plt.show()
    plt.clf()

    return model


def test(model, test_loader, criterion, plots_path = None, roc_plot = True, cm_plot = True):

    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    scores=torch.zeros(0,dtype=torch.long, device='cpu')
    criterion = criterion.to(device)
    model = model.to(device)

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0

    # model in evaluation mode
    model.eval()
    for batch in tqdm(test_loader):
        
        data = batch[0].to(device)
        target = batch[1].long().to(device)
        output = model(data)

        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        scores=torch.cat([scores, output.cpu().detach()])
        predlist=torch.cat([predlist, pred.view(-1).cpu()])
        lbllist=torch.cat([lbllist, target.view(-1).cpu()])

    scores = scores.T.numpy()
    ben_scores = scores[0][:]
    mal_scores = scores[1][:]

    b_fpr, b_tpr, _ = metrics.roc_curve(y_true=lbllist.numpy(), y_score=ben_scores, pos_label=0)
    m_fpr, m_tpr, _ = metrics.roc_curve(y_true=lbllist.numpy(), y_score=mal_scores, pos_label=1)
    b_auc = metrics.auc(b_fpr, b_tpr)
    m_auc = metrics.auc(m_fpr, m_tpr)
    confusion_matrix = metrics.confusion_matrix(lbllist.numpy(), predlist.numpy())

    
    if roc_plot:
        plt.title('Receiver Operating Characteristic')
        plt.plot(b_fpr, b_tpr, color='steelblue', label = 'Goodware AUC = %0.6f' % b_auc)
        plt.plot(m_fpr, m_tpr, color='orange', label = 'Malware AUC = %0.6f' % m_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if plots_path is not None:
            plt.savefig(os.path.join(plots_path, "ROC.png"))
            print("ROC Curve Plot saved")
        plt.show()
        plt.clf()
    if cm_plot:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Goodware", "Malware"])
        cm_display.plot()
        if plots_path is not None:
            plt.savefig(os.path.join(plots_path, "confusion_matrix.png"))
            print("Confusion Matrix Plot saved")
        plt.show()
        plt.clf()
        
    
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    prediction_report = {
        "correct" : accuracy_score(lbllist, predlist, normalize=False),
        "total": len(predlist),
        "accuracy" : float(accuracy_score(lbllist, predlist, normalize = False) / len(predlist)),
        "conf_mx" : confusion_matrix
    }

    print("Predicted %d out of %d samples. Accuracy: %.4f" % (prediction_report["correct"], prediction_report["total"], prediction_report["accuracy"]))
    print(prediction_report["conf_mx"])

    return prediction_report, classification_report(lbllist, predlist, digits=4, output_dict=True) 
